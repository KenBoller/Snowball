# advanced_logger.py
"""
AdvancedSnowballLogger — robust, non-blocking logger with:
- Per-category rotating file logs
- Optional console output
- Optional Azure Cosmos DB sink (fully guarded; never blocks)
- Thread-safe, with a single background worker for cloud writes
- API compatible with your project’s calls

Drop-in usage:
    from advanced_logger import AdvancedSnowballLogger as SnowballLogger
    logger = SnowballLogger()
    logger.log_event("System", "Startup", "Snowball AI initialized.")
"""

import os
import sys
import json
import uuid
import time
import queue
import logging
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Any, Tuple

# --------- Optional Azure Cosmos DB (guarded) ----------
_HAS_COSMOS = True
try:
    from azure.cosmos import CosmosClient, exceptions  # type: ignore
    from azure.cosmos.partition_key import PartitionKey  # type: ignore
except Exception:
    _HAS_COSMOS = False
    CosmosClient = None  # type: ignore
    exceptions = None  # type: ignore
    PartitionKey = None  # type: ignore

# --------- Config ----------
DEFAULT_LOG_DIR = "S:/Snowball/storage/logs"
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

VALID_LOG_CATEGORIES = {
    "memory", "system", "ai", "decisionmaker", "chatagent", "config",
    "error", "event", "interaction", "security", "systemhealth", "task",
    "warning", "vision"
}

# Thread-safety for rotation on Windows
_FILE_LOCK = threading.Lock()


def load_cosmosdb_credentials(paths=None) -> Tuple[Optional[str], Optional[str]]:
    """
    Load CosmosDB (uri, key) from env or common config files.
    Priority: ENV > account_integrations.json > api_keys.json
    """
    uri = os.getenv("COSMOSDB_URI")
    key = os.getenv("COSMOSDB_KEY")
    paths = paths or [
        "S:/Snowball/config/account_integrations.json",
        "S:/Snowball/config/api_keys.json",
    ]
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            src = data.get("api_keys", data)
            uri = uri or src.get("cosmosdb_uri")
            key = key or src.get("cosmosdb_key")
        except Exception:
            continue
    return (uri.strip() if uri else None, key.strip() if key else None)


class SafeRotatingFileHandler(RotatingFileHandler):
    """Rotating file handler with a lock to avoid Windows rename/permission hiccups."""
    def __init__(self, filename, max_bytes=5 * 1024 * 1024, backup_count=10):
        super().__init__(filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")

    def doRollover(self):
        with _FILE_LOCK:
            try:
                super().doRollover()
            except PermissionError:
                time.sleep(0.5)
                try:
                    super().doRollover()
                except Exception:
                    # If still failing, silently skip rotation to avoid crashing the app.
                    pass


class AdvancedSnowballLogger:
    """
    Robust logger with:
      - File logs per category
      - Optional console logs
      - Optional CosmosDB sink via a single background worker (never blocks)
    API methods:
      log_event(category, event_type, message, severity="INFO")
      log_error(category, message)
      log_warning(category, message)
      log_decision(details)
      log_memory(details)
      log_interaction(user_message, ai_response)
      log_security(message)
      log_system_health(metrics)
      log_task(task_name, status)
      shutdown()
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # Singleton-ish to avoid duplicate handlers on repeated construction
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        log_dir: str = DEFAULT_LOG_DIR,
        base_logger_name: str = "Snowball",
        use_console: bool = True,
        enable_cosmos: bool = True,
        cosmos_db_name: str = "snowballDB",
        cosmos_container_name: str = "logsContainer",
    ):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Base logger (for generic info)
        self._base = logging.getLogger(base_logger_name)
        self._base.setLevel(logging.INFO)
        self._ensure_handler(self._base, os.path.join(self.log_dir, "snowball.log"), console=use_console)

        # Per-category loggers
        self.loggers: Dict[str, logging.Logger] = {}
        for cat in VALID_LOG_CATEGORIES:
            self._get_category_logger(cat)

        # Cosmos
        self.enable_cosmos = enable_cosmos and _HAS_COSMOS
        self.cosmos_db_name = cosmos_db_name
        self.cosmos_container_name = cosmos_container_name
        self.cosmos_uri, self.cosmos_key = load_cosmosdb_credentials()
        self._cosmos_client = None
        self._cosmos_container = None

        # Async queue for cloud writes (so logging never blocks)
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=10000)
        self._stop_evt = threading.Event()
        self._worker: Optional[threading.Thread] = None

        if self.enable_cosmos and self.cosmos_uri and self.cosmos_key:
            self._start_cosmos_worker()
        else:
            self._base.info("Cosmos logging disabled or credentials missing; local logs only.")

        self._base.info("✅ AdvancedSnowballLogger initialized.")

    # ---------- Setup helpers ----------
    def _ensure_handler(self, logger: logging.Logger, file_path: str, console: bool = False):
        if logger.handlers:
            return
        file_handler = SafeRotatingFileHandler(file_path, max_bytes=1_000_000, backup_count=5)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)
        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(ch)

    def _get_category_logger(self, category: str) -> logging.Logger:
        key = category.lower().strip()
        if key not in self.loggers:
            path = os.path.join(self.log_dir, f"{key}.log")
            lg = logging.getLogger(f"Snowball.{key}")
            lg.setLevel(logging.INFO)
            self._ensure_handler(lg, path, console=False)
            self.loggers[key] = lg
        return self.loggers[key]

    # ---------- Cosmos worker ----------
    def _start_cosmos_worker(self):
        try:
            self._cosmos_client = CosmosClient(self.cosmos_uri, credential=self.cosmos_key)  # type: ignore
            db = self._cosmos_client.create_database_if_not_exists(id=self.cosmos_db_name)
            self._cosmos_container = db.create_container_if_not_exists(
                id=self.cosmos_container_name,
                partition_key=PartitionKey(path="/category"),  # type: ignore
                offer_throughput=400,
            )
            self._worker = threading.Thread(target=self._cosmos_loop, name="SnowballLogWorker", daemon=True)
            self._worker.start()
            self._base.info("✅ Cosmos logging worker started.")
        except Exception as e:
            self._base.warning(f"⚠️ Failed to start Cosmos logging; falling back to local only. Error: {e}")
            self.enable_cosmos = False
            self._cosmos_client = None
            self._cosmos_container = None

    def _cosmos_loop(self):
        # Pulls entries from queue and writes to cosmos; never raises to main thread
        while not self._stop_evt.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self._cosmos_container is not None:
                    self._cosmos_container.create_item(item)  # type: ignore
            except Exception as e:
                # Log locally if cloud write fails; do not re-queue to avoid spin
                self._base.warning(f"Cosmos write failed: {e}")
            finally:
                self._q.task_done()

    # ---------- Public API ----------
    def log_event(self, category: str, event_type: str, message: str, severity: str = "INFO"):
        """
        Main logging call.
        category: free text; normalized to a file logger key
        event_type: sub-type (e.g., "Startup", "Query")
        message: details
        severity: INFO | WARNING | ERROR
        """
        key = (category or "event").lower().strip()
        if key not in self.loggers:
            # if unknown, fall back to 'event' file logger but keep category in the payload
            key = "event"

        line = f"[{category}] {event_type} — {message}"
        lg = self.loggers[key]

        sev = (severity or "INFO").upper()
        if   sev == "ERROR":   lg.error(line)
        elif sev == "WARNING": lg.warning(line)
        else:                  lg.info(line)

        # Also echo to base logger for a unified stream
        if   sev == "ERROR":   self._base.error(line)
        elif sev == "WARNING": self._base.warning(line)
        else:                  self._base.info(line)

        # Queue to Cosmos (non-blocking)
        if self.enable_cosmos and self._cosmos_container is not None:
            log_entry = {
                "id": str(uuid.uuid4()),
                "category": category,
                "event_type": event_type,
                "message": message,
                "severity": sev,
                "timestamp": datetime.utcnow().isoformat(),
            }
            try:
                self._q.put_nowait(log_entry)
            except queue.Full:
                # Drop if queue is full to avoid blocking application flow
                self._base.warning("Cosmos log queue full; dropping log entry.")

    # Convenience wrappers (used by your code)
    def log_error(self, category: str, message: str):
        self.log_event(category, "Error", f"❌ {message}", severity="ERROR")

    def log_warning(self, category: str, message: str):
        self.log_event(category, "Warning", f"⚠️ {message}", severity="WARNING")

    def log_decision(self, details: str):
        self.log_event("DecisionMaker", "Decision", details, severity="INFO")

    def log_memory(self, details: str):
        self.log_event("Memory", "Change", details, severity="INFO")

    def log_interaction(self, user_message: str, ai_response: str):
        self.log_event("Interaction", "Chat", f"User: {user_message} | AI: {ai_response}", severity="INFO")

    def log_security(self, message: str):
        self.log_event("Security", "Alert", message, severity="WARNING")

    def log_system_health(self, metrics: Any):
        if isinstance(metrics, dict):
            msg = " | ".join(f"{k}={v}" for k, v in metrics.items())
        else:
            msg = str(metrics)
        self.log_event("SystemHealth", "Monitoring", msg, severity="INFO")

    def log_task(self, task_name: str, status: str):
        self.log_event("Task", "Execution", f"{task_name}: {status}", severity="INFO")

    # ---------- Shutdown ----------
    def shutdown(self):
        """Flush and close all handlers; stop cosmos worker."""
        self._base.info("🛑 Logger shutdown requested.")
        # Stop worker
        if self._worker and self._worker.is_alive():
            self._stop_evt.set()
            self._worker.join(timeout=2.0)

        # Close cosmos client
        try:
            if self._cosmos_client:
                self._cosmos_client.close()  # type: ignore
        except Exception:
            pass

        # Flush/close handlers
        def _close_logger(lg: logging.Logger):
            for h in list(lg.handlers):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
                lg.removeHandler(h)

        _close_logger(self._base)
        for lg in self.loggers.values():
            _close_logger(lg)

        logging.shutdown()
        # Best-effort wait for queue drain without blocking app
        try:
            while not self._q.empty():
                self._q.get_nowait()
                self._q.task_done()
        except Exception:
            pass
        self._base = logging.getLogger("Snowball")  # reset reference
        # Note: reusing after shutdown will re-add handlers on new __init__

# Aliased name so you can import it like your previous modules if desired
SnowballLogger = AdvancedSnowballLogger


if __name__ == "__main__":
    # Quick self-test
    log = SnowballLogger(use_console=True, enable_cosmos=False)
    log.log_event("System", "Startup", "Snowball AI initialized.")
    log.log_interaction("How do you work?", "With style. 😎")
    log.log_warning("Config", "Using default settings.")
    log.log_error("AI", "Failed to process voice command.")
    log.log_system_health({"cpu": "12%", "mem": "48%"})
    log.log_task("DailyCleanup", "OK")
    log.shutdown()
