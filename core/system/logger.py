"""
SnowballLogger (Unified, MVP-safe)

- Rotating log file + console
- No cloud logging, no background threads
- Extremely tolerant API (positional + keyword + legacy)
- Portable (defaults to Snowball/storage/logs unless overridden)

Env:
- SNOWBALL_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL
- SNOWBALL_LOG_DIR=<path>   (optional override)
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Any, Optional


# ------------------------- Helpers -------------------------

def _env_log_level(default: int = logging.INFO) -> int:
    val = (os.getenv("SNOWBALL_LOG_LEVEL") or "").strip().upper()
    return getattr(logging, val, default)


def _snowball_root_guess() -> str:
    """
    Best-effort locate Snowball/ root based on this file:
      Snowball/core/system/logger.py -> go up 3 -> Snowball/
    """
    try:
        here = os.path.abspath(os.path.dirname(__file__))
        return os.path.abspath(os.path.join(here, "..", "..", ".."))
    except Exception:
        return os.getcwd()


def _default_log_dir() -> str:
    """
    Unified default:
      Snowball/storage/logs
    """
    root = _snowball_root_guess()
    return os.path.join(root, "storage", "logs")


def _ensure_log_dir() -> str:
    """
    Priority:
      1) SNOWBALL_LOG_DIR env var
      2) Snowball/storage/logs
      3) current working directory fallback
    """
    log_dir = (os.getenv("SNOWBALL_LOG_DIR") or "").strip() or _default_log_dir()
    try:
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    except Exception:
        try:
            return os.getcwd()
        except Exception:
            return "."


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return "<unprintable>"


# ------------------------- Logger -------------------------

class SnowballLogger:
    """
    Minimal, safe logger.

    Design goals:
    - Safe to instantiate multiple times
    - Never duplicate handlers
    - Never throw exceptions during logging
    - Accept inconsistent call signatures from legacy code
    """

    def __init__(
        self,
        name: str = "snowball",
        filename: str = "snowball.log",
        level: Optional[int] = None,
        max_bytes: int = 1_000_000,
        backup_count: int = 5,
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        lvl = _env_log_level(logging.INFO) if level is None else int(level)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(lvl)
        self._logger.propagate = False

        if getattr(self._logger, "_snowball_initialized", False):
            return

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        if enable_file:
            try:
                log_dir = _ensure_log_dir()
                log_path = os.path.join(log_dir, filename)
                fh = RotatingFileHandler(
                    log_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                fh.setFormatter(formatter)
                fh.setLevel(lvl)
                self._logger.addHandler(fh)
            except Exception:
                pass

        if enable_console:
            try:
                ch = logging.StreamHandler(sys.stdout)
                ch.setFormatter(formatter)
                ch.setLevel(lvl)
                self._logger.addHandler(ch)
            except Exception:
                pass

        setattr(self._logger, "_snowball_initialized", True)
        self.info("✅ SnowballLogger initialized.")

    # ------------------------- Basic wrappers -------------------------

    def debug(self, msg: str) -> None:
        try:
            self._logger.debug(_safe_str(msg))
        except Exception:
            pass

    def info(self, msg: str) -> None:
        try:
            self._logger.info(_safe_str(msg))
        except Exception:
            pass

    def warning(self, msg: str) -> None:
        try:
            self._logger.warning(_safe_str(msg))
        except Exception:
            pass

    def error(self, msg: str) -> None:
        try:
            self._logger.error(_safe_str(msg))
        except Exception:
            pass

    def exception(self, msg: str) -> None:
        try:
            self._logger.exception(_safe_str(msg))
        except Exception:
            pass

    # ------------------------- Compatibility API -------------------------

    def log_event(self, *args: Any, **kwargs: Any) -> None:
        """
        Supported call styles:
        - log_event(category, event_type, message, severity="INFO")
        - log_event(category, message)
        - log_event("fully formatted string")
        - log_event(category=..., event_type=..., message=..., severity=...)
        """
        try:
            category = ""
            event_type = ""
            message = ""
            severity = "INFO"

            if kwargs:
                category = _safe_str(kwargs.get("category", ""))
                event_type = _safe_str(kwargs.get("event_type", ""))
                message = _safe_str(kwargs.get("message", ""))
                severity = _safe_str(kwargs.get("severity", severity))

            if args:
                if len(args) == 1 and not kwargs:
                    message = _safe_str(args[0])
                elif len(args) >= 2 and not message:
                    category = _safe_str(args[0])
                    message = _safe_str(args[1])
                elif len(args) >= 3:
                    category = _safe_str(args[0])
                    event_type = _safe_str(args[1])
                    message = _safe_str(args[2])

                if len(args) >= 4 and ("severity" not in kwargs):
                    severity = _safe_str(args[3])

            if category and event_type and message:
                line = f"[{category}] {event_type} — {message}"
            elif category and message:
                line = f"[{category}] {message}"
            else:
                line = message or category or ""

            lvl = (severity or "INFO").upper()
            if lvl == "ERROR":
                self.error(line)
            elif lvl in {"WARN", "WARNING"}:
                self.warning(line)
            elif lvl == "DEBUG":
                self.debug(line)
            elif lvl == "CRITICAL":
                try:
                    self._logger.critical(line)
                except Exception:
                    self.error(line)
            else:
                self.info(line)
        except Exception:
            pass

    def log_error(self, category: str, message: str) -> None:
        try:
            self.error(f"[{_safe_str(category)}] ❌ {_safe_str(message)}")
        except Exception:
            pass

    def log_warning(self, category: str, message: str) -> None:
        try:
            self.warning(f"[{_safe_str(category)}] ⚠️ {_safe_str(message)}")
        except Exception:
            pass

    def log_decision(self, details: str) -> None:
        try:
            self.info(f"[Decision] {_safe_str(details)}")
        except Exception:
            pass

    def log_memory(self, details: str) -> None:
        try:
            self.info(f"[Memory] {_safe_str(details)}")
        except Exception:
            pass

    # ------------------------- Child loggers -------------------------

    def get_child(self, suffix: str) -> "SnowballLogger":
        try:
            return SnowballLogger(name=f"{self._logger.name}.{_safe_str(suffix)}")
        except Exception:
            return self

    # ------------------------- Shutdown -------------------------

    def shutdown(self) -> None:
        try:
            self.info("🛑 Logger shutdown.")
            for h in list(self._logger.handlers):
                try:
                    h.flush()
                except Exception:
                    pass
                try:
                    h.close()
                except Exception:
                    pass
                try:
                    self._logger.removeHandler(h)
                except Exception:
                    pass
            setattr(self._logger, "_snowball_initialized", False)
        except Exception:
            pass
