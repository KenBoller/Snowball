# S:/Snowball/core/api/chat_api.py
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# We now use the unified SnowballAI (which includes the deterministic router + KV memory + TOOL_RESULT injection)
from core.ai.chat import SnowballAI

_LOCK = threading.RLock()
_LOGGER = None
_AGENT: Optional[SnowballAI] = None
_BOOTSTRAPPED = False


# ----------------------------
# Pathing (dev + PyInstaller)
# ----------------------------

def _snowball_root() -> str:
    """
    Dev: .../Snowball/core/api/chat_api.py -> root is .../Snowball
    PyInstaller: sys._MEIPASS is the extracted app root.
    """
    if getattr(sys, "_MEIPASS", None):
        return str(getattr(sys, "_MEIPASS"))
    # core/ai/chat_api.py -> go up 3 levels to Snowball/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _maybe_set_default_log_dir() -> None:
    if not os.getenv("SNOWBALL_LOG_DIR"):
        os.environ["SNOWBALL_LOG_DIR"] = os.path.join(_snowball_root(), "storage", "logs")


def _maybe_set_default_local_memory_dir() -> None:
    if not os.getenv("SNOWBALL_LOCAL_MEMORY_DIR"):
        os.environ["SNOWBALL_LOCAL_MEMORY_DIR"] = os.path.join(_snowball_root(), "storage", "memory")


def _maybe_set_default_storage_dir() -> None:
    """
    Used for KV memory + any other local persistence.
    """
    if not os.getenv("SNOWBALL_STORAGE_DIR"):
        os.environ["SNOWBALL_STORAGE_DIR"] = os.path.join(_snowball_root(), "storage")


def _maybe_set_default_config_path() -> None:
    """
    Preferred location: Snowball/config/account_integrations.json
    Fallback: Snowball/account_integrations.json
    """
    if os.getenv("SNOWBALL_CONFIG_PATH"):
        return

    p1 = os.path.join(_snowball_root(), "config", "account_integrations.json")
    p2 = os.path.join(_snowball_root(), "account_integrations.json")

    if os.path.exists(p1):
        os.environ["SNOWBALL_CONFIG_PATH"] = p1
    elif os.path.exists(p2):
        os.environ["SNOWBALL_CONFIG_PATH"] = p2


def _bootstrap_env_defaults() -> None:
    global _BOOTSTRAPPED
    with _LOCK:
        if _BOOTSTRAPPED:
            return
        _maybe_set_default_storage_dir()
        _maybe_set_default_log_dir()
        _maybe_set_default_local_memory_dir()
        _maybe_set_default_config_path()
        _BOOTSTRAPPED = True


# ----------------------------
# Logger
# ----------------------------

def get_logger():
    global _LOGGER
    with _LOCK:
        if _LOGGER is not None:
            return _LOGGER

        _bootstrap_env_defaults()

        from core.system.logger import SnowballLogger
        _LOGGER = SnowballLogger()
        return _LOGGER


# ----------------------------
# Agent lifecycle
# ----------------------------

def get_agent(
    state_getters: Optional[Dict[str, Callable[[], Any]]] = None,
    state_setters: Optional[Dict[str, Callable[[Any], None]]] = None,
) -> SnowballAI:
    """
    Returns a singleton agent instance.
    Optionally allows callers (GUI/main_main) to inject real state getters/setters.
    """
    global _AGENT
    with _LOCK:
        if _AGENT is not None:
            return _AGENT

        _bootstrap_env_defaults()

        logger = get_logger()

        storage_dir = os.getenv("SNOWBALL_STORAGE_DIR") or os.path.join(_snowball_root(), "storage")
        Path(storage_dir).mkdir(parents=True, exist_ok=True)

        _AGENT = SnowballAI(
            logger=logger,
            state_getters=state_getters,
            state_setters=state_setters,
            storage_dir=storage_dir,
        )
        return _AGENT


def get_agent_optional() -> Optional[SnowballAI]:
    """
    Returns the agent if already initialized; otherwise None.
    Useful for GUIs that want to avoid forcing init.
    """
    with _LOCK:
        return _AGENT


def init_agent(
    state_getters: Optional[Dict[str, Callable[[], Any]]] = None,
    state_setters: Optional[Dict[str, Callable[[Any], None]]] = None,
) -> None:
    """
    Explicit initializer (optional). Safe to call multiple times.
    """
    _ = get_agent(state_getters=state_getters, state_setters=state_setters)


# ----------------------------
# Public API used by the UI
# ----------------------------

def send_message(text: str) -> str:
    """
    Main call used by UI.
    """
    text = (text or "").strip()
    if not text:
        return "Say something and I’ll jump in."

    try:
        agent = get_agent()
        return agent.chat(text)
    except Exception as e:
        try:
            get_logger().log_event("ChatAPI", "Error", f"send_message failed: {e}", severity="ERROR")
        except Exception:
            pass
        return f"System: Error ({type(e).__name__}): {e}"


def ping_ollama(require_any: bool = False) -> Dict[str, Any]:
    """
    Deterministic check (used by router command: 'ping ollama').
    """
    try:
        agent = get_agent()
        if not hasattr(agent, "validate_api_keys"):
            return {"ollama": False, "error": "Agent does not expose validate_api_keys()."}

        return agent.validate_api_keys(require_any=require_any)  # type: ignore
    except Exception as e:
        try:
            get_logger().log_event("ChatAPI", "PingError", f"{type(e).__name__}: {e}", severity="ERROR")
        except Exception:
            pass
        return {"ollama": False, "error": f"{type(e).__name__}: {e}"}

def get_chat_agent(
    state_getters: Optional[Dict[str, Callable[[], Any]]] = None,
    state_setters: Optional[Dict[str, Callable[[Any], None]]] = None,
) -> SnowballAI:
    """
    Backward-compatible alias for get_agent().
    """
    return get_agent(
        state_getters=state_getters,
        state_setters=state_setters,
    )


def reset_chat_agent() -> None:
    """
    Backward-compatible alias for reset().
    """
    reset()
    
def reset() -> None:
    """
    Clears singleton instances.
    """
    global _LOGGER, _AGENT, _BOOTSTRAPPED
    with _LOCK:
        _AGENT = None
        _LOGGER = None
        _BOOTSTRAPPED = False
