# S:/Snowball/core/ai/router.py
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


# ----------------------------
# Tool helpers (Ollama)
# ----------------------------

def _run_cmd(cmd: list[str], timeout: int = 10) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False
        )
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return 124, "", "Command timed out"
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def ollama_list_models() -> Dict[str, Any]:
    """
    Try to list models via `ollama list`.
    Returns: {reachable: bool, models: [str], raw: str, error: str|None}
    """
    rc, out, err = _run_cmd(["ollama", "list"], timeout=10)
    if rc != 0:
        return {"reachable": False, "models": [], "raw": out, "error": err or out or "ollama list failed"}

    # Typical output:
    # NAME               ID              SIZE      MODIFIED
    # llama3:8b          ...             ...       ...
    # deepseek-r1:14b    ...             ...       ...
    models: list[str] = []
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) >= 2:
        for ln in lines[1:]:
            # NAME is first column; split on whitespace
            parts = ln.split()
            if parts:
                models.append(parts[0])

    return {"reachable": True, "models": models, "raw": out, "error": None}


def ollama_ping() -> Dict[str, Any]:
    """
    "Ping" Ollama by running `ollama list` as reachability check.
    """
    return ollama_list_models()


# ----------------------------
# KV Memory (tiny + safe)
# ----------------------------

class KVMemory:
    """
    Simple JSON file key/value memory for deterministic remember/recall,
    plus prompt injection context.
    """
    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            try:
                self._cache = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(self._cache, dict):
                    self._cache = {}
            except Exception:
                self._cache = {}
        self._loaded = True

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8")

    def set(self, key: str, value: str) -> None:
        self._load()
        self._cache[key] = value
        self._save()

    def get(self, key: str) -> Optional[str]:
        self._load()
        return self._cache.get(key)

    def items(self) -> Dict[str, str]:
        self._load()
        return dict(self._cache)

    def delete(self, key: str) -> bool:
        self._load()
        if key in self._cache:
            del self._cache[key]
            self._save()
            return True
        return False


# ----------------------------
# Router
# ----------------------------

@dataclass
class ToolResult:
    name: str
    payload: Dict[str, Any]
    ts: float

    def to_prompt_block(self) -> str:
        return (
            f"TOOL_RESULT ({self.name}):\n"
            f"{json.dumps(self.payload, indent=2, ensure_ascii=False)}\n"
        )


@dataclass
class RouterResponse:
    handled: bool
    reply_text: str = ""
    tool_result: Optional[ToolResult] = None


class CommandRouter:
    """
    Deterministic command router. If handled=True, do not call LLM.
    """
    def __init__(
        self,
        kv_memory: KVMemory,
        state_getters: Optional[Dict[str, Callable[[], Any]]] = None,
        state_setters: Optional[Dict[str, Callable[[Any], None]]] = None,
    ):
        self.mem = kv_memory
        self.getters = state_getters or {}
        self.setters = state_setters or {}

        # Compile patterns once
        self._pat_ping = re.compile(r"^\s*ping\s+ollama\s*$", re.IGNORECASE)
        self._pat_models = re.compile(r"^\s*(models|list\s+models|available\s+models)\s*$", re.IGNORECASE)

        self._pat_remember = re.compile(
            r"^\s*remember\s+([a-zA-Z0-9_\-\.]+)\s*=\s*(.+?)\s*$",
            re.IGNORECASE
        )
        self._pat_recall = re.compile(
            r"^\s*(recall|get|remembered)\s+([a-zA-Z0-9_\-\.]+)\s*$",
            re.IGNORECASE
        )
        self._pat_list_memory = re.compile(
            r"^\s*(what\s+did\s+i\s+ask\s+you\s+to\s+remember\??|list\s+memory|memory\s+list)\s*$",
            re.IGNORECASE
        )
        self._pat_forget = re.compile(
            r"^\s*(forget|delete)\s+([a-zA-Z0-9_\-\.]+)\s*$",
            re.IGNORECASE
        )

        self._pat_status = re.compile(
            r"^\s*(status)\s+([a-zA-Z0-9_\-\.]+)\s*$",
            re.IGNORECASE
        )
        self._pat_enable = re.compile(
            r"^\s*(enable|start|turn\s+on)\s+([a-zA-Z0-9_\-\.]+)\s*$",
            re.IGNORECASE
        )
        self._pat_disable = re.compile(
            r"^\s*(disable|stop|turn\s+off)\s+([a-zA-Z0-9_\-\.]+)\s*$",
            re.IGNORECASE
        )

    def route(self, user_text: str) -> RouterResponse:
        text = user_text.strip()

        # --- Ollama ---
        if self._pat_ping.match(text):
            payload = ollama_ping()
            tr = ToolResult(name="ollama_ping", payload=payload, ts=time.time())
            if payload.get("reachable"):
                models = payload.get("models", [])
                if models:
                    reply = "✅ Ollama reachable.\nModels:\n- " + "\n- ".join(models)
                else:
                    reply = "✅ Ollama reachable, but no models were listed."
            else:
                reply = f"❌ Ollama not reachable.\nError: {payload.get('error')}"
            return RouterResponse(True, reply_text=reply, tool_result=tr)

        if self._pat_models.match(text):
            payload = ollama_list_models()
            tr = ToolResult(name="ollama_list_models", payload=payload, ts=time.time())
            if payload.get("reachable"):
                models = payload.get("models", [])
                if models:
                    reply = "Available Ollama models:\n- " + "\n- ".join(models)
                else:
                    reply = "Ollama is reachable, but no models were listed."
            else:
                reply = f"Couldn’t list models (Ollama not reachable).\nError: {payload.get('error')}"
            return RouterResponse(True, reply_text=reply, tool_result=tr)

        # --- Memory ---
        m = self._pat_remember.match(text)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            self.mem.set(key, value)
            return RouterResponse(True, reply_text=f"✅ Saved memory: {key} = {value}")

        m = self._pat_recall.match(text)
        if m:
            key = m.group(2).strip()
            val = self.mem.get(key)
            if val is None:
                return RouterResponse(True, reply_text=f"ℹ️ I don’t have a saved value for `{key}` yet.")
            return RouterResponse(True, reply_text=f"{key} = {val}")

        if self._pat_list_memory.match(text):
            items = self.mem.items()
            if not items:
                return RouterResponse(True, reply_text="ℹ️ Memory is empty right now.")
            lines = [f"- {k} = {v}" for k, v in sorted(items.items(), key=lambda kv: kv[0].lower())]
            return RouterResponse(True, reply_text="Here’s what you’ve asked me to remember:\n" + "\n".join(lines))

        m = self._pat_forget.match(text)
        if m:
            key = m.group(2).strip()
            ok = self.mem.delete(key)
            if ok:
                return RouterResponse(True, reply_text=f"✅ Deleted memory key `{key}`.")
            return RouterResponse(True, reply_text=f"ℹ️ No memory key `{key}` found to delete.")

        # --- Status / toggles (state-driven, not LLM) ---
        m = self._pat_status.match(text)
        if m:
            name = m.group(2).strip().lower()
            getter = self.getters.get(name)
            if not getter:
                return RouterResponse(True, reply_text=f"ℹ️ I don’t have a status provider for `{name}`.")
            try:
                val = getter()
                return RouterResponse(True, reply_text=f"{name}: {val}")
            except Exception as e:
                return RouterResponse(True, reply_text=f"❌ Failed to read `{name}` status: {type(e).__name__}: {e}")

        m = self._pat_enable.match(text)
        if m:
            name = m.group(2).strip().lower()
            setter = self.setters.get(name)
            if not setter:
                return RouterResponse(True, reply_text=f"ℹ️ I don’t have an enable/disable controller for `{name}`.")
            try:
                setter(True)
                # re-check if possible
                getter = self.getters.get(name)
                val = getter() if getter else "enabled"
                return RouterResponse(True, reply_text=f"✅ {name} is now {val}.")
            except Exception as e:
                return RouterResponse(True, reply_text=f"❌ Failed to enable `{name}`: {type(e).__name__}: {e}")

        m = self._pat_disable.match(text)
        if m:
            name = m.group(2).strip().lower()
            setter = self.setters.get(name)
            if not setter:
                return RouterResponse(True, reply_text=f"ℹ️ I don’t have an enable/disable controller for `{name}`.")
            try:
                setter(False)
                getter = self.getters.get(name)
                val = getter() if getter else "disabled"
                return RouterResponse(True, reply_text=f"✅ {name} is now {val}.")
            except Exception as e:
                return RouterResponse(True, reply_text=f"❌ Failed to disable `{name}`: {type(e).__name__}: {e}")

        # Not handled -> let LLM handle natural language
        return RouterResponse(False)
