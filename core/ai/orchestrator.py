"""
Snowball Orchestrator (Unified)

Key upgrades preserved:
- extra_context dict-safe across the pipeline
- per-stage timeouts (draft/critique/revise)
- retry/backoff around Ollama calls
- optional final sanity + fallback competition if final looks weak
- anti-echo guard for INTERNAL CONTEXT leakage
- MVP-safe: vision remains lazy

Notes:
- Uses Ollama /api/generate with prompt strings.
- Vision routing stays explicit: context_category=="Vision" OR "vision:" prefix.
- Config is loaded from Snowball/config/account_integrations.json (dev + PyInstaller).
"""

from __future__ import annotations

import os
import sys
import json
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import urllib.request
import urllib.error


# ----------------------------- Path helpers -----------------------------

def _here() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def _snowball_root() -> str:
    """
    Dev:
      Snowball/core/ai/orchestrator.py -> root is .../Snowball
    PyInstaller:
      sys._MEIPASS points to extraction dir containing config/, icon/, storage/, etc.
    """
    if getattr(sys, "_MEIPASS", None):
        return str(getattr(sys, "_MEIPASS"))
    return os.path.abspath(os.path.join(_here(), "..", "..", ".."))


def _config_path() -> str:
    return os.path.join(_snowball_root(), "config", "account_integrations.json")


# ----------------------------- Small helpers -----------------------------

def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    if val is None:
        return default
    val = str(val).strip()
    return val if val else default


def _clamp_int(val: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(val)
        return max(lo, min(hi, n))
    except Exception:
        return default


def _clamp_float(val: Any, default: float, lo: float, hi: float) -> float:
    try:
        n = float(val)
        return max(lo, min(hi, n))
    except Exception:
        return default


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _strip_disallowed(text: str) -> str:
    """Remove common boilerplate / disallowed phrases and normalize spacing."""
    if not isinstance(text, str):
        return ""
    t = text.strip()

    t = re.sub(
        r"\b(as an ai language model|i (?:can'?t|cannot) (?:provide|do)|i do not have the ability)\b.*",
        "",
        t,
        flags=re.I,
    )

    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _strip_internal_context_echo(text: str) -> str:
    """
    If a model accidentally repeats the INTERNAL CONTEXT block, strip it.
    Defensive: prompts already say "do not mention".
    """
    if not isinstance(text, str):
        return ""
    t = text
    t = re.sub(
        r"(?is)\bINTERNAL CONTEXT\s*\(do not mention\)\s*:\s*\n.*?\n\n",
        "",
        t,
    )
    return t.strip()


def _format_extra_context(extra_context: Optional[Any]) -> str:
    """
    Accepts str/dict/list/tuple/any. Ensures internal context is bounded and separated.
    """
    if extra_context is None:
        return ""

    if isinstance(extra_context, (dict, list, tuple)):
        try:
            ctx = json.dumps(extra_context, ensure_ascii=False, indent=2)
        except Exception:
            ctx = str(extra_context)
    else:
        ctx = str(extra_context)

    ctx = (ctx or "").strip()
    if not ctx:
        return ""

    if len(ctx) > 2000:
        ctx = ctx[:2000].rstrip() + "…"

    return (
        "INTERNAL CONTEXT (do not mention):\n"
        f"{ctx}\n\n"
    )


# ----------------------------- Logging -----------------------------

def _get_null_logger():
    class _NullLogger:
        def log_event(self, *a, **k): ...
        def log_error(self, *a, **k): ...
        def log_warning(self, *a, **k): ...
        def log_decision(self, *a, **k): ...
    return _NullLogger()


def get_logger():
    """
    Package-first logger import.
    Falls back to legacy 'logger.py' if you still have it in an older layout.
    """
    try:
        from Snowball.core.system.logger import SnowballLogger  # type: ignore
        return SnowballLogger()
    except Exception:
        try:
            from logger import SnowballLogger  # type: ignore
            return SnowballLogger()
        except Exception:
            return _get_null_logger()


def _log(logger, category: str, event_type: str, message: str, severity: str = "INFO"):
    try:
        logger.log_event(category, event_type, message, severity=severity)
    except Exception:
        try:
            logger.log_event(f"{category} | {event_type} | {message}")
        except Exception:
            pass


def _err(logger, category: str, message: str):
    try:
        logger.log_error(category, message)
    except Exception:
        try:
            logger.log_error(f"{category}: {message}")
        except Exception:
            pass


# ----------------------------- Decision fallback -----------------------------

def _maybe_get_decision_maker(logger):
    """
    Package-first DecisionMaker import, with legacy fallback.
    """
    try:
        from Snowball.core.ai.decision_maker import DecisionMaker  # type: ignore
        try:
            return DecisionMaker(logger=logger)
        except TypeError:
            return DecisionMaker(logger)
    except Exception:
        try:
            from decision_maker import DecisionMaker  # type: ignore
            try:
                return DecisionMaker(logger=logger)
            except TypeError:
                return DecisionMaker(logger)
        except Exception:
            return None


def _validate_response_quality(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 10:
        return False
    if "i don't know" in t.lower():
        return False
    return True


# ----------------------------- Data -----------------------------

@dataclass
class OrchestratorConfig:
    ollama_base_url: str
    primary_model: str
    planner_model: str
    critic_model: str
    fallback_models: List[str]
    mode: str

    timeout_sec: int
    max_tokens: int

    draft_timeout_sec: int = 120
    critique_timeout_sec: int = 45
    revise_timeout_sec: int = 75
    competition_timeout_sec: int = 90

    temperature: float = 0.7
    top_p: float = 0.9

    vision_enabled: bool = True
    vision_yolo_repo_dir: Optional[str] = None
    vision_yolo_local_path: Optional[str] = None
    vision_allow_hub_download: bool = False
    vision_show_windows: bool = True
    vision_camera_index: int = 0

    retries: int = 1
    retry_backoff_sec: float = 1.0


# ----------------------------- Orchestrator -----------------------------

class Orchestrator:
    """
    Orchestrator:
    - Text modes: pipeline | competition | draft_only
    - Vision: routed when context_category=="Vision" OR input starts with "vision:"
    - extra_context (dict/str/etc.) injected as INTERNAL CONTEXT (do not mention)
    """

    def __init__(self, logger=None, config: Optional[OrchestratorConfig] = None):
        self.logger = logger or get_logger()
        self.config = config or self._load_config()
        self.decision_maker = _maybe_get_decision_maker(self.logger)

        self._vision = None  # lazy

        _log(
            self.logger,
            "Orchestrator",
            "INIT",
            f"mode={self.config.mode} primary={self.config.primary_model} "
            f"critic={self.config.critic_model} planner={self.config.planner_model} "
            f"base_url={self.config.ollama_base_url} vision={self.config.vision_enabled}",
        )

    # ------------------------- Public API -------------------------

    def respond(
        self,
        user_input: str,
        query_type: str = "General",
        context_category: str = "Text",
        extra_context: Optional[Any] = None,
    ) -> str:
        user_input = (user_input or "").strip()
        if not user_input:
            return "Tell me what you want to do, and I’ll help."

        # Vision routing
        if self._should_route_to_vision(user_input, context_category):
            return self.respond_vision(user_input=user_input, query_type=query_type)

        mode = (self.config.mode or "pipeline").strip().lower()

        if mode == "competition":
            return self._competition_mode(
                user_input=user_input,
                query_type=query_type,
                context_category=context_category,
                extra_context=extra_context,
            )

        if mode == "draft_only":
            draft = self._draft(user_input=user_input, query_type=query_type, extra_context=extra_context)
            if _validate_response_quality(draft):
                return draft
            return self._fallback_to_decision(
                user_input=user_input,
                query_type=query_type,
                context_category=context_category,
                extra_context=extra_context,
            )

        return self._pipeline_mode(
            user_input=user_input,
            query_type=query_type,
            context_category=context_category,
            extra_context=extra_context,
        )

    # ------------------------- Vision routing -------------------------

    def _should_route_to_vision(self, user_input: str, context_category: str) -> bool:
        if not self.config.vision_enabled:
            return False
        if (context_category or "").strip().lower() == "vision":
            return True
        return user_input.strip().lower().startswith("vision:")

    def _ensure_vision(self):
        if self._vision is not None:
            return self._vision

        # Package-first import, legacy fallback
        try:
            from Snowball.core.ai.vision import Vision, VisionConfig  # type: ignore
        except Exception:
            try:
                from vision import Vision, VisionConfig  # type: ignore
            except Exception as e:
                _err(self.logger, "Orchestrator", f"Vision import failed: {e}")
                self._vision = None
                return None

        try:
            vcfg = VisionConfig(
                camera_index=int(self.config.vision_camera_index),
                open_camera_on_init=False,
                show_windows=bool(self.config.vision_show_windows),
                yolo_repo_dir=self.config.vision_yolo_repo_dir,
                yolo_local_path=self.config.vision_yolo_local_path,
                allow_torch_hub_download=bool(self.config.vision_allow_hub_download),
            )
            self._vision = Vision(logger=self.logger, config=vcfg)
            return self._vision
        except Exception as e:
            _err(self.logger, "Orchestrator", f"Vision init failed: {e}")
            self._vision = None
            return None

    def respond_vision(self, user_input: str, query_type: str = "Vision") -> str:
        raw = user_input.strip()
        if raw.lower().startswith("vision:"):
            raw = raw.split(":", 1)[1].strip()

        raw = _normalize_space(raw)
        if not raw:
            raw = "health"

        vision = self._ensure_vision()
        if vision is None:
            return (
                "Vision module isn’t available (failed to load). "
                "Check dependencies: opencv-python, deepface, torch (optional)."
            )

        parts = raw.split()
        verb = parts[0].lower() if parts else "health"
        arg1 = parts[1] if len(parts) > 1 else None

        try:
            if verb in {"health", "status"}:
                h = vision.health()
                return (
                    "Vision status:\n"
                    f"- opencv: {h.get('opencv')}\n"
                    f"- torch: {h.get('torch')}\n"
                    f"- deepface: {h.get('deepface')}\n"
                    f"- camera_open: {h.get('camera_open')}\n"
                    f"- face_detection: {h.get('face_detection')}\n"
                    f"- deepface_ready: {h.get('deepface_ready')}\n"
                    f"- yolo_loaded: {h.get('yolo_loaded')}\n"
                    f"- yolo_repo_dir: {h.get('yolo_repo_dir')}\n"
                    f"- yolo_local_path: {h.get('yolo_local_path')}\n"
                    f"- allow_hub_download: {h.get('allow_hub_download')}\n"
                )

            if verb == "capture":
                filename = arg1 or "captured_image.jpg"
                ok = vision.capture_and_save_image(filename=filename)
                return f"📸 Capture {'succeeded' if ok else 'failed'}: {filename}"

            if verb in {"sentiment", "emotion"}:
                sent = vision.detect_facial_sentiment()
                return f"Detected facial sentiment: {sent}"

            if verb in {"objects", "object"}:
                frame = vision._read_frame()  # type: ignore[attr-defined]
                if frame is None:
                    return "Couldn’t read from the camera."
                labels = vision.recognize_objects(frame)
                if not labels:
                    return "No objects detected (or YOLO isn’t available)."
                return f"Objects detected: {', '.join(labels[:10])}"

            if verb in {"faces", "face"}:
                vision.start_facial_recognition()
                vision.stop()
                return "Facial recognition session ended."

            if verb in {"detect", "object_detect"}:
                vision.start_object_detection()
                vision.stop()
                return "Object detection session ended."

            if verb in {"stop", "close"}:
                vision.stop()
                return "Vision stopped."

            return (
                "Unknown vision command.\n"
                "Try:\n"
                "- vision: health\n"
                "- vision: capture [filename]\n"
                "- vision: sentiment\n"
                "- vision: objects\n"
                "- vision: faces\n"
                "- vision: detect\n"
                "- vision: stop\n"
            )

        except Exception as e:
            _err(self.logger, "Orchestrator", f"Vision command failed: {e}")
            return f"Vision command failed: {e}"

    # ------------------------- Pipeline mode -------------------------

    def _pipeline_mode(self, user_input: str, query_type: str, context_category: str, extra_context: Optional[Any]) -> str:
        t0 = time.time()

        draft = self._draft(user_input=user_input, query_type=query_type, extra_context=extra_context)
        if not _validate_response_quality(draft):
            _log(self.logger, "Orchestrator", "PIPE", "Draft low quality; falling back.", severity="WARNING")
            return self._fallback_to_decision(
                user_input=user_input,
                query_type=query_type,
                context_category=context_category,
                extra_context=extra_context,
            )

        critique = self._critique(user_input=user_input, draft=draft, query_type=query_type, extra_context=extra_context)
        if not critique:
            _log(self.logger, "Orchestrator", "PIPE", "Critique missing; returning draft.", severity="WARNING")
            final = draft
        else:
            revised = self._revise(
                user_input=user_input,
                draft=draft,
                critique=critique,
                query_type=query_type,
                extra_context=extra_context,
            )
            revised = _strip_disallowed(_strip_internal_context_echo(revised))
            final = revised if _validate_response_quality(revised) else draft

        if not _validate_response_quality(final):
            _log(self.logger, "Orchestrator", "PIPE", "Final low quality; competition fallback.", severity="WARNING")
            final = self._fallback_to_decision(
                user_input=user_input,
                query_type=query_type,
                context_category=context_category,
                extra_context=extra_context,
            )

        elapsed = time.time() - t0
        self._log_decision_summary(
            mode="pipeline",
            user_input=user_input,
            query_type=query_type,
            chosen="pipeline_final",
            scores=None,
            meta={
                "elapsed_sec": round(elapsed, 3),
                "draft_len": len(draft or ""),
                "critique_len": len(critique or ""),
                "final_len": len(final or ""),
                "had_extra_context": bool(extra_context),
            },
        )
        return final

    def _draft(self, user_input: str, query_type: str, extra_context: Optional[Any]) -> str:
        prompt = self._prompt_draft(user_input=user_input, query_type=query_type, extra_context=extra_context)
        out = self._call_ollama(
            model=self.config.primary_model,
            prompt=prompt,
            timeout_sec=self.config.draft_timeout_sec,
        )
        out = _strip_disallowed(_strip_internal_context_echo(out))
        _log(self.logger, "Orchestrator", "DRAFT", f"len={len(out)}")
        return out

    def _critique(self, user_input: str, draft: str, query_type: str, extra_context: Optional[Any]) -> str:
        prompt = self._prompt_critique(user_input=user_input, draft=draft, query_type=query_type, extra_context=extra_context)
        out = self._call_ollama(
            model=self.config.critic_model,
            prompt=prompt,
            timeout_sec=self.config.critique_timeout_sec,
        )
        out = (out or "").strip()
        _log(self.logger, "Orchestrator", "CRITIQUE", f"len={len(out)}")
        return out

    def _revise(self, user_input: str, draft: str, critique: str, query_type: str, extra_context: Optional[Any]) -> str:
        prompt = self._prompt_revise(
            user_input=user_input,
            draft=draft,
            critique=critique,
            query_type=query_type,
            extra_context=extra_context,
        )
        out = self._call_ollama(
            model=self.config.planner_model,
            prompt=prompt,
            timeout_sec=self.config.revise_timeout_sec,
        )
        out = _strip_disallowed(_strip_internal_context_echo(out))
        _log(self.logger, "Orchestrator", "REVISE", f"len={len(out)}")
        return out

    # ------------------------- Competition fallback -------------------------

    def _competition_mode(self, user_input: str, query_type: str, context_category: str, extra_context: Optional[Any]) -> str:
        responses = self._generate_candidate_responses(user_input=user_input, query_type=query_type, extra_context=extra_context)
        return self._pick_best_with_decision_maker(responses, user_input, query_type, context_category)

    def _fallback_to_decision(self, user_input: str, query_type: str, context_category: str, extra_context: Optional[Any]) -> str:
        responses = self._generate_candidate_responses(user_input=user_input, query_type=query_type, extra_context=extra_context)
        chosen = self._pick_best_with_decision_maker(responses, user_input, query_type, context_category)
        self._log_decision_summary(
            mode="fallback_decision",
            user_input=user_input,
            query_type=query_type,
            chosen="DecisionMaker" if self.decision_maker else "Longest",
            scores={k: (1.0 if (v or "").strip() == chosen.strip() else 0.0) for k, v in responses.items()},
            meta={"candidates": list(responses.keys()), "had_extra_context": bool(extra_context)},
        )
        return chosen

    def _generate_candidate_responses(self, user_input: str, query_type: str, extra_context: Optional[Any]) -> Dict[str, str]:
        models: List[str] = []
        for m in [self.config.primary_model, self.config.planner_model, self.config.critic_model]:
            if m and m not in models:
                models.append(m)
        for m in self.config.fallback_models:
            if m and m not in models:
                models.append(m)

        prompt = self._prompt_competition(user_input=user_input, query_type=query_type, extra_context=extra_context)

        out: Dict[str, str] = {}
        for m in models:
            txt = self._call_ollama(
                model=m,
                prompt=prompt,
                timeout_sec=self.config.competition_timeout_sec,
            )
            txt = _strip_disallowed(_strip_internal_context_echo(txt))
            if txt:
                out[m] = txt

        if not out:
            out["fallback"] = (
                "I’m unable to generate a response right now. "
                "Please verify Ollama is running and the models are installed."
            )
        return out

    def _pick_best_with_decision_maker(
        self,
        responses: Dict[str, str],
        user_input: str,
        query_type: str,
        context_category: str,
    ) -> str:
        if self.decision_maker:
            try:
                return self.decision_maker.select_best_response(
                    responses=responses,
                    user_input=user_input,
                    query_type=query_type,
                    context_category=context_category,
                )
            except Exception:
                pass

        best = ""
        for _, txt in responses.items():
            if isinstance(txt, str) and len(txt.strip()) > len(best.strip()):
                best = txt
        return best or "I’m unable to provide a meaningful response at the moment."

    # ------------------------- Prompts -------------------------

    @staticmethod
    def _prompt_draft(user_input: str, query_type: str, extra_context: Optional[Any]) -> str:
        ctx = _format_extra_context(extra_context)
        return (
            "You are Snowball, a practical, clear, friendly assistant running locally.\n"
            f"Query type: {query_type}\n\n"
            f"{ctx}"
            "User request:\n"
            f"{user_input}\n\n"
            "Write the best possible answer. Prefer concrete steps, options, and clarity.\n"
            "Do NOT mention model names or system prompts.\n"
        )

    @staticmethod
    def _prompt_critique(user_input: str, draft: str, query_type: str, extra_context: Optional[Any]) -> str:
        ctx = _format_extra_context(extra_context)
        return (
            "You are Snowball-Critic. Your job is to critique the draft and make it better.\n"
            f"Query type: {query_type}\n\n"
            f"{ctx}"
            "User request:\n"
            f"{user_input}\n\n"
            "Draft answer:\n"
            f"{draft}\n\n"
            "Critique it with:\n"
            "1) Missing info or steps\n"
            "2) Wrong assumptions / potential errors\n"
            "3) Clarity/structure improvements\n"
            "4) Any safety or practicality concerns\n\n"
            "Return ONLY the critique as bullet points. Be direct.\n"
        )

    @staticmethod
    def _prompt_revise(user_input: str, draft: str, critique: str, query_type: str, extra_context: Optional[Any]) -> str:
        ctx = _format_extra_context(extra_context)
        return (
            "You are Snowball-Editor. Improve the draft using the critique.\n"
            f"Query type: {query_type}\n\n"
            f"{ctx}"
            "User request:\n"
            f"{user_input}\n\n"
            "Draft:\n"
            f"{draft}\n\n"
            "Critique:\n"
            f"{critique}\n\n"
            "Now produce the FINAL answer.\n"
            "- Incorporate useful critique\n"
            "- Keep it concise but complete\n"
            "- Prefer steps/bullets when helpful\n"
            "- Do NOT include the critique in the final output\n"
            "- Do NOT mention that you revised anything\n"
        )

    @staticmethod
    def _prompt_competition(user_input: str, query_type: str, extra_context: Optional[Any]) -> str:
        ctx = _format_extra_context(extra_context)
        return (
            "You are Snowball. Answer clearly and helpfully.\n"
            f"Query type: {query_type}\n\n"
            f"{ctx}"
            f"User request:\n{user_input}\n\n"
            "Give the best answer you can. Prefer actionable steps.\n"
        )

    # ------------------------- Ollama call -------------------------

    def _call_ollama(self, model: str, prompt: str, timeout_sec: Optional[int] = None) -> str:
        model = (model or "").strip()
        if not model:
            return ""

        base = (self.config.ollama_base_url or "").strip().rstrip("/")
        if not base:
            return ""

        url = f"{base}/api/generate"

        max_tokens = _clamp_int(self.config.max_tokens, default=1200, lo=64, hi=8000)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": float(self.config.temperature),
                "top_p": float(self.config.top_p),
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        timeout = float(_clamp_int(timeout_sec or self.config.timeout_sec, default=30, lo=5, hi=240))

        retries = max(0, int(self.config.retries))
        backoff = float(self.config.retry_backoff_sec)

        last_err: Optional[str] = None
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                j = json.loads(raw) if raw else {}
                out = j.get("response") or ""
                return out if isinstance(out, str) else str(out)

            except urllib.error.URLError as e:
                last_err = f"URLError model={model}: {e}"
                _err(self.logger, "Orchestrator", last_err)

            except Exception as e:
                last_err = f"Ollama call failed model={model}: {e}"
                _err(self.logger, "Orchestrator", last_err)

            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))

        return ""

    # ------------------------- Decision logging -------------------------

    def _log_decision_summary(
        self,
        mode: str,
        user_input: str,
        query_type: str,
        chosen: str,
        scores: Optional[Dict[str, float]],
        meta: Optional[Dict[str, Any]] = None,
    ):
        entry = {
            "mode": mode,
            "query_type": query_type,
            "chosen": chosen,
            "scores": scores,
            "meta": meta or {},
            "user_input_preview": (user_input or "")[:180],
        }
        try:
            self.logger.log_decision(json.dumps(entry, ensure_ascii=False))
        except Exception:
            _log(self.logger, "Orchestrator", "DECISION", str(entry))

    # ------------------------- Config loading -------------------------

    def _load_config(self) -> OrchestratorConfig:
        cfg = _safe_read_json(_config_path())
        src = cfg.get("api_keys", cfg) if isinstance(cfg, dict) else {}

        base_url = _env("OLLAMA_BASE_URL", src.get("ollama_base_url") if isinstance(src, dict) else None) or "http://localhost:11434"
        primary = _env("SNOWBALL_PRIMARY_MODEL", src.get("primary_model") if isinstance(src, dict) else None) or "deepseek-r1:14b"
        planner = _env("SNOWBALL_PLANNER_MODEL", src.get("planner_model") if isinstance(src, dict) else None) or "qwen2.5:7b-instruct"
        critic = _env("SNOWBALL_CRITIC_MODEL", src.get("critic_model") if isinstance(src, dict) else None) or "mistral:7b-instruct"

        fb = src.get("fallback_models", []) if isinstance(src, dict) else []
        fallback_models: List[str] = []
        if isinstance(fb, list):
            fallback_models = [str(x).strip() for x in fb if str(x).strip()]

        env_fbs = _env("SNOWBALL_FALLBACK_MODELS")
        if env_fbs:
            fallback_models = [x.strip() for x in env_fbs.split(",") if x.strip()]

        mode = (_env("SNOWBALL_ORCH_MODE", "pipeline") or "pipeline").strip().lower()

        timeout_sec = _clamp_int(_env("SNOWBALL_ORCH_TIMEOUT_SEC", "30"), default=30, lo=5, hi=240)
        max_tokens = _clamp_int(_env("SNOWBALL_ORCH_MAX_TOKENS", "1200"), default=1200, lo=64, hi=8000)

        draft_timeout = _clamp_int(_env("SNOWBALL_ORCH_TIMEOUT_DRAFT", "120"), default=120, lo=10, hi=600)
        critique_timeout = _clamp_int(_env("SNOWBALL_ORCH_TIMEOUT_CRITIQUE", "45"), default=45, lo=10, hi=300)
        revise_timeout = _clamp_int(_env("SNOWBALL_ORCH_TIMEOUT_REVISE", "75"), default=75, lo=10, hi=600)
        comp_timeout = _clamp_int(_env("SNOWBALL_ORCH_TIMEOUT_COMP", "90"), default=90, lo=10, hi=600)

        temperature = _clamp_float(_env("SNOWBALL_ORCH_TEMPERATURE", "0.7"), default=0.7, lo=0.0, hi=2.0)
        top_p = _clamp_float(_env("SNOWBALL_ORCH_TOP_P", "0.9"), default=0.9, lo=0.05, hi=1.0)

        retries = _clamp_int(_env("SNOWBALL_ORCH_RETRIES", "1"), default=1, lo=0, hi=5)
        retry_backoff = _clamp_float(_env("SNOWBALL_ORCH_RETRY_BACKOFF", "1.0"), default=1.0, lo=0.1, hi=10.0)

        vision_enabled = (_env("SNOWBALL_ENABLE_VISION", "1") == "1")
        vision_yolo_repo_dir = _env("SNOWBALL_VISION_YOLO_REPO", src.get("vision_yolo_repo_dir") if isinstance(src, dict) else None)
        vision_yolo_local_path = _env("SNOWBALL_VISION_YOLO_PATH", src.get("vision_yolo_local_path") if isinstance(src, dict) else None)
        vision_allow_hub_download = (_env("SNOWBALL_VISION_ALLOW_HUB", "0") == "1")
        vision_show_windows = (_env("SNOWBALL_VISION_SHOW_WINDOWS", "1") == "1")
        vision_camera_index = _clamp_int(_env("SNOWBALL_VISION_CAMERA_INDEX", "0"), default=0, lo=0, hi=10)

        return OrchestratorConfig(
            ollama_base_url=base_url,
            primary_model=primary,
            planner_model=planner,
            critic_model=critic,
            fallback_models=fallback_models,
            mode=mode,
            timeout_sec=timeout_sec,
            max_tokens=max_tokens,
            draft_timeout_sec=draft_timeout,
            critique_timeout_sec=critique_timeout,
            revise_timeout_sec=revise_timeout,
            competition_timeout_sec=comp_timeout,
            temperature=temperature,
            top_p=top_p,
            vision_enabled=vision_enabled,
            vision_yolo_repo_dir=vision_yolo_repo_dir,
            vision_yolo_local_path=vision_yolo_local_path,
            vision_allow_hub_download=vision_allow_hub_download,
            vision_show_windows=vision_show_windows,
            vision_camera_index=vision_camera_index,
            retries=retries,
            retry_backoff_sec=retry_backoff,
        )


if __name__ == "__main__":
    log = get_logger()
    orch = Orchestrator(logger=log)
    while True:
        try:
            u = input("> ").strip()
            if not u:
                continue
            if u.lower() in {"exit", "quit"}:
                break
            print(orch.respond(u))
        except KeyboardInterrupt:
            break
