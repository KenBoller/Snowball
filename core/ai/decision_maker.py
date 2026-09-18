"""
DecisionMaker (Unified, local-model friendly)

Goals:
- No TensorFlow/Keras usage (ZERO imports)
- No import-time model loading or disk IO
- Optional external QLearningAgent support (only if module exists AND enabled)
- Strong heuristics + clean logging + stable behavior
- Provider-agnostic (works with local model names like "deepseek-r1:14b")

UPGRAYEDD features preserved:
- extra_context-aware (dict-safe) scoring + RL payload
- prompt-echo / internal-context leak penalties
- stronger relevance scoring (token + bigram overlap + intent boosts)
- improved tie-break strategy

Unified packaging upgrades:
- ✅ package-first logger import (Snowball.core.system.logger) w/ legacy fallback
- ✅ reinforcement module import attempts package path first, then legacy
- ✅ optional provider_order can also come from account_integrations.json (keeps config centralized)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
import os
import sys
import importlib
import json
import re


# ----------------------------- Path helpers -----------------------------

def _snowball_root() -> str:
    if getattr(sys, "_MEIPASS", None):
        return str(getattr(sys, "_MEIPASS"))
    # Snowball/core/ai/decision_maker.py -> root is .../Snowball
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _load_account_integrations() -> Dict[str, Any]:
    path = os.path.join(_snowball_root(), "config", "account_integrations.json")
    return _safe_read_json(path)


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
    Package-first logger import, legacy fallback.
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


def _maybe_get_qlearning_agent(logger):
    """
    Optional external reinforcement agent.
    Enabled only if:
      SNOWBALL_ENABLE_QLEARNING=1

    Expected module provides:
      QLearningAgent(state_size, action_size, logger)

    Import order:
      1) Snowball.core.ai.reinforcement
      2) core.ai.reinforcement (legacy)
    """
    if os.getenv("SNOWBALL_ENABLE_QLEARNING") != "1":
        return None

    for mod_name in ("Snowball.core.ai.reinforcement", "core.ai.reinforcement"):
        try:
            reinforcement_module = importlib.import_module(mod_name)
            return reinforcement_module.QLearningAgent(state_size=10, action_size=8, logger=logger)
        except Exception:
            continue

    return None


# ----------------------------- Data -----------------------------

@dataclass
class DecisionMeta:
    query_type: str = "General"
    context_category: str = "Text"


# ----------------------------- DecisionMaker -----------------------------

class DecisionMaker:
    """
    Selects best response using stable heuristics.

    Signals (weighted):
    - relevance: token overlap + bigram overlap + intent keyword boost
    - quality: length sweet spot + structure + directness + low repetition
    - penalties: boilerplate/disclaimers, internal context leaks, prompt-echo, uncertainty-only
    - tie-break: provider order -> higher quality -> shorter/clearer
    """

    IRRELEVANT_PATTERNS = (
        "amazon snowball",
        "i'm back from a long break",
    )

    BOILERPLATE_PATTERNS = (
        "as an ai language model",
        "i can't browse the internet",
        "i cannot browse the internet",
        "i don't have personal experiences",
        "i do not have personal experiences",
    )

    LOW_QUALITY_PATTERNS = (
        "i don't know",
        "i do not know",
        "i'm not sure",
        "im not sure",
    )

    INTERNAL_CONTEXT_LEAK_PATTERNS = (
        "internal context (do not mention)",
        "do not mention",
        "system prompt",
        "these instructions",
    )

    CODE_INTENT_WORDS = ("code", "script", "python", "powershell", "cmd", "batch", "bash", "refactor", "rewrite")
    STRUCT_MARKERS = ("\n- ", "\n1.", "\n2.", "\n•", "\n\n")

    def __init__(self, logger=None, provider_order=None, **kwargs):
        self.logger = logger or get_logger()

        # 1) explicit arg wins
        # 2) env var next
        # 3) account_integrations.json (optional)
        # 4) defaults
        env_order = (os.getenv("SNOWBALL_PROVIDER_ORDER") or "").strip()

        cfg = _load_account_integrations()
        src = cfg.get("api_keys", cfg) if isinstance(cfg, dict) else {}
        cfg_order = None
        if isinstance(src, dict):
            cfg_order = src.get("provider_order") or src.get("provider_preference_order")

        if provider_order:
            self.provider_order = [str(x).strip() for x in provider_order if str(x).strip()]
        elif env_order:
            self.provider_order = [x.strip() for x in env_order.split(",") if x.strip()]
        elif isinstance(cfg_order, list):
            self.provider_order = [str(x).strip() for x in cfg_order if str(x).strip()]
        else:
            self.provider_order = [
                "deepseek-r1:14b",
                "qwen2.5:7b-instruct",
                "mistral:7b-instruct",
            ]

        self.reinforcement_agent = _maybe_get_qlearning_agent(self.logger)
        self._log("DecisionMaker", "INIT", f"Initialized. Provider order={self.provider_order}")

    # ------------------------- Public API -------------------------

    def select_best_response(
        self,
        responses: Dict[str, Optional[str]],
        user_input: str,
        query_type: str = "General",
        context_category: str = "Text",
        extra_context: Optional[Any] = None,
    ) -> str:
        valid = {k: v.strip() for k, v in responses.items() if isinstance(v, str) and v.strip()}
        if not valid:
            self._err("DecisionMaker", "No valid responses; returning fallback.")
            return "I'm unable to provide a meaningful response at the moment."

        if query_type == "General":
            query_type = self.detect_query_type(user_input=user_input, context_category=context_category)

        scores: Dict[str, float] = {}
        details: Dict[str, Dict[str, float]] = {}

        for provider, text in valid.items():
            s, d = self._score_response(
                provider=provider,
                text=text,
                user_input=user_input,
                query_type=query_type,
                context_category=context_category,
                extra_context=extra_context,
            )
            scores[provider] = s
            details[provider] = d

        chosen_provider = self._resolve_best(scores, details=details)
        chosen_text = valid[chosen_provider]

        self._log_unified_decision(
            user_input=user_input,
            query_type=query_type,
            context_category=context_category,
            chosen_model=chosen_provider,
            response=chosen_text,
            scores=scores,
            details=details,
        )

        self._learn_external({
            "user_input": user_input,
            "query_type": query_type,
            "context_category": context_category,
            "scores": scores,
            "details": details,
            "chosen": chosen_provider,
            "success": True,
            "extra_context": extra_context if isinstance(extra_context, (dict, list, tuple, str)) else None,
        })

        return chosen_text

    # ------------------------- Query type detection -------------------------

    def detect_query_type(self, user_input: str, context_category: str = "Text") -> str:
        ui = (user_input or "").lower()

        if (context_category or "").lower() == "vision":
            q = "Vision"
        elif any(w in ui for w in ["joke", "funny", "poem", "story", "creative"]):
            q = "Creative"
        elif any(w in ui for w in ["how", "what", "why", "when", "where", "explain", "difference", "compare"]):
            q = "Factual"
        elif context_category == "Voice":
            q = "Command"
        else:
            q = "General"

        self._log("DecisionMaker", "QueryType", f"{q} (context={context_category})")
        return q

    # ------------------------- Scoring -------------------------

    def _score_response(
        self,
        provider: str,
        text: str,
        user_input: str,
        query_type: str,
        context_category: str,
        extra_context: Optional[Any],
    ) -> Tuple[float, Dict[str, float]]:
        if not text:
            return -1.0, {"final": -1.0}

        tl = text.lower()
        uil = (user_input or "").lower()

        breakdown: Dict[str, float] = {}

        if any(p in tl for p in self.IRRELEVANT_PATTERNS):
            score = -0.90
            breakdown.update({"final": score, "hard_irrelevant": 1.0})
            self._log("DecisionMaker", "Score", f"{provider}: {score:.3f} (irrelevant pattern)")
            return score, breakdown

        internal_leak = 1.0 if any(p in tl for p in self.INTERNAL_CONTEXT_LEAK_PATTERNS) else 0.0

        pw = set(self._tokenize(user_input))
        tw = set(self._tokenize(text))
        overlap = len(pw & tw)
        denom = max(1, min(len(pw), len(tw)))
        overlap_ratio = overlap / denom

        pbg = set(self._bigrams(self._tokenize(user_input)))
        tbg = set(self._bigrams(self._tokenize(text)))
        bg_overlap = len(pbg & tbg)
        bg_denom = max(1, min(len(pbg), len(tbg)))
        bg_ratio = bg_overlap / bg_denom if bg_denom else 0.0

        intent_boost = 0.0
        if any(k in uil for k in ["steps", "checklist", "how to", "guide", "plan", "roadmap"]):
            if any(m in text for m in ["\n- ", "\n1.", "\n2.", "\n•"]):
                intent_boost = 0.12

        length = len(text)
        length_prior = 1.0 if 80 <= length <= 1600 else (0.7 if 40 <= length <= 2400 else 0.45)

        has_structure = any(m in text for m in self.STRUCT_MARKERS)
        structure_bonus = 0.18 if has_structure else 0.0

        repetition_penalty = self._repetition_penalty(text)

        wants_code = any(k in uil for k in self.CODE_INTENT_WORDS)
        looks_like_code = ("```" in text) or bool(re.search(r"\bdef\s+\w+\(|\bclass\s+\w+\(|\bimport\s+\w+", text))
        code_penalty = 0.25 if (looks_like_code and not wants_code) else 0.0

        boilerplate_penalty = 0.25 if any(p in tl for p in self.BOILERPLATE_PATTERNS) else 0.0

        uncertainty_penalty = 0.0
        if any(p in tl for p in self.LOW_QUALITY_PATTERNS):
            uncertainty_penalty = 0.9 if length < 180 else 0.45

        hedging_penalty = 0.0
        if query_type == "Factual" and any(h in tl for h in ["maybe", "i think", "probably", "might be"]):
            hedging_penalty = 0.20

        creative_short_penalty = 0.0
        if query_type == "Creative" and len(text.split()) < 10:
            creative_short_penalty = 0.60

        ignore_penalty = 0.0
        if overlap_ratio < 0.05 and length < 120:
            ignore_penalty = 0.35

        leak_penalty = 0.75 * internal_leak

        relevance = (overlap_ratio * 0.85) + (bg_ratio * 0.45) + intent_boost
        quality = (0.28 * length_prior) + structure_bonus

        penalties = (
            boilerplate_penalty
            + uncertainty_penalty
            + hedging_penalty
            + creative_short_penalty
            + ignore_penalty
            + repetition_penalty
            + leak_penalty
            + code_penalty
        )

        score = relevance + quality - penalties
        score = max(score, -0.99)

        breakdown.update({
            "final": score,
            "relevance_token": overlap_ratio,
            "relevance_bigram": bg_ratio,
            "intent_boost": intent_boost,
            "length_prior": length_prior,
            "structure_bonus": structure_bonus,
            "repetition_penalty": repetition_penalty,
            "boilerplate_penalty": boilerplate_penalty,
            "uncertainty_penalty": uncertainty_penalty,
            "hedging_penalty": hedging_penalty,
            "creative_short_penalty": creative_short_penalty,
            "ignore_penalty": ignore_penalty,
            "leak_penalty": leak_penalty,
            "code_penalty": code_penalty,
        })

        self._log("DecisionMaker", "Score", f"{provider}: {score:.3f}")
        return score, breakdown

    def _resolve_best(self, scores: Dict[str, float], details: Optional[Dict[str, Dict[str, float]]] = None) -> str:
        best_val = max(scores.values())
        tied = [k for k, v in scores.items() if v == best_val]

        if len(tied) == 1:
            return tied[0]

        if details:
            def tie_key(p: str) -> Tuple[float, float]:
                d = details.get(p, {})
                return (
                    float(d.get("relevance_token", 0.0)) + float(d.get("relevance_bigram", 0.0)),
                    float(d.get("structure_bonus", 0.0)) + float(d.get("length_prior", 0.0)),
                )
            tied_sorted = sorted(tied, key=tie_key, reverse=True)
            tied = [tied_sorted[0]] if tied_sorted else tied

        if len(tied) == 1:
            return tied[0]

        for p in self.provider_order:
            if p in tied:
                return p

        return sorted(tied)[0]

    # ------------------------- Optional external RL hook -------------------------

    def _learn_external(self, interaction: Dict[str, Any]) -> None:
        if not self.reinforcement_agent:
            return
        try:
            self.reinforcement_agent.learn_from_interaction(interaction)
        except Exception:
            pass

    # ------------------------- Tokenization -------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        return [t for t in tokens if len(t) >= 3]

    @staticmethod
    def _bigrams(tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        return [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]

    @staticmethod
    def _repetition_penalty(text: str) -> float:
        if not text or not isinstance(text, str):
            return 0.0

        lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
        if len(lines) <= 3:
            return 0.0

        unique = len(set(lines))
        rep_ratio = 1.0 - (unique / max(1, len(lines)))

        words = re.findall(r"[a-z0-9_]+", text.lower())
        burst = 0
        for i in range(2, len(words)):
            if words[i] == words[i - 1] == words[i - 2]:
                burst += 1

        penalty = 0.0
        if rep_ratio > 0.25:
            penalty += min(0.35, rep_ratio)

        if burst > 0:
            penalty += min(0.25, 0.05 * burst)

        return penalty

    # ------------------------- Logging -------------------------

    def _log_unified_decision(
        self,
        user_input: str,
        query_type: str,
        context_category: str,
        chosen_model: str,
        response: str,
        scores: Dict[str, float],
        details: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        entry = {
            "user_input": user_input,
            "query_type": query_type,
            "context_category": context_category,
            "chosen_model": chosen_model,
            "scores": scores,
            "details": details or {},
            "response_preview": response[:220],
        }
        try:
            self.logger.log_decision(json.dumps(entry, ensure_ascii=False))
        except Exception:
            self._log("DecisionMaker", "Decision", str(entry))

    def _log(self, category: str, event_type: str, message: str) -> None:
        try:
            self.logger.log_event(category, event_type, message)
        except Exception:
            try:
                self.logger.log_event(f"{category} | {event_type} | {message}")
            except Exception:
                pass

    def _err(self, category: str, message: str) -> None:
        try:
            self.logger.log_error(category, message)
        except Exception:
            try:
                self.logger.log_error(f"{category}: {message}")
            except Exception:
                pass


# ----------------------------- Backwards compat utility -----------------------------

def validate_response_quality(response: Optional[str]) -> bool:
    if not response or len(response.strip()) < 10:
        return False
    if "i don't know" in response.lower():
        return False
    return True
