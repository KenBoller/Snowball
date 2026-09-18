# Snowball/core/ai/memory.py
"""
Snowball Memory Manager — merged + improved (MVP + UPGRAYEDD)

Default behavior (SAFE):
- Local JSONL persistence: ON (survives restarts)
- CosmosDB: OFF unless SNOWBALL_ENABLE_COSMOS=1 and credentials + SDK installed
- Embeddings/semantic search: OFF unless SNOWBALL_ENABLE_EMBEDDINGS=1

Key improvements implemented:
1) ✅ No duplicate in-memory appends
   - memory_docs: ALL docs (interaction, file_meta, file_analysis)
   - semantic_docs + semantic_vectors: ONLY embedded docs (aligned)
2) ✅ Stable Snowball root + config resolution
   - Prefers: <SNOWBALL_ROOT>/config/account_integrations.json
   - Legacy fallback: S:/Snowball/config/account_integrations.json
   - NO dependency on api_keys.json
3) ✅ Stable local memory dir default
   - Prefers SNOWBALL_LOCAL_MEMORY_DIR
   - Else uses: <SNOWBALL_ROOT>/storage/memory/local_memory.jsonl
4) ✅ Optional Cosmos partition-key hashing
   - If SNOWBALL_COSMOS_HASH_PARTITION=1:
     - doc["user_input"] becomes pk:<hash>
     - original text preserved in doc["user_text"] / doc["assistant_text"] when present
5) ✅ Optional local dedupe to avoid accidental double writes
   - SNOWBALL_LOCAL_DEDUPE=1
6) ✅ File ingest support
   - store_file_analysis(file_path, extracted_text, meta=None)
   - store_file_metadata(file_path, meta_dict)  # preferred
   - Backward compatible: store_file_metadata(file_name, file_path, last_modified, ...)
7) ✅ Embeddings gracefully degrade
   - numpy / sentence-transformers / faiss are optional and gated
8) ✅ Keeps MVP retrieval helpers
   - get_last_interaction, get_all_interactions, get_interactions
   - get_memory (semantic-first then exact match when Cosmos enabled)
   - search_files, search_files_by_tags (Cosmos-backed; local returns [])

Env flags:
- SNOWBALL_DISABLE_LOCAL_MEMORY=1
- SNOWBALL_ENABLE_COSMOS=1
- SNOWBALL_ENABLE_EMBEDDINGS=1
- SNOWBALL_ENABLE_FAISS=1
- SNOWBALL_LOCAL_MEMORY_DIR=...
- SNOWBALL_ROOT=... (optional)
- SNOWBALL_COSMOS_HASH_PARTITION=0/1 (default 0)
- SNOWBALL_LOCAL_DEDUPE=0/1 (default 0)

Cosmos creds:
- COSMOSDB_URI, COSMOSDB_KEY (env), OR account_integrations.json keys:
  - cosmosdb_uri, cosmosdb_key
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from cachetools import LRUCache

# ------------------------- Feature flags (SAFE defaults) -------------------------

_ENABLE_EMBEDDINGS = os.getenv("SNOWBALL_ENABLE_EMBEDDINGS") == "1"
_ENABLE_FAISS = os.getenv("SNOWBALL_ENABLE_FAISS") == "1"
_ENABLE_COSMOS = os.getenv("SNOWBALL_ENABLE_COSMOS") == "1"

_COSMOS_HASH_PARTITION = os.getenv("SNOWBALL_COSMOS_HASH_PARTITION") == "1"
_LOCAL_DEDUPE = os.getenv("SNOWBALL_LOCAL_DEDUPE") == "1"

# ------------------------- Optional deps (graceful fallbacks) -------------------------

_has_np = False
np = None  # type: ignore
if _ENABLE_EMBEDDINGS:
    try:
        import numpy as np  # type: ignore
        _has_np = True
    except Exception:
        _has_np = False

_has_faiss = False
faiss = None  # type: ignore
if _ENABLE_EMBEDDINGS and _ENABLE_FAISS:
    try:
        import faiss  # type: ignore
        _has_faiss = True
    except Exception:
        _has_faiss = False

_has_st = False
SentenceTransformer = None  # type: ignore
if _ENABLE_EMBEDDINGS:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _has_st = True
    except Exception:
        _has_st = False

_has_cosmos = True
try:
    from azure.cosmos import CosmosClient, exceptions  # type: ignore
    from azure.cosmos.partition_key import PartitionKey  # type: ignore
except Exception:
    _has_cosmos = False

    class _Ex:
        class CosmosHttpResponseError(Exception): ...
        class CosmosResourceNotFoundError(Exception): ...

    exceptions = _Ex()  # type: ignore
    CosmosClient = None  # type: ignore
    PartitionKey = None  # type: ignore


# ------------------------- Logging helpers -------------------------

def get_logger():
    try:
        from Snowball.core.system.logger import SnowballLogger  # ✅ correct
        return SnowballLogger()
    except Exception:
        class _NullLogger:
            def log_event(self, *a, **k): pass
            def log_error(self, *a, **k): pass
            def log_warning(self, *a, **k): pass
            def log_memory(self, *a, **k): pass
            def log_decision(self, *a, **k): pass
        return _NullLogger()


def _log(logger, category: str, event_type: str, message: str):
    try:
        logger.log_event(category=category, event_type=event_type, message=message)
    except Exception:
        try:
            logger.log_event(f"{category} | {event_type} | {message}")
        except Exception:
            pass


def _warn(logger, category: str, message: str):
    try:
        logger.log_warning(category, message)
    except Exception:
        try:
            logger.log_event(category=category, event_type="WARN", message=message, severity="WARNING")
        except Exception:
            pass


def _err(logger, category: str, message: str):
    try:
        logger.log_error(category, message)
    except Exception:
        try:
            logger.log_event(category=category, event_type="ERROR", message=message, severity="ERROR")
        except Exception:
            pass


# ------------------------- Path + config helpers -------------------------

def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _merge_first_wins(dest: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if k not in dest and v is not None:
            dest[k] = v


def _sha1(s: str) -> str:
    b = (s or "").encode("utf-8", errors="ignore")
    return hashlib.sha1(b).hexdigest()


def _snowball_root() -> str:
    """
    Resolve Snowball root directory.

    Priority:
    1) SNOWBALL_ROOT env var
    2) Two levels up from this file (core/ai -> core -> Snowball)
    3) Best-effort fallback: current working directory
    """
    env_root = (os.getenv("SNOWBALL_ROOT") or "").strip()
    if env_root:
        return os.path.abspath(env_root)

    # memory.py assumed at: <root>/core/ai/memory.py
    candidate = os.path.abspath(os.path.join(_here(), "..", ".."))
    if os.path.isdir(candidate):
        return candidate

    return os.path.abspath(os.getcwd())


def _account_integrations_paths() -> List[str]:
    root = _snowball_root()
    return [
        os.path.join(root, "config", "account_integrations.json"),
        # legacy absolute fallback you mentioned
        "S:/Snowball/config/account_integrations.json",
    ]


def load_cosmosdb_credentials(paths: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (uri, key) from env or account_integrations.json.

    Env:
      - COSMOSDB_URI
      - COSMOSDB_KEY

    JSON keys supported:
      - cosmosdb_uri
      - cosmosdb_key

    Accepts either:
      { "cosmosdb_uri": "...", "cosmosdb_key": "..." }
    or:
      { "api_keys": { "cosmosdb_uri": "...", "cosmosdb_key": "..." } }
    """
    uri = (os.getenv("COSMOSDB_URI") or "").strip() or None
    key = (os.getenv("COSMOSDB_KEY") or "").strip() or None

    merged: Dict[str, Any] = {}
    for p in (paths or _account_integrations_paths()):
        data = _read_json(p)
        if not data:
            continue
        src = data.get("api_keys", data)
        if isinstance(src, dict):
            _merge_first_wins(merged, src)

    if not uri:
        v = merged.get("cosmosdb_uri")
        uri = v.strip() if isinstance(v, str) and v.strip() else None
    if not key:
        v = merged.get("cosmosdb_key")
        key = v.strip() if isinstance(v, str) and v.strip() else None

    return uri, key


@dataclass
class MemoryPaths:
    local_dir: str
    jsonl_path: str


def _resolve_local_paths() -> MemoryPaths:
    """
    Default local memory dir:
      1) SNOWBALL_LOCAL_MEMORY_DIR env var
      2) <SNOWBALL_ROOT>/storage/memory/

    JSONL file:
      <dir>/local_memory.jsonl
    """
    env_dir = (os.getenv("SNOWBALL_LOCAL_MEMORY_DIR") or "").strip()
    if env_dir:
        local_dir = os.path.abspath(env_dir)
    else:
        local_dir = os.path.join(_snowball_root(), "storage", "memory")

    _ensure_dir(local_dir)
    return MemoryPaths(local_dir=local_dir, jsonl_path=os.path.join(local_dir, "local_memory.jsonl"))


# ------------------------- Memory -------------------------

class Memory:
    """
    Snowball Memory Manager

    Storage:
      - Local JSONL (default ON)
      - Cosmos (OFF unless enabled + configured)

    Semantic search:
      - Requires embeddings enabled + numpy + sentence-transformers
      - Optional FAISS

    Data:
      - memory_docs: ALL docs loaded/created (interactions, file_meta, file_analysis)
      - semantic_docs + semantic_vectors: ONLY embedded docs (aligned indices)
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        _log(self.logger, "Memory", "INIT", "Initializing Memory module")

        self._lock = threading.RLock()

        # Local persistence
        paths = _resolve_local_paths()
        self.local_dir = paths.local_dir
        self.local_jsonl_path = paths.jsonl_path
        self.local_enabled = os.getenv("SNOWBALL_DISABLE_LOCAL_MEMORY") != "1"

        # Cosmos (optional)
        self.cosmosdb_uri, self.cosmosdb_key = load_cosmosdb_credentials()
        self.cosmos_client = None
        self.cosmos_container = None
        self._init_cosmosdb()

        # Embeddings / FAISS (optional)
        self.embedding_model = None  # SentenceTransformer
        self.faiss_index = None      # faiss IndexFlatL2

        # ✅ Split docs
        self.memory_docs: List[Dict[str, Any]] = []
        self.semantic_docs: List[Dict[str, Any]] = []
        self.semantic_vectors: List[Any] = []

        # Cache
        self.cache = LRUCache(maxsize=500)

        # Optional local dedupe
        self._recent_sig_cache = LRUCache(maxsize=5000) if _LOCAL_DEDUPE else None

        # Archiving
        self.archiving_active = False

        # Load local history
        if self.local_enabled:
            self._load_local_jsonl(limit=3000)

        emb_ok = bool(_ENABLE_EMBEDDINGS and _has_st and _has_np)
        faiss_ok = bool(_ENABLE_FAISS and _has_faiss and emb_ok)

        _log(
            self.logger,
            "Memory",
            "FEATURES",
            f"local={self.local_enabled} cosmos={bool(self.cosmos_container)} "
            f"embeddings={emb_ok} faiss={faiss_ok} cosmos_hash_pk={_COSMOS_HASH_PARTITION} local_dedupe={_LOCAL_DEDUPE} "
            f"local_dir={self.local_dir}"
        )

        if _ENABLE_EMBEDDINGS and not _has_np:
            _warn(self.logger, "Memory", "numpy missing; embeddings disabled even though SNOWBALL_ENABLE_EMBEDDINGS=1")
        if _ENABLE_EMBEDDINGS and not _has_st:
            _warn(self.logger, "Memory", "sentence-transformers missing; embeddings disabled even though SNOWBALL_ENABLE_EMBEDDINGS=1")

        _log(self.logger, "Memory", "READY", "Memory ready")

    # ------------------------- Cosmos init -------------------------

    def _init_cosmosdb(self) -> None:
        if not _ENABLE_COSMOS:
            _log(self.logger, "Memory", "COSMOS", "Cosmos disabled (SNOWBALL_ENABLE_COSMOS!=1). Local-only mode.")
            return
        if not _has_cosmos:
            _log(self.logger, "Memory", "COSMOS", "Azure Cosmos SDK not installed. Local-only mode.")
            return
        if not (self.cosmosdb_uri and self.cosmosdb_key):
            _log(self.logger, "Memory", "COSMOS", "No CosmosDB credentials. Local-only mode.")
            return

        try:
            self.cosmos_client = CosmosClient(self.cosmosdb_uri, credential=self.cosmosdb_key)

            # Ensure database
            try:
                database = self.cosmos_client.get_database_client("snowballDB")
                database.read()
            except exceptions.CosmosHttpResponseError:
                database = self.cosmos_client.create_database_if_not_exists(id="snowballDB")

            # Ensure container (your setup uses partition key /user_input)
            try:
                self.cosmos_container = database.get_container_client("memoryContainer")
                self.cosmos_container.read()
            except exceptions.CosmosHttpResponseError:
                self.cosmos_container = database.create_container_if_not_exists(
                    id="memoryContainer",
                    partition_key=PartitionKey(path="/user_input") if PartitionKey else None,
                )

            _log(self.logger, "Memory", "COSMOS", "Connected to snowballDB/memoryContainer.")
        except Exception as e:
            _err(self.logger, "Memory", f"Cosmos init error: {e}")
            self.cosmos_client = None
            self.cosmos_container = None

    # ------------------------- Embeddings / FAISS -------------------------

    def _load_embedding_model(self) -> None:
        if self.embedding_model is not None:
            return
        if not (_ENABLE_EMBEDDINGS and _has_st and _has_np):
            return
        try:
            _log(self.logger, "Memory", "EMBED", "Loading all-MiniLM-L6-v2...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # type: ignore
            _log(self.logger, "Memory", "EMBED", "Loaded sentence-transformer.")
        except Exception as e:
            _err(self.logger, "Memory", f"Embedding model load failed: {e}")
            self.embedding_model = None

    def _ensure_faiss_index(self, dim: int) -> None:
        if not (_ENABLE_EMBEDDINGS and _ENABLE_FAISS and _has_faiss and _has_np and _has_st):
            return
        if self.faiss_index is None:
            try:
                self.faiss_index = faiss.IndexFlatL2(dim)  # type: ignore
            except Exception as e:
                _err(self.logger, "Memory", f"FAISS init failed: {e}")
                self.faiss_index = None

    def _embed_text(self, text: str):
        if not (_ENABLE_EMBEDDINGS and _has_st and _has_np):
            return None
        self._load_embedding_model()
        if self.embedding_model is None:
            return None
        try:
            vec = self.embedding_model.encode(text)  # type: ignore
            return np.asarray(vec, dtype=np.float32)  # type: ignore
        except Exception as e:
            _err(self.logger, "Memory", f"Embedding error: {e}")
            return None

    # ------------------------- Public API (chat interactions) -------------------------

    def store_interaction(self, *args, **kwargs) -> None:
        """
        Backward-compatible:
        - New style: store_interaction(user_input=..., ai_response=..., query_type="General")
        - Old style: store_interaction("user", text) and store_interaction("assistant", text)
        """
        with self._lock:
            # old style: (role, content)
            if len(args) >= 2 and isinstance(args[0], str) and args[0] in {"user", "assistant"}:
                role, content = args[0], args[1]
                doc = {
                    "id": str(uuid.uuid4()),
                    "type": "interaction",
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                    # Cosmos partition key safety baseline
                    "user_input": "__system__",
                }
                self._write_doc(doc)

                if isinstance(content, str) and content.strip():
                    self._add_to_semantic_index(content, doc)
                return

            # new style
            user_input = kwargs.get("user_input", args[0] if len(args) > 0 else None)
            ai_response = kwargs.get("ai_response", args[1] if len(args) > 1 else None)
            query_type = kwargs.get("query_type", "General")

            if user_input is None and ai_response is None:
                _err(self.logger, "Memory", "store_interaction called without data.")
                return

            doc = {
                "id": str(uuid.uuid4()),
                "type": "interaction",
                "user_input": user_input if user_input is not None else "__system__",
                "ai_response": ai_response,
                "query_type": query_type,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._write_doc(doc)

            text_for_index = user_input if isinstance(user_input, str) and user_input.strip() else (ai_response or "")
            if isinstance(text_for_index, str) and text_for_index.strip():
                self._add_to_semantic_index(text_for_index, doc)

    def store_memory(self, user_input: str, ai_response: str) -> None:
        self.store_interaction(user_input=user_input, ai_response=ai_response, query_type="General")

    # ------------------------- Public API (file ingest) -------------------------

    def store_file_analysis(self, file_path: str, extracted_text: str, meta: Optional[dict] = None) -> None:
        """
        Store extracted text for a file (preferred for FileManager after extraction).
        """
        meta = meta or {}
        fp = str(file_path)

        doc = {
            "id": str(uuid.uuid4()),
            "type": "file_analysis",
            "file_path": fp,
            "file_name": os.path.basename(fp),
            "ext": os.path.splitext(fp)[1].lower(),
            "meta": meta,
            "text": extracted_text,
            "text_len": len(extracted_text or ""),
            "checksum": self.compute_checksum(fp),
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": "__system__",  # cosmos partition safety baseline
        }
        self._write_doc(doc)

        # Optional semantic index (bounded)
        if isinstance(extracted_text, str) and extracted_text.strip():
            self._add_to_semantic_index(extracted_text[:4000], doc)

    def store_file_metadata(self, *args, **kwargs) -> None:
        """
        Backward compatible:

        NEW signature (preferred):
          store_file_metadata(file_path, meta_dict)

        OLD signature (MVP):
          store_file_metadata(file_name, file_path, last_modified, file_size=None, tags=None, analysis_result=None)
        """
        # New signature
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            self._store_file_meta_v2(file_path=args[0], meta=args[1])
            return

        if "file_path" in kwargs and "meta" in kwargs and isinstance(kwargs["meta"], dict):
            self._store_file_meta_v2(file_path=str(kwargs["file_path"]), meta=kwargs["meta"])
            return

        # Old signature
        file_name = kwargs.get("file_name", args[0] if len(args) > 0 else None)
        file_path = kwargs.get("file_path", args[1] if len(args) > 1 else None)
        last_modified = kwargs.get("last_modified", args[2] if len(args) > 2 else None)
        file_size = kwargs.get("file_size", None)
        tags = kwargs.get("tags", None)
        analysis_result = kwargs.get("analysis_result", None)

        if not file_path:
            _warn(self.logger, "Memory", "store_file_metadata called without file_path; skipping.")
            return

        fp = str(file_path)
        doc = {
            "id": str(uuid.uuid4()),
            "type": "file_meta",
            "name": str(file_name) if file_name else os.path.basename(fp),
            "path": fp,
            "last_modified": str(last_modified) if last_modified is not None else "",
            "file_size": file_size,
            "tags": tags or [],
            "analysis_result": analysis_result,
            "checksum": self.compute_checksum(fp),
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": "__system__",  # cosmos partition safety baseline
        }
        self._write_doc(doc)

    def _store_file_meta_v2(self, file_path: str, meta: dict) -> None:
        fp = str(file_path)
        doc = {
            "id": str(uuid.uuid4()),
            "type": "file_meta",
            "name": meta.get("file_name") or os.path.basename(fp),
            "path": fp,
            "last_modified": meta.get("modified_iso") or meta.get("last_modified") or "",
            "file_size": meta.get("size_bytes") or meta.get("file_size"),
            "tags": meta.get("tags") or [],
            "analysis_result": meta.get("analysis_result"),
            "meta": meta,
            "checksum": self.compute_checksum(fp),
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": "__system__",  # cosmos partition safety baseline
        }
        self._write_doc(doc)

    # ------------------------- Retrieval helpers -------------------------

    def get_last_interaction(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            cached = self.cache.get("last_interaction")
            if cached is not None:
                return cached

        if self.cosmos_container:
            try:
                query = "SELECT TOP 1 * FROM c WHERE c.type = 'interaction' ORDER BY c.timestamp DESC"
                items = list(self.cosmos_container.query_items(query=query, enable_cross_partition_query=True))
                if items:
                    with self._lock:
                        self.cache["last_interaction"] = items[0]
                    return items[0]
            except Exception as e:
                _err(self.logger, "Memory", f"Cosmos last_interaction error: {e}")

        with self._lock:
            for doc in reversed(self.memory_docs):
                if doc.get("type") == "interaction":
                    self.cache["last_interaction"] = doc
                    return doc
        return None

    def get_all_interactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        limit = int(limit)
        if self.cosmos_container:
            try:
                query = f"SELECT TOP {limit} * FROM c WHERE c.type = 'interaction' ORDER BY c.timestamp DESC"
                return list(self.cosmos_container.query_items(query=query, enable_cross_partition_query=True))
            except Exception as e:
                _err(self.logger, "Memory", f"Cosmos get_all_interactions error: {e}")

        with self._lock:
            data = [d for d in self.memory_docs if d.get("type") == "interaction"]
            return list(reversed(data[-limit:]))

    def get_interactions(
        self,
        query_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        MVP-compatible filtering. Local mode filters memory_docs. Cosmos mode runs a query.
        """
        limit = int(limit)

        # Local filtering
        if not self.cosmos_container:
            with self._lock:
                data = [d for d in self.memory_docs if d.get("type") == "interaction"]
            if query_type:
                data = [d for d in data if d.get("query_type") == query_type]
            if start_time:
                data = [d for d in data if str(d.get("timestamp", "")) >= start_time]
            if end_time:
                data = [d for d in data if str(d.get("timestamp", "")) <= end_time]
            return list(reversed(data[-limit:]))

        # Cosmos filtering
        q = f"SELECT TOP {limit} * FROM c WHERE c.type = 'interaction'"
        params = []

        if query_type:
            q += " AND c.query_type = @qt"
            params.append({"name": "@qt", "value": query_type})
        if start_time:
            q += " AND c.timestamp >= @st"
            params.append({"name": "@st", "value": start_time})
        if end_time:
            q += " AND c.timestamp <= @et"
            params.append({"name": "@et", "value": end_time})

        q += " ORDER BY c.timestamp DESC"

        try:
            items = list(
                self.cosmos_container.query_items(
                    query=q,
                    parameters=params,
                    enable_cross_partition_query=True
                )
            )
            _log(self.logger, "Memory", "QUERY", f"Returned {len(items)} items.")
            return items
        except Exception as e:
            _err(self.logger, "Memory", f"Cosmos get_interactions error: {e}")
            return []

    def get_recent_file_ingests(self, limit: int = 25) -> List[Dict[str, Any]]:
        limit = int(limit)
        with self._lock:
            data = [d for d in self.memory_docs if d.get("type") in {"file_meta", "file_analysis"}]
            return list(reversed(data[-limit:]))

    def get_memory(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most relevant memory for user_input.

        Retrieval order:
        1. Semantic search, when embeddings are enabled.
        2. Lightweight local text search over persisted interactions.
        3. Cosmos exact-match, when Cosmos is enabled.

        Local retrieval intentionally requires no external services or
        embedding models so Snowball can remember across restarts by default.
        """
        user_input = (user_input or "").strip()
        if not user_input:
            return None

        # 1) Semantic search when available.
        ss = self.semantic_search(user_input, top_k=1)
        if ss:
            _log(
                self.logger,
                "Memory",
                "SEMANTIC",
                f"Semantic match for: {user_input}",
            )
            return ss[0]

        # 2) Local fallback.
        #
        # Compare useful words from the query against both sides of previous
        # interactions. More overlapping words = a more relevant memory.
        query_words = set(re.findall(r"[a-z0-9]+", user_input.lower()))
        query_words -= {
            "a", "about", "an", "and", "are", "as", "at", "be", "but", "by",
            "for", "from", "how", "i", "in", "is", "it", "me", "my",
            "of", "on", "or", "tell", "that", "the", "this", "to", "was",
            "what", "when", "where", "who", "why", "with", "you",
        }

        best_doc = None
        best_score = 0

        if query_words:
            with self._lock:
                docs = list(self.memory_docs)

            # Newest memories win ties.
            for doc in reversed(docs):
                if doc.get("type") != "interaction":
                    continue

                memory_user = str(doc.get("user_input", "")).lower()
                memory_response = str(doc.get("ai_response", "")).lower()

                user_words = set(re.findall(r"[a-z0-9]+", memory_user))
                response_words = set(re.findall(r"[a-z0-9]+", memory_response))

                user_overlap = len(query_words & user_words)
                response_overlap = len(query_words & response_words)

                # Words appearing in the user's original message are much stronger
                # evidence than words merely appearing somewhere in Snowball's reply.
                score = (user_overlap * 5) + min(response_overlap, 1)

                # User-provided statements are stronger memory evidence than old questions.
                # This prevents Snowball's previous answers from becoming self-reinforcing
                # "facts" simply because the same question was asked before.
                question_starters = (
                    "what ", "who ", "where ", "when ", "why ", "how ",
                    "is ", "are ", "was ", "were ", "do ", "does ", "did ",
                    "can ", "could ", "would ", "should ", "will "
                )

                memory_is_question = (
                    memory_user.strip().endswith("?")
                    or memory_user.strip().startswith(question_starters)
                )

                if not memory_is_question:
                    score += 5
                    
                if score > best_score:
                    best_score = score
                    best_doc = doc

        if best_doc is not None:
            _log(
                self.logger,
                "Memory",
                "LOCAL",
                f"Local match score={best_score} for: {user_input}",
            )
            return best_doc

        # 3) Cosmos exact match when available.
        if self.cosmos_container:
            try:
                q = (
                    "SELECT TOP 1 * FROM c "
                    "WHERE c.type = 'interaction' "
                    "AND c.user_input = @u"
                )
                items = list(
                    self.cosmos_container.query_items(
                        query=q,
                        parameters=[{"name": "@u", "value": user_input}],
                        enable_cross_partition_query=True,
                    )
                )
                if items:
                    return items[0]
            except Exception as e:
                _err(
                    self.logger,
                    "Memory",
                    f"Cosmos exact-match error: {e}",
                )

        return None
    # ------------------------- Semantic search -------------------------

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        vec = self._embed_text(query)
        if vec is None:
            return []

        with self._lock:
            if not self.semantic_docs:
                return []

        # FAISS path (aligned)
        if _ENABLE_EMBEDDINGS and _ENABLE_FAISS and _has_faiss and _has_np:
            with self._lock:
                idx = self.faiss_index
            if idx is not None:
                try:
                    qv = np.expand_dims(vec, axis=0).astype(np.float32)  # type: ignore
                    _, idxs = idx.search(qv, int(top_k))
                    out: List[Dict[str, Any]] = []
                    with self._lock:
                        for i in idxs[0]:
                            if 0 <= int(i) < len(self.semantic_docs):
                                out.append(self.semantic_docs[int(i)])
                    return out
                except Exception as e:
                    _err(self.logger, "Memory", f"FAISS search error: {e}")

        # cosine fallback (aligned)
        with self._lock:
            vectors = list(self.semantic_vectors)
            docs = list(self.semantic_docs)

        if not vectors:
            return []

        try:
            qn = vec / (np.linalg.norm(vec) + 1e-12)  # type: ignore
            sims = []
            for i, v in enumerate(vectors):
                vn = v / (np.linalg.norm(v) + 1e-12)  # type: ignore
                sims.append((float(np.dot(qn, vn)), i))  # type: ignore
            sims.sort(reverse=True, key=lambda x: x[0])
            return [docs[i] for _, i in sims[: int(top_k)]]
        except Exception as e:
            _err(self.logger, "Memory", f"Cosine search error: {e}")
            return []

    # ------------------------- File search helpers (Cosmos-backed) -------------------------

    def search_files(self, keyword: str) -> List[Dict[str, Any]]:
        """
        MVP-compatible: Cosmos-backed search over file_meta.
        Local mode returns [] (no heavy indexing over JSONL).
        """
        if not self.cosmos_container:
            return []
        try:
            q = "SELECT * FROM c WHERE c.type = 'file_meta' AND (CONTAINS(c.name, @k) OR CONTAINS(c.path, @k))"
            items = list(
                self.cosmos_container.query_items(
                    query=q,
                    parameters=[{"name": "@k", "value": keyword}],
                    enable_cross_partition_query=True
                )
            )
            _log(self.logger, "Memory", "FILES", f"keyword '{keyword}' -> {len(items)}")
            return items
        except Exception as e:
            _err(self.logger, "Memory", f"File search error: {e}")
            return []

    def search_files_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """
        MVP-compatible: Cosmos-backed tag search over file_meta tags.
        Local mode returns [].
        """
        if not (self.cosmos_container and tags):
            return []
        try:
            cond = " OR ".join([f"ARRAY_CONTAINS(c.tags, @t{i})" for i in range(len(tags))])
            params = [{"name": f"@t{i}", "value": t} for i, t in enumerate(tags)]
            q = f"SELECT * FROM c WHERE c.type = 'file_meta' AND ({cond})"
            items = list(
                self.cosmos_container.query_items(
                    query=q,
                    parameters=params,
                    enable_cross_partition_query=True
                )
            )
            _log(self.logger, "Memory", "FILES", f"tags {tags} -> {len(items)}")
            return items
        except Exception as e:
            _err(self.logger, "Memory", f"Tag search error: {e}")
            return []

    # ------------------------- File metadata helpers -------------------------

    def compute_checksum(self, file_path: str) -> str:
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            _err(self.logger, "Memory", f"Checksum error for {file_path}: {e}")
            return "UNKNOWN"

    # ------------------------- Archiving -------------------------

    def start_auto_archiving(self, interval: int = 86400) -> None:
        if self.archiving_active:
            _log(self.logger, "Memory", "ARCHIVE", "Already running.")
            return
        self.archiving_active = True

        def _task():
            while self.archiving_active:
                try:
                    self.archive_old_interactions()
                except Exception as e:
                    _err(self.logger, "Memory", f"Archive tick error: {e}")
                time.sleep(max(60, int(interval)))

        threading.Thread(target=_task, daemon=True).start()
        _log(self.logger, "Memory", "ARCHIVE", "Auto-archiving started.")

    def archive_old_interactions(self) -> None:
        _log(self.logger, "Memory", "ARCHIVE", "archive_old_interactions() tick (no-op).")

    def close(self) -> None:
        if self.cosmos_client:
            try:
                self.cosmos_client.close()
                _log(self.logger, "Memory", "CLOSE", "Closed Cosmos connection.")
            except Exception:
                pass

    # ------------------------- Internal: persistence + indexing -------------------------

    def _doc_signature(self, doc: Dict[str, Any]) -> str:
        """
        Used only when SNOWBALL_LOCAL_DEDUPE=1.
        Signature tries to ignore volatile fields like timestamp/id.
        """
        try:
            core = {
                "type": doc.get("type"),
                "user_input": doc.get("user_input"),
                "ai_response": doc.get("ai_response"),
                "role": doc.get("role"),
                "content": doc.get("content"),
                "file_path": doc.get("file_path") or doc.get("path"),
                "checksum": doc.get("checksum"),
                "text_len": doc.get("text_len"),
            }
            raw = json.dumps(core, ensure_ascii=False, sort_keys=True)
            return _sha1(raw)
        except Exception:
            return _sha1(str(doc))

    def _normalize_doc_for_cosmos(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cosmos container uses partition key '/user_input' (your existing setup).
        Optionally hash partition key to avoid storing raw user text in the PK.
        """
        if "id" not in doc or not doc["id"]:
            doc["id"] = str(uuid.uuid4())

        ui = doc.get("user_input")
        if ui is None:
            ui = "__system__"

        # Preserve assistant text in a consistent key for hashing mode (optional)
        assistant_text = doc.get("ai_response")
        if assistant_text is None and doc.get("role") == "assistant":
            assistant_text = doc.get("content")

        if _COSMOS_HASH_PARTITION:
            if isinstance(ui, str) and ui not in {"__system__", ""}:
                doc.setdefault("user_text", ui)

            if isinstance(assistant_text, str) and assistant_text.strip():
                doc.setdefault("assistant_text", assistant_text)

            doc["user_input"] = f"pk:{_sha1(str(ui))[:16]}"
        else:
            doc["user_input"] = ui if isinstance(ui, str) else "__system__"

        return doc

    def _append_local_jsonl(self, doc: Dict[str, Any]) -> None:
        if not self.local_enabled:
            return
        try:
            line = json.dumps(doc, ensure_ascii=False)
            with self._lock:
                with open(self.local_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            _err(self.logger, "Memory", f"Local JSONL write error: {e}")

    def _load_local_jsonl(self, limit: int = 3000) -> None:
        if not os.path.exists(self.local_jsonl_path):
            return
        try:
            with open(self.local_jsonl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-int(limit):]
        except Exception:
            return

        loaded = 0
        with self._lock:
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    doc = json.loads(ln)
                except Exception:
                    continue

                if doc.get("type") not in {"interaction", "file_meta", "file_analysis"}:
                    continue

                self.memory_docs.append(doc)
                loaded += 1

        if loaded:
            _log(self.logger, "Memory", "LOCAL", f"Loaded {loaded} docs from local JSONL.")

    def _write_doc(self, doc: Dict[str, Any]) -> None:
        """
        Single source of truth:
        - Append once to memory_docs
        - Write to local JSONL
        - Write to cosmos (optional)
        - Update caches
        """
        # Optional dedupe (local + memory) to avoid accidental double writes
        if _LOCAL_DEDUPE and self._recent_sig_cache is not None:
            sig = self._doc_signature(doc)
            with self._lock:
                if sig in self._recent_sig_cache:
                    return
                self._recent_sig_cache[sig] = True

        with self._lock:
            self.memory_docs.append(doc)

            if doc.get("type") == "interaction":
                self.cache["last_interaction"] = doc

        self._append_local_jsonl(doc)

        if self.cosmos_container:
            try:
                cdoc = self._normalize_doc_for_cosmos(dict(doc))
                self.cosmos_container.create_item(cdoc)
            except Exception as e:
                _err(self.logger, "Memory", f"Cosmos write error: {e}")

    def _add_to_semantic_index(self, text: str, doc: Dict[str, Any]) -> None:
        """
        Does NOT append doc to memory_docs.
        Only adds embedding + semantic_docs/vectors + faiss (aligned).
        """
        vec = self._embed_text(text)
        if vec is None:
            return
        if not _has_np:
            return

        with self._lock:
            if _ENABLE_EMBEDDINGS and _ENABLE_FAISS and _has_faiss:
                self._ensure_faiss_index(int(vec.shape[0]))  # type: ignore

            if _ENABLE_EMBEDDINGS and _ENABLE_FAISS and _has_faiss and self.faiss_index is not None:
                try:
                    self.faiss_index.add(np.expand_dims(vec, axis=0).astype(np.float32))  # type: ignore
                except Exception as e:
                    _err(self.logger, "Memory", f"Add to FAISS failed: {e}")
                    return

            self.semantic_vectors.append(vec)
            self.semantic_docs.append(doc)
