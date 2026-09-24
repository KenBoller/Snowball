import os
import sys
from pathlib import Path
import pytest


# Ensure S:\Snowball is importable as project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Make tests deterministic + safe:
os.environ.setdefault("SNOWBALL_DISABLE_LOCAL_MEMORY", "1")
os.environ.setdefault("SNOWBALL_USE_MEMORY", "0")
os.environ.setdefault("SNOWBALL_USE_DM", "0")
os.environ.setdefault("SNOWBALL_USE_SENTIMENT", "0")
os.environ.setdefault("SNOWBALL_FANOUT", "0")
os.environ.setdefault("SNOWBALL_WARMUP", "0")


@pytest.fixture(autouse=True)
def isolate_snowball_storage(monkeypatch, tmp_path):
    storage_dir = tmp_path / "snowball-storage"

    monkeypatch.setenv(
        "SNOWBALL_STORAGE_DIR",
        str(storage_dir),
    )
    monkeypatch.setenv(
        "SNOWBALL_LOCAL_MEMORY_DIR",
        str(storage_dir / "memory"),
    )
    monkeypatch.setenv(
        "SNOWBALL_LOG_DIR",
        str(storage_dir / "logs"),
    )

    try:
        import core.api.chat_api as chat_api
        chat_api.reset()
    except ImportError:
        pass

    yield

    try:
        import core.api.chat_api as chat_api
        chat_api.reset()
    except ImportError:
        pass