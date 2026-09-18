import os
import sys
from pathlib import Path

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
