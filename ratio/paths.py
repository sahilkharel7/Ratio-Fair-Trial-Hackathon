"""Filesystem locations used by Ratio. Everything lives inside the repository."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CONFIG_DIR = PACKAGE_DIR / "config"

MODELS_DIR = REPO_ROOT / "models"
MINILM_DIR = MODELS_DIR / "all-MiniLM-L6-v2"

DATA_DIR = REPO_ROOT / "data"
DEMO_DIR = DATA_DIR / "demo"
DEMO_CASE_DIR = DEMO_DIR / "case"
GOLD_DIR = DEMO_DIR / "gold"
HISTORY_DIR = DEMO_DIR / "history"
ALIAS_DECISIONS = HISTORY_DIR / "alias_decisions.yaml"  # decisions on the synthetic demo history
PUBLIC_ALIAS_DECISIONS = DATA_DIR / "alias_decisions.yaml"  # decisions on public cases, never in a SYNTHETIC file
DEMO_CACHE_DIR = DEMO_DIR / "cache"
RUNTIME_CACHE_DIR = DATA_DIR / "cache" / "llm"


def db_path() -> Path:
    """SQLite store location; RATIO_DB overrides the default (tests use a temp file)."""
    override = os.environ.get("RATIO_DB")
    return Path(override) if override else DATA_DIR / "ratio.db"
