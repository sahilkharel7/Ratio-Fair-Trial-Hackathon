"""One-time, online setup step: download the MiniLM embedding model into models/.

Ratio itself never downloads anything at runtime. Run this once while online:

    python scripts/fetch_models.py

This script deliberately does not import the `ratio` package, because importing `ratio`
forces Hugging Face offline mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "models" / "all-MiniLM-L6-v2"


def main() -> int:
    if (TARGET_DIR / "modules.json").exists():
        print(f"Model already present at {TARGET_DIR}")
        return 0

    from sentence_transformers import SentenceTransformer

    print(f"Downloading {MODEL_ID}@{MODEL_REVISION[:8]} (about 90 MB) ...")
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(TARGET_DIR))
    print(f"Saved to {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
