"""Ratio: offline fair-trial monitoring analysis (FairTrial AI Hackathon, Track 2)."""

import os

# Hugging Face libraries read these once, when they are first imported, so they are set here,
# before any module of this package can import them. Ratio never downloads anything at runtime.
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
os.environ.update(_OFFLINE_ENV)

__version__ = "0.1.0"
