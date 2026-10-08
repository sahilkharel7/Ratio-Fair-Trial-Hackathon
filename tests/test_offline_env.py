"""Hard rule 1 (offline, no telemetry): checked for the Hugging Face libraries and Streamlit config."""

import os
import subprocess
import sys
import tomllib

from ratio.paths import REPO_ROOT


def test_importing_ratio_forces_hugging_face_offline_mode():
    code = "import ratio, huggingface_hub.constants as c; print(c.HF_HUB_OFFLINE, c.HF_HUB_DISABLE_TELEMETRY)"
    env = {**os.environ, "HF_HUB_OFFLINE": "0", "HF_HUB_DISABLE_TELEMETRY": "0"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert result.stdout.split() == ["True", "True"]


def test_streamlit_config_is_private_and_local():
    config = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["address"] == "127.0.0.1"
    assert config["server"]["headless"] is True
    assert config["server"]["showEmailPrompt"] is False
    assert config["client"]["showErrorLinks"] is False
    assert config["client"]["toolbarMode"] == "viewer"
