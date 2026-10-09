"""Hard rule 1 at launch: the privacy settings apply from any working directory, and the startup
check refuses settings that would let the browser or the server reach another computer."""

import json
import os
import subprocess
import sys
import tomllib

import pytest
from streamlit import config as st_config

from ratio.paths import REPO_ROOT

PROJECT_CONFIG = REPO_ROOT / ".streamlit" / "config.toml"
SCRIPT_CONFIG = REPO_ROOT / "app" / ".streamlit" / "config.toml"
PROBE = """
import json, sys, streamlit.config as c
# The probe deliberately runs outside the checkout; test this source tree without depending on
# an editable-install finder (some bundled Python runtimes do not execute .pth import hooks).
sys.path.insert(0, {repo!r})
from ratio import netguard
netguard.install()  # a theme file fetched from a URL would fail loudly here
c._main_script_path = {script!r}
c.get_config_options(force_reparse=True)
keys = ["browser.gatherUsageStats", "server.address", "server.headless", "client.showErrorLinks", "theme.fontFaces", "theme.font", "theme.base"]
fonts = [k for k in c.get_config_options() if k.startswith("theme.") and k.endswith(("font", "Font")) and isinstance(c.get_option(k), str)]
remote = [k for k in fonts if "//" in c.get_option(k)]
print(json.dumps({{**{{key: c.get_option(key) for key in keys}}, "remote_fonts": remote, "fonts_checked": len(fonts)}}))
"""


def test_the_project_and_script_level_configs_are_identical():
    assert PROJECT_CONFIG.read_bytes() == SCRIPT_CONFIG.read_bytes()
    settings = tomllib.loads(SCRIPT_CONFIG.read_text(encoding="utf-8"))
    assert settings["browser"]["gatherUsageStats"] is False
    assert settings["server"]["address"] == "127.0.0.1" and settings["client"]["showErrorLinks"] is False
    assert settings["theme"]["fontFaces"] == []


def test_settings_apply_when_launched_from_another_folder_with_a_hostile_global_config(tmp_path):
    home = tmp_path / "home"
    (home / ".streamlit").mkdir(parents=True)
    (home / ".streamlit" / "config.toml").write_text(
        '[browser]\ngatherUsageStats = true\n[client]\nshowErrorLinks = true\n'
        '[theme]\nbase = "https://attacker.example/theme.toml"\nfont = "Inter"\n'
        '[[theme.fontFaces]]\nfamily = "Inter"\nurl = "https://fonts.example.com/i.woff2"\n'
        '[theme.light]\nfont = "Inter:https://fonts.example.com/css2?family=Inter"\n'
        '[theme.dark.sidebar]\ncodeFont = "Mono:https://fonts.example.com/mono.css"\n',
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    probe = PROBE.format(script=str(REPO_ROOT / "app" / "main.py"), repo=str(REPO_ROOT))
    env = {**os.environ, "HOME": str(home)}
    out = subprocess.run([sys.executable, "-c", probe], cwd=elsewhere, env=env, capture_output=True, text=True, check=True)
    effective = json.loads(out.stdout.strip().splitlines()[-1])
    assert effective == {
        "browser.gatherUsageStats": False, "server.address": "127.0.0.1", "server.headless": True,
        "client.showErrorLinks": False, "theme.fontFaces": [], "theme.font": "sans-serif", "theme.base": "light",
        "remote_fonts": [], "fonts_checked": 18,
    }  # fmt: skip


@pytest.mark.parametrize(
    "value",
    [
        [{"family": "Inter", "url": "https://fonts.example.com/inter.woff2"}],
        [{"family": "Inter", "url": "//fonts.example.com/inter.woff2"}],
    ],
)
def test_the_startup_check_refuses_a_remote_font(value):
    from ratio_ui import privacy

    original = st_config.get_option("theme.fontFaces")
    try:
        st_config.set_option("theme.fontFaces", value)
        assert any("theme.fontFaces" in problem and "remove it there" in problem for problem in privacy.problems())
    finally:
        st_config.set_option("theme.fontFaces", original)
    assert not any("theme." in problem for problem in privacy.problems())
