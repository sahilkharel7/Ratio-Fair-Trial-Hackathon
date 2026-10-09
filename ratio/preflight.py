"""Pre-demo checks: is this computer ready to run the demo with the network off?

    python -m ratio preflight

Every check stays on this computer: files in the repository, the local Ollama server on
127.0.0.1 and the app's port. Each problem names the command that fixes it. A failure means the
demo will not run as scripted (exit 1); a warning only affects the optional live-model step or
the setup. A check that meets something unexpected reports it instead of stopping the report.
"""

from __future__ import annotations

import http.client
import importlib.metadata
import json
import os
import socket
import sqlite3
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ratio.config import ConfigError, RatioConfig
from ratio.context import Embedder
from ratio.embeddings import EmbeddingModelMissing
from ratio.evaluation import flag_recall
from ratio.expected import ExpectedFlags
from ratio.extraction.build import load_case
from ratio.extraction.loader import LoaderError
from ratio.history import load_all_alias_decisions, load_history
from ratio.llm import CachedLLM, CacheMiss, LLMError, OllamaClient
from ratio.netguard import is_loopback_host, names_remote_resource
from ratio.paths import ALIAS_DECISIONS, DEMO_CASE_DIR, GOLD_DIR, MINILM_DIR, PUBLIC_ALIAS_DECISIONS, REPO_ROOT, db_path
from ratio.pipeline import DEMO_CACHE_MANIFEST, analyze_judges, demo_llm, process, read_demo_manifest
from ratio.results import CaseAnalysis, JudgeReport
from ratio.schema import CaseRecord

Status = Literal["ok", "warn", "fail"]

PYTHON, PACKAGES, EMBEDDINGS, PRIVACY, STORE = "Python", "Packages", "Embedding model", "Privacy settings", "Case store"
DEMO, MODEL, CLOUD, PORT = "Demo without the model", "Local model (live-note step only)", "Ollama cloud features", "App port"

PYPROJECT = REPO_ROOT / "pyproject.toml"
STREAMLIT_CONFIGS = (REPO_ROOT / ".streamlit" / "config.toml", REPO_ROOT / "app" / ".streamlit" / "config.toml")
OLLAMA_SERVER_JSON = Path.home() / ".ollama" / "server.json"
APP_HOST = "127.0.0.1"
APP_PORT = 8501
STORE_TIMEOUT_SECONDS = 2.0
INSTALL = 'uv pip install -r requirements-lock.txt -e ".[dev]"'
FETCH_MODEL = "python scripts/fetch_models.py (once, while online)"
RESTORE_CONFIGS = "git checkout -- .streamlit/config.toml app/.streamlit/config.toml"
RESTORE_DEMO = "git checkout -- data/demo"
START_OLLAMA = "brew services start ollama (or open the Ollama app)"
PRIVACY_VALUES: Mapping[tuple[str, str], object] = {
    ("browser", "gatherUsageStats"): False,
    ("server", "address"): APP_HOST,
    ("server", "headless"): True,
    ("client", "showErrorLinks"): False,
    ("theme", "fontFaces"): [],
}
# `streamlit run` applies these variables over every config file.
PRIVACY_VARIABLES: Mapping[str, tuple[str, str]] = {
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": ("browser", "gatherUsageStats"),
    "STREAMLIT_SERVER_ADDRESS": ("server", "address"),
    "STREAMLIT_SERVER_HEADLESS": ("server", "headless"),
    "STREAMLIT_CLIENT_SHOW_ERROR_LINKS": ("client", "showErrorLinks"),
}
_TRUE = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE = frozenset({"0", "false", "f", "no", "n", "off"})
# What a damaged file or folder, or an unexpected local server, can raise.
DAMAGE: tuple[type[Exception], ...] = (OSError, ValueError, TypeError, AttributeError, KeyError, http.client.HTTPException)


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    fix: str = ""


def _shown(path: Path) -> str:
    """A path as a person would type it: relative to the repository, or under ~."""
    for base, prefix in ((REPO_ROOT, ""), (Path.home(), "~/")):
        try:
            return prefix + str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def python_version(info: Sequence[int] = sys.version_info) -> Check:
    version = ".".join(str(part) for part in info[:3])
    if tuple(info[:2]) == (3, 11):
        return Check(PYTHON, "ok", f"Python {version}")
    return Check(PYTHON, "warn", f"Python {version}; Ratio is tested on Python 3.11", "uv venv -p 3.11, then install again")


def _pins(pyproject: Path) -> dict[str, str]:
    requirements = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    return {name.strip(): version.strip() for name, _, version in (r.partition("==") for r in requirements)}


def packages(pyproject: Path = PYPROJECT, installed: Callable[[str], str] = importlib.metadata.version) -> Check:
    try:
        pins = _pins(pyproject)
    except DAMAGE as exc:
        return Check(PACKAGES, "fail", f"{_shown(pyproject)} cannot be read: {type(exc).__name__}: {exc}", "git checkout -- pyproject.toml")
    missing, different = [], []
    for name, pinned in pins.items():
        try:
            found = installed(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if found != pinned:
            different.append(f"{name} {found} (pinned {pinned})")
    if missing:
        return Check(PACKAGES, "fail", "Not installed: " + ", ".join(missing), INSTALL)
    if different:
        return Check(PACKAGES, "warn", "Not the pinned versions: " + "; ".join(different), INSTALL)
    return Check(PACKAGES, "ok", f"All {len(pins)} direct dependencies at their pinned versions")


def embedding_model(model_dir: Path = MINILM_DIR) -> Check:
    if (model_dir / "modules.json").is_file():
        return Check(EMBEDDINGS, "ok", f"Found in {_shown(model_dir)}")
    return Check(EMBEDDINGS, "fail", f"Not found in {_shown(model_dir)}", FETCH_MODEL)


def _setting(settings: Mapping[str, object], section: str, key: str) -> object:
    table = settings.get(section)
    return table.get(key) if isinstance(table, Mapping) else None


def _remote_keys(value: object, prefix: str) -> list[str]:
    if isinstance(value, Mapping):
        return [key for name, item in value.items() for key in _remote_keys(item, f"{prefix}.{name}")]
    return [prefix] if names_remote_resource(value) else []


def _overridden(environ: Mapping[str, str]) -> list[str]:
    """STREAMLIT_* variables that would apply an unsafe value over the config files."""
    unsafe = []
    for variable, setting in PRIVACY_VARIABLES.items():
        if variable not in environ:
            continue
        value = environ[variable].strip()
        required = PRIVACY_VALUES[setting]
        safe = is_loopback_host(value) if setting == ("server", "address") else value.lower() in (_TRUE if required else _FALSE)
        if not safe:
            unsafe.append(variable)
    return unsafe + sorted(name for name, value in environ.items() if name.startswith("STREAMLIT_THEME_") and names_remote_resource(value))


def privacy_settings(paths: Sequence[Path] = STREAMLIT_CONFIGS, environ: Mapping[str, str] = os.environ) -> Check:
    """Hard rule 1 for the browser: usage statistics off, this computer only, nothing fetched from the web."""
    missing = [_shown(path) for path in paths if not path.is_file()]
    if missing:
        return Check(PRIVACY, "fail", "Missing: " + ", ".join(missing), RESTORE_CONFIGS)
    try:
        contents = [path.read_bytes() for path in paths]
        settings = tomllib.loads(contents[0].decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return Check(PRIVACY, "fail", f"{_shown(paths[0])} cannot be read: {exc}", RESTORE_CONFIGS)
    if any(content != contents[0] for content in contents[1:]):
        return Check(PRIVACY, "fail", "The Streamlit config files differ: " + ", ".join(_shown(path) for path in paths), RESTORE_CONFIGS)
    wrong = [f"{section}.{key}" for (section, key), value in PRIVACY_VALUES.items() if _setting(settings, section, key) != value]
    remote = _remote_keys(settings.get("theme", {}), "theme")
    variables = _overridden(environ)
    problems, fixes = [], []
    if wrong or remote:
        problems += ["Not set as required: " + ", ".join(wrong)] if wrong else []
        problems += ["Web addresses in: " + ", ".join(remote)] if remote else []
        fixes.append(RESTORE_CONFIGS)
    if variables:
        problems.append("Overridden by environment variables: " + ", ".join(variables))
        fixes.append("unset " + " ".join(variables))
    if problems:
        return Check(PRIVACY, "fail", "; ".join(problems), "; ".join(fixes))
    return Check(PRIVACY, "ok", f"Usage statistics off, served on {APP_HOST} only, no error links, no web fonts")


def _nearest_existing(folder: Path) -> Path:
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    return folder


def case_store(path: Path | None = None) -> Check:
    """The store the app will use (RATIO_DB, or data/ratio.db), probed the way the app writes to it."""
    target = path or db_path()
    shown = _shown(target)
    if target.is_dir():
        return Check(STORE, "fail", f"{shown} is a folder, not a database file", "Set RATIO_DB to a file path")
    if not target.exists():
        folder = _nearest_existing(target.parent)  # the app creates the missing folders
        if folder.is_dir() and os.access(folder, os.W_OK | os.X_OK):
            return Check(STORE, "ok", f"{shown} will be created")
        return Check(STORE, "fail", f"{shown} cannot be created in {_shown(folder)}", "Fix that folder's permissions, or set RATIO_DB to a writable file path")
    try:
        with closing(sqlite3.connect(target, timeout=STORE_TIMEOUT_SECONDS)) as conn:
            conn.execute("BEGIN IMMEDIATE")  # an explicit transaction, so the test write below is rolled back
            conn.execute("CREATE TABLE ratio_preflight_probe (x)")
            conn.rollback()
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc):
            return Check(STORE, "warn", f"{shown} is busy: another program is writing to it", "Close the other program, or set RATIO_DB to another file")
        return Check(STORE, "fail", f"{shown} cannot be written: {exc}", "Fix the folder's permissions, or set RATIO_DB to a writable file path")
    except (sqlite3.Error, OSError) as exc:
        return Check(STORE, "fail", f"{shown} cannot be used: {exc}", "Move the file away, or set RATIO_DB to another file path")
    return Check(STORE, "ok", f"{shown} can be written")


def _judge_shown(record: CaseRecord, report: JudgeReport, minimum: int) -> tuple[str, list[str]]:
    """What the judge page shows for the demo judge, and what stops it showing the scripted pattern."""
    profile = next((p for p in report.profiles if record.case_id in p.case_ids), None)
    problems = [f"{_count(report.dropped_flags, 'judge pattern')} dropped because the source text was not found"] if report.dropped_flags else []
    if profile is None:
        return "no judge profile", [*problems, "the demo case has no judge profile"]
    cases, patterns = len(profile.case_ids), len(profile.flags)
    found = "1 pattern that warrants review" if patterns == 1 else f"{patterns} patterns that warrant review"
    summary = f"a judge profile of {_count(cases, 'case')} with {found}"
    if cases < minimum:
        problems.append(f"the judge profile has {_count(cases, 'case')}, fewer than {minimum}, so every indicator is hidden")
    elif not patterns:
        problems.append("the judge profile shows no pattern that warrants review")
    return summary, problems


def _demo_problems(record: CaseRecord, analysis: CaseAnalysis) -> list[str]:
    """What the demo script narrates and the replay does not show (the expected outputs of the eval)."""
    expected = ExpectedFlags.model_validate(json.loads((GOLD_DIR / "expected_flags.json").read_text(encoding="utf-8")))
    recall = flag_recall(record, analysis, expected)
    problems = [f"{_count(analysis.dropped_flags, 'finding')} dropped because the source text was not found"] if analysis.dropped_flags else []
    problems += ["expected outputs missing: " + ", ".join(recall.missed)] if recall.missed else []
    problems += ["outputs present that must not be: " + ", ".join(recall.violations)] if recall.violations else []
    return problems


def _demo_model(config: RatioConfig) -> str:
    """The model the recorded demo answers came from."""
    try:
        manifest = read_demo_manifest()
    except DAMAGE:
        manifest = None
    return manifest.model if manifest else config.settings.llm.default_model


def demo_replay(
    config: RatioConfig, llm_factory: Callable[[RatioConfig], CachedLLM] = demo_llm, embedder: Embedder | None = None
) -> Check:
    """The whole demo from the recorded model answers, as the app's "Load the demo case" runs it,
    compared with what the demo script narrates."""
    started = time.monotonic()
    decision_files = f"{_shown(ALIAS_DECISIONS)} and {_shown(PUBLIC_ALIAS_DECISIONS)}"
    try:
        decisions = load_all_alias_decisions()
    except (LoaderError, *DAMAGE) as exc:
        return Check(DEMO, "fail", f"A file of name decisions cannot be read: {exc}", f"Fix the file named in the message (one of {decision_files})")
    try:
        record, analysis, _ = process(load_case(DEMO_CASE_DIR), llm_factory(config), config, embedder=embedder)
        report = analyze_judges([*load_history(), record], {record.case_id: analysis}, decisions, config)
        problems = _demo_problems(record, analysis)
    except CacheMiss as exc:
        return Check(DEMO, "fail", f"A recorded model answer is missing on this computer: {exc}",
                     f"Start Ollama, then RATIO_MODEL={_demo_model(config)} python -m ratio build-demo-cache (asks the model only the missing questions)")  # fmt: skip
    except EmbeddingModelMissing:
        return Check(DEMO, "fail", "Cannot run without the embedding model", FETCH_MODEL)
    except (LLMError, LoaderError, ConfigError, *DAMAGE) as exc:
        return Check(DEMO, "fail", f"The demo stopped: {type(exc).__name__}: {exc}",
                     f"Restore the demo files ({RESTORE_DEMO}); if the embedding model is damaged, delete models/all-MiniLM-L6-v2 and run {FETCH_MODEL}")  # fmt: skip
    seconds = time.monotonic() - started
    judge, judge_problems = _judge_shown(record, report, config.settings.judges.min_case_count)
    if problems or judge_problems:
        return Check(DEMO, "fail", "The demo would not show what the script says: " + "; ".join(problems + judge_problems), f"Restore the demo files: {RESTORE_DEMO}")
    return Check(DEMO, "ok", f"Replayed in {seconds:.1f} s: {_count(len(analysis.all_flags()), 'finding')}, each with its source text, and {judge}")


def ollama_server(config: RatioConfig, client_factory: Callable[..., OllamaClient] = OllamaClient) -> Check:
    """Only the optional "Run this note live" step needs the model; the rest of the demo replays."""
    host = os.environ.get("RATIO_OLLAMA_HOST") or config.settings.llm.host  # as the client resolves it
    try:
        manifest = read_demo_manifest()
    except DAMAGE as exc:
        return Check(MODEL, "warn", f"{_shown(DEMO_CACHE_MANIFEST)} cannot be read: {exc}", RESTORE_DEMO)
    try:
        client = client_factory(config.settings.llm, model=manifest.model if manifest else None)
    except (LLMError, ValueError) as exc:  # refused before any connection: not this computer, or a cloud model
        return Check(MODEL, "warn", f"Refused before connecting: {exc}",
                     "unset RATIO_OLLAMA_HOST and RATIO_MODEL (Ratio only talks to a local model on 127.0.0.1)")  # fmt: skip
    try:
        version = client.version()
    except (LLMError, *DAMAGE) as exc:
        return Check(MODEL, "warn", f"Ollama is not answering on {host} ({exc}); the rest of the demo works", START_OLLAMA)
    try:
        client.check_model()
    except (LLMError, *DAMAGE) as exc:
        return Check(MODEL, "warn", f"Ollama {version} is running, but {client.model} is not ready: {exc}", f"ollama pull {client.model}")
    return Check(MODEL, "ok", f"Ollama {version} on {host} with {client.model}")


def _reported_cloud(config: RatioConfig, client_factory: Callable[..., OllamaClient]) -> bool | None:
    """Whether the running Ollama server says its cloud features are disabled; None when it does not say."""
    try:
        status = client_factory(config.settings.llm).status()
    except (LLMError, *DAMAGE):
        return None
    cloud = status.get("cloud") if isinstance(status, dict) else None
    disabled = cloud.get("disabled") if isinstance(cloud, dict) else None
    return disabled if isinstance(disabled, bool) else None


def ollama_cloud(
    config: RatioConfig, server_json: Path = OLLAMA_SERVER_JSON, client_factory: Callable[..., OllamaClient] = OllamaClient
) -> Check:
    """Asks the running Ollama server first (it reads its settings when it starts), then the settings file."""
    fix = f"echo '{{\"disable_ollama_cloud\": true}}' > {_shown(server_json)}, then brew services restart ollama"
    reported = _reported_cloud(config, client_factory)
    if reported is True:
        return Check(CLOUD, "ok", "Disabled, as the running Ollama server reports")
    if reported is False:
        return Check(CLOUD, "warn", "The running Ollama server reports its cloud features on, so it may contact ollama.com", fix)
    try:
        settings = json.loads(server_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = None
    if isinstance(settings, dict) and settings.get("disable_ollama_cloud") is True:
        return Check(CLOUD, "ok", f"Disabled in {_shown(server_json)}")
    return Check(CLOUD, "warn", f"Not disabled in {_shown(server_json)}, so the Ollama server may contact ollama.com", fix)


def app_port(host: str = APP_HOST, port: int = APP_PORT) -> Check:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        in_use = probe.connect_ex((host, port)) == 0
    if in_use:
        return Check(PORT, "warn", f"Something already listens on {host}:{port}; if it is Ratio, open http://{host}:{port}",
                     "Stop the other program, or use the app that is already running")  # fmt: skip
    return Check(PORT, "ok", f"{host}:{port} is free")


def _guarded(name: str, run: Callable[[], Check], status: Status = "fail") -> Check:
    """A check that meets something unexpected reports it instead of stopping the whole report."""
    try:
        return run()
    except Exception as exc:  # noqa: BLE001 - a diagnostic must report every problem, never crash on one
        return Check(name, status, f"The check stopped: {type(exc).__name__}: {exc}", "Fix the file or setting named here, then run the check again")


def run_all(config: RatioConfig) -> tuple[Check, ...]:
    return (
        _guarded(PYTHON, lambda: python_version()),
        _guarded(PACKAGES, lambda: packages()),
        _guarded(EMBEDDINGS, lambda: embedding_model()),
        _guarded(PRIVACY, lambda: privacy_settings()),
        _guarded(STORE, lambda: case_store()),
        _guarded(DEMO, lambda: demo_replay(config)),
        _guarded(MODEL, lambda: ollama_server(config), "warn"),
        _guarded(CLOUD, lambda: ollama_cloud(config), "warn"),
        _guarded(PORT, lambda: app_port(), "warn"),
    )


def format_report(checks: Sequence[Check]) -> str:
    lines = ["Ratio preflight (every check runs on this computer)"]
    for check in checks:
        lines.append(f"  {check.status:<5} {check.name}: {check.detail}")
        if check.fix and check.status != "ok":
            lines.append(f"        fix: {check.fix}")
    failures = sum(check.status == "fail" for check in checks)
    warnings = sum(check.status == "warn" for check in checks)
    counts = f"{_count(failures, 'failure')}, {_count(warnings, 'warning')}"
    lines.append(f"Not ready for the demo: {counts}." if failures else f"Ready for the demo: {counts}.")
    return "\n".join(lines)


def exit_code(checks: Sequence[Check]) -> int:
    return 1 if any(check.status == "fail" for check in checks) else 0
