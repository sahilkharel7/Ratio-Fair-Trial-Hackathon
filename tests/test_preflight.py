"""The pre-demo check: every problem comes with its fix, only failures block the demo, nothing it
meets crashes the report, and the checks never leave this computer (the suite's network guard is
installed)."""

import importlib.metadata
import os
import socket
import sqlite3
from types import SimpleNamespace

import pytest

import ratio.__main__ as cli
from ratio import preflight
from ratio.config import default_config
from ratio.embeddings import EmbeddingModelMissing
from ratio.evaluation import FlagRecall
from ratio.extraction.loader import LoaderError
from ratio.llm import ModelNotAvailable, OllamaUnavailable
from ratio.pipeline import make_llm
from ratio.preflight import Check

CONFIG = default_config()
COMMITTED_CONFIG = preflight.STREAMLIT_CONFIGS[0].read_text(encoding="utf-8")


def write_configs(tmp_path, text):
    paths = (tmp_path / "project.toml", tmp_path / "script.toml")
    for path in paths:
        path.write_text(text, encoding="utf-8")
    return paths


def fake_client(version_error=None, model_error=None, status=None, version="0.35.0"):
    class Client:
        def __init__(self, settings, *, model=None):
            self.model = model or "qwen2.5:7b-instruct"

        def version(self):
            if version_error:
                raise version_error
            return version

        def check_model(self):
            if model_error:
                raise model_error
            return {}

        def status(self):
            if isinstance(status, Exception):
                raise status
            return status

    return Client


def raising(error):
    def run(*args, **kwargs):
        raise error

    return run


def test_python_3_11_is_ok_and_other_versions_warn():
    assert preflight.python_version((3, 11, 17)).status == "ok"
    newer = preflight.python_version((3, 12, 4))
    assert newer.status == "warn" and "3.11" in newer.detail and newer.fix


def test_packages_must_match_their_pins(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["streamlit==1.65.0", "pydantic==2.14.0"]\n', encoding="utf-8")
    versions = {"streamlit": "1.65.0", "pydantic": "2.14.0"}
    assert preflight.packages(pyproject, versions.__getitem__).status == "ok"
    drift = preflight.packages(pyproject, {**versions, "pydantic": "2.13.5"}.__getitem__)
    assert drift.status == "warn" and "pydantic 2.13.5 (pinned 2.14.0)" in drift.detail

    def without_streamlit(name):
        if name == "streamlit":
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    absent = preflight.packages(pyproject, without_streamlit)
    assert absent.status == "fail" and "streamlit" in absent.detail and "requirements-lock.txt" in absent.fix


@pytest.mark.parametrize("content", ["[project\n", "[project]\nname = 'x'\n", "project = 3\n", None])
def test_an_unreadable_pyproject_is_reported_not_raised(tmp_path, content):
    pyproject = tmp_path / "pyproject.toml"
    if content is not None:
        pyproject.write_text(content, encoding="utf-8")
    check = preflight.packages(pyproject)
    assert check.status == "fail" and "cannot be read" in check.detail and "pyproject.toml" in check.fix


def test_this_environment_has_the_pinned_packages():
    check = preflight.packages()
    assert check.status == "ok", check.detail


def test_the_embedding_model_must_be_downloaded(tmp_path):
    missing = preflight.embedding_model(tmp_path)
    assert missing.status == "fail" and "fetch_models.py" in missing.fix
    (tmp_path / "modules.json").write_text("[]", encoding="utf-8")
    assert preflight.embedding_model(tmp_path).status == "ok"


def test_the_committed_privacy_settings_pass():
    check = preflight.privacy_settings(environ={})
    assert check.status == "ok", check.detail


@pytest.mark.parametrize(
    ("old", "new", "named"),
    [
        ("gatherUsageStats = false", "gatherUsageStats = true", "browser.gatherUsageStats"),
        ('address = "127.0.0.1"', 'address = "0.0.0.0"', "server.address"),
        ("showErrorLinks = false", "showErrorLinks = true", "client.showErrorLinks"),
        ('codeFont = "monospace"', 'codeFont = "Mono:https://fonts.example.com/mono.css"', "theme.codeFont"),
        ("fontFaces = []", 'fontFaces = [{family = "Inter", url = "//fonts.example.com/inter.woff2"}]', "theme.fontFaces"),
        ("[theme.light]\nfont = \"sans-serif\"", "[theme.light]\nfont = \"url(https://fonts.example.com/a.woff2)\"", "theme.light.font"),
    ],
)
def test_a_setting_that_could_reach_another_computer_fails(tmp_path, old, new, named):
    assert old in COMMITTED_CONFIG
    check = preflight.privacy_settings(write_configs(tmp_path, COMMITTED_CONFIG.replace(old, new, 1)), environ={})
    assert check.status == "fail" and named in check.detail and "git checkout" in check.fix


def test_a_section_of_the_wrong_type_is_reported_not_raised(tmp_path):
    check = preflight.privacy_settings(write_configs(tmp_path, "browser = 1\n" + COMMITTED_CONFIG.split("[browser]", 1)[1].split("\n", 2)[2]), environ={})
    assert check.status == "fail" and "browser.gatherUsageStats" in check.detail


def test_the_two_config_files_must_exist_and_agree(tmp_path):
    project, script = write_configs(tmp_path, COMMITTED_CONFIG)
    script.write_text(COMMITTED_CONFIG + "\n# edited\n", encoding="utf-8")
    assert "differ" in preflight.privacy_settings((project, script), environ={}).detail
    script.unlink()
    missing = preflight.privacy_settings((project, script), environ={})
    assert missing.status == "fail" and missing.detail.startswith("Missing")
    project.write_text("[browser\n", encoding="utf-8")
    assert "cannot be read" in preflight.privacy_settings((project,), environ={}).detail


@pytest.mark.parametrize(
    ("environ", "fix"),
    [
        ({"STREAMLIT_SERVER_ADDRESS": "0.0.0.0"}, "unset STREAMLIT_SERVER_ADDRESS"),
        ({"STREAMLIT_BROWSER_GATHER_USAGE_STATS": "true"}, "unset STREAMLIT_BROWSER_GATHER_USAGE_STATS"),
        ({"STREAMLIT_CLIENT_SHOW_ERROR_LINKS": "1", "STREAMLIT_SERVER_HEADLESS": "false"}, "unset STREAMLIT_SERVER_HEADLESS STREAMLIT_CLIENT_SHOW_ERROR_LINKS"),
        ({"STREAMLIT_THEME_FONT": "Inter:https://fonts.example.com/inter.css"}, "unset STREAMLIT_THEME_FONT"),
        ({"STREAMLIT_SERVER_ADDRESS": "127.0.0.1", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "False", "STREAMLIT_THEME_BASE": "light"}, None),
    ],
)
def test_environment_variables_that_override_the_config_files_are_checked(environ, fix):
    check = preflight.privacy_settings(environ=environ)
    if fix is None:
        assert check.status == "ok", check.detail
    else:
        assert check.status == "fail" and "Overridden by environment variables" in check.detail and check.fix == fix


def test_a_case_store_that_will_be_created_passes(tmp_path):
    assert preflight.case_store(tmp_path / "new" / "folders" / "ratio.db").status == "ok"


def test_an_existing_case_store_is_probed_without_changing_it(tmp_path):
    path = tmp_path / "ratio.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    before = path.read_bytes()
    assert preflight.case_store(path).status == "ok"
    assert path.read_bytes() == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_a_store_in_a_read_only_folder_fails(tmp_path):
    folder = tmp_path / "locked"
    folder.mkdir()
    existing = folder / "ratio.db"
    with sqlite3.connect(existing) as conn:
        conn.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    folder.chmod(0o500)
    try:
        written = preflight.case_store(existing)
        created = preflight.case_store(folder / "sub" / "ratio.db")
    finally:
        folder.chmod(0o700)
    assert written.status == "fail" and "RATIO_DB" in written.fix
    assert created.status == "fail" and "cannot be created" in created.detail


def test_a_store_that_is_a_folder_or_not_a_database_fails(tmp_path):
    assert preflight.case_store(tmp_path).status == "fail"
    junk = tmp_path / "junk.db"
    junk.write_text("not a database, just text that is long enough to have a header" * 20, encoding="utf-8")
    check = preflight.case_store(junk)
    assert check.status == "fail" and "cannot be used" in check.detail


def test_a_busy_store_is_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "STORE_TIMEOUT_SECONDS", 0.1)
    path = tmp_path / "ratio.db"
    holder = sqlite3.connect(path)
    try:
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("CREATE TABLE held (x)")
        check = preflight.case_store(path)
    finally:
        holder.rollback()
        holder.close()
    assert check.status == "warn" and "busy" in check.detail


@pytest.mark.embed
def test_the_whole_demo_replays_with_no_model_and_no_network():
    check = preflight.demo_replay(CONFIG)
    assert check.status == "ok", check.detail
    assert "13 findings, each with its source text, and a judge profile of 9 cases with 1 pattern that warrants review" in check.detail


def test_a_missing_recorded_answer_fails_with_the_command_that_records_it(tmp_path):
    def empty_cache(config):
        return make_llm(config, mode="replay", write_dir=tmp_path, read_dirs=())

    check = preflight.demo_replay(CONFIG, llm_factory=empty_cache)
    assert check.status == "fail" and "missing on this computer" in check.detail
    assert "RATIO_MODEL=qwen2.5:7b-instruct python -m ratio build-demo-cache" in check.fix


@pytest.mark.parametrize(
    ("error", "detail", "fix"),
    [
        (EmbeddingModelMissing("no model"), "embedding model", "fetch_models.py"),
        (LoaderError("case.yaml is damaged"), "case.yaml is damaged", "git checkout -- data/demo"),
        (OSError("model.safetensors is missing"), "OSError: model.safetensors is missing", "delete models/all-MiniLM-L6-v2"),
        (PermissionError("hearing_1.txt"), "PermissionError", "git checkout -- data/demo"),
    ],
)
def test_a_demo_that_cannot_run_fails_with_its_fix(monkeypatch, error, detail, fix):
    monkeypatch.setattr(preflight, "process", raising(error))
    check = preflight.demo_replay(CONFIG)
    assert check.status == "fail" and detail in check.detail and fix in check.fix


def test_a_broken_decisions_file_is_named_with_both_candidates(monkeypatch):
    monkeypatch.setattr(preflight, "load_all_alias_decisions", raising(LoaderError("data/alias_decisions.yaml is not valid YAML")))
    check = preflight.demo_replay(CONFIG)
    assert check.status == "fail" and "data/alias_decisions.yaml is not valid YAML" in check.detail
    assert "data/demo/history/alias_decisions.yaml" in check.fix and "git checkout" not in check.fix


@pytest.mark.embed
def test_a_judge_history_too_small_for_the_script_fails(monkeypatch):
    monkeypatch.setattr(preflight, "load_history", lambda: ())
    check = preflight.demo_replay(CONFIG)
    assert check.status == "fail" and "the judge profile has 1 case, fewer than 5" in check.detail


@pytest.mark.embed
def test_a_missing_expected_output_or_dropped_finding_fails(monkeypatch):
    monkeypatch.setattr(preflight, "flag_recall", lambda *args: FlagRecall(found=(), missed=("f_follow_up",), violations=()))
    real_process = preflight.process

    def dropping(*args, **kwargs):
        record, analysis, report = real_process(*args, **kwargs)
        return record, analysis.model_copy(update={"dropped_flags": 1}), report

    monkeypatch.setattr(preflight, "process", dropping)
    check = preflight.demo_replay(CONFIG)
    assert check.status == "fail" and "1 finding dropped because" in check.detail and "expected outputs missing: f_follow_up" in check.detail


def test_the_judge_page_must_show_the_scripted_pattern():
    record = SimpleNamespace(case_id="venn-2025")
    profile = SimpleNamespace(case_ids=("venn-2025", *(f"h{i}" for i in range(8))), flags=())
    summary, problems = preflight._judge_shown(record, SimpleNamespace(profiles=(profile,), dropped_flags=1), minimum=5)
    assert summary == "a judge profile of 9 cases with 0 patterns that warrant review"
    assert problems == ["1 judge pattern dropped because the source text was not found", "the judge profile shows no pattern that warrants review"]
    assert preflight._judge_shown(record, SimpleNamespace(profiles=(), dropped_flags=0), minimum=5)[1] == ["the demo case has no judge profile"]


def test_the_fix_for_a_missing_answer_names_the_recorded_model_even_without_a_manifest(monkeypatch):
    monkeypatch.setattr(preflight, "read_demo_manifest", raising(ValueError("damaged")))
    assert preflight._demo_model(CONFIG) == CONFIG.settings.llm.default_model


def test_the_local_model_is_optional_for_the_demo():
    down = preflight.ollama_server(CONFIG, fake_client(version_error=OllamaUnavailable("connection refused")))
    assert down.status == "warn" and "the rest of the demo works" in down.detail and "brew services start ollama" in down.fix
    not_pulled = preflight.ollama_server(CONFIG, fake_client(model_error=ModelNotAvailable("not pulled")))
    assert not_pulled.status == "warn" and not_pulled.fix == "ollama pull qwen2.5:7b-instruct"
    ready = preflight.ollama_server(CONFIG, fake_client())
    assert ready.status == "ok" and "Ollama 0.35.0" in ready.detail and "qwen2.5:7b-instruct" in ready.detail


def test_a_non_loopback_ollama_host_is_refused_not_contacted(monkeypatch):
    monkeypatch.setenv("RATIO_OLLAMA_HOST", "http://ollama.example.com:11434")
    check = preflight.ollama_server(CONFIG)
    assert check.status == "warn" and "Refused before connecting" in check.detail and "ollama.example.com" in check.detail
    assert "unset RATIO_OLLAMA_HOST" in check.fix


@pytest.mark.parametrize("error", [ValueError("Expecting value"), AttributeError("'list' object has no attribute 'get'"), OSError("reset")])
def test_a_local_server_that_is_not_ollama_is_a_warning(error):
    check = preflight.ollama_server(CONFIG, fake_client(version_error=error))
    assert check.status == "warn" and "not answering" in check.detail


def test_an_unreadable_demo_manifest_is_a_warning_for_the_model_check(monkeypatch):
    monkeypatch.setattr(preflight, "read_demo_manifest", raising(ValueError("manifest.json: Expecting value")))
    check = preflight.ollama_server(CONFIG, fake_client())
    assert check.status == "warn" and "data/demo/cache/manifest.json cannot be read" in check.detail


@pytest.mark.parametrize(
    ("status", "expected"),
    [({"cloud": {"disabled": True, "source": "config"}}, "ok"), ({"cloud": {"disabled": False}}, "warn")],
)
def test_the_running_ollama_server_reports_its_cloud_setting(tmp_path, status, expected):
    check = preflight.ollama_cloud(CONFIG, tmp_path / "server.json", fake_client(status=status))
    assert check.status == expected and "running Ollama server" in check.detail


@pytest.mark.parametrize(
    ("content", "status"),
    [('{"disable_ollama_cloud": true}', "ok"), ('{"disable_ollama_cloud": false}', "warn"), ("[true]", "warn"), ("{", "warn"), (None, "warn")],
)
def test_with_ollama_stopped_the_settings_file_decides(tmp_path, content, status):
    path = tmp_path / "server.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    for server in (OllamaUnavailable("connection refused"), {"version": "0.1"}, ["not", "a", "dict"]):
        check = preflight.ollama_cloud(CONFIG, path, fake_client(status=server))
        assert check.status == status
        assert status == "ok" or "disable_ollama_cloud" in check.fix


def test_a_busy_app_port_is_reported_and_a_free_one_passes():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        busy = preflight.app_port(port=port)
    assert busy.status == "warn" and f"http://127.0.0.1:{port}" in busy.detail
    assert preflight.app_port(port=port).status == "ok"


def test_run_all_checks_everything_in_order(monkeypatch):
    names = ["python_version", "packages", "embedding_model", "privacy_settings", "case_store", "demo_replay", "ollama_server", "ollama_cloud", "app_port"]
    for name in names:
        monkeypatch.setattr(preflight, name, lambda *args, _name=name, **kwargs: Check(_name, "ok", "fine"))
    assert [check.name for check in preflight.run_all(CONFIG)] == names


def test_a_check_that_meets_something_unexpected_reports_it(monkeypatch):
    for name in ("python_version", "packages", "embedding_model", "privacy_settings", "case_store", "demo_replay", "app_port"):
        monkeypatch.setattr(preflight, name, lambda *args, **kwargs: Check("x", "ok", "fine"))
    monkeypatch.setattr(preflight, "packages", raising(RuntimeError("something odd")))
    monkeypatch.setattr(preflight, "ollama_server", raising(RuntimeError("odd reply")))
    monkeypatch.setattr(preflight, "ollama_cloud", raising(KeyError("cloud")))
    checks = {check.name: check for check in preflight.run_all(CONFIG)}
    assert checks["Packages"].status == "fail" and "RuntimeError: something odd" in checks["Packages"].detail
    assert checks["Local model (live-note step only)"].status == "warn" and checks["Ollama cloud features"].status == "warn"


def test_the_report_shows_fixes_and_only_failures_block_the_demo():
    checks = (Check("A", "ok", "fine", "never shown"), Check("B", "warn", "slow", "do x"), Check("C", "fail", "broken", "do y"))
    report = preflight.format_report(checks)
    assert "  fail  C: broken\n        fix: do y" in report and "fix: do x" in report and "never shown" not in report
    assert report.endswith("Not ready for the demo: 1 failure, 1 warning.")
    assert preflight.exit_code(checks) == 1
    assert preflight.exit_code(checks[:2]) == 0
    assert preflight.format_report(checks[:1]).endswith("Ready for the demo: 0 failures, 0 warnings.")


@pytest.mark.parametrize(("checks", "code"), [((Check("A", "ok", "fine"),), 0), ((Check("A", "warn", "w"), Check("B", "fail", "f", "do y")), 1)])
def test_the_command_prints_the_report_and_exits_with_its_status(monkeypatch, capsys, checks, code):
    monkeypatch.setattr(preflight, "run_all", lambda config: checks)
    assert cli.main(["preflight"]) == code
    out = capsys.readouterr().out
    assert out.startswith("Ratio preflight") and ("Not ready" if code else "Ready for the demo") in out
