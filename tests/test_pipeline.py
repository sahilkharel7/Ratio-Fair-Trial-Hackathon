"""Pipeline wiring: replay from the committed demo cache, live-mode wiring, and the CLI."""

import time

import pytest

from ratio.__main__ import main
from ratio.config import load_config
from ratio.extraction.build import load_case
from ratio.llm import CacheMiss
from ratio.paths import DEMO_CASE_DIR
from ratio.pipeline import demo_llm, ingest, make_llm, read_demo_manifest

CONFIG = load_config()


def test_replay_mode_never_creates_a_client_and_raises_on_a_miss(tmp_path):
    llm = make_llm(CONFIG, mode="replay", write_dir=tmp_path, read_dirs=())
    assert llm.replay_only
    with pytest.raises(CacheMiss):
        ingest(load_case(DEMO_CASE_DIR), llm, CONFIG)


def test_live_mode_uses_the_configured_local_model(tmp_path, monkeypatch):
    monkeypatch.delenv("RATIO_MODEL", raising=False)
    llm = make_llm(CONFIG, mode="live", write_dir=tmp_path)
    assert not llm.replay_only
    assert llm.model == "qwen2.5:7b-instruct"


def test_demo_cache_manifest_matches_the_current_demo_documents():
    manifest = read_demo_manifest()
    assert manifest is not None, "run `python -m ratio build-demo-cache` once with Ollama running"
    assert manifest.synthetic
    current = {doc.id: doc.sha256 for doc in load_case(DEMO_CASE_DIR).documents}
    assert manifest.documents == current, "demo documents changed: rebuild the demo cache"


def test_demo_case_replays_completely_from_the_committed_cache():
    started = time.monotonic()
    llm = demo_llm(CONFIG)
    record, report = ingest(load_case(DEMO_CASE_DIR), llm, CONFIG)
    assert llm.replay_only
    assert llm.stats.misses == 0
    assert report.events_kept > 0
    assert any(event.type == "arrest" for event in record.events)
    assert time.monotonic() - started < 20


def test_cli_ingest_replays_the_demo_case_into_the_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RATIO_DB", str(tmp_path / "ratio.db"))
    assert main(["ingest"]) == 0
    output = capsys.readouterr().out
    assert "venn-2025" in output and "replay" in output


def test_cli_reports_errors_without_a_traceback(tmp_path, capsys):
    assert main(["ingest", str(tmp_path / "missing")]) == 1
    assert "error:" in capsys.readouterr().err
