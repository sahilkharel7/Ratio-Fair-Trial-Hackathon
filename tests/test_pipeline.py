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


def test_a_reuse_pair_whose_flag_fails_provenance_is_dropped_with_it(monkeypatch):
    from ratio.context import AnalysisContext
    from ratio.modules import reuse
    from ratio.paths import GOLD_DIR
    from ratio.pipeline import analyze
    from ratio.results import ReusePair, ReuseResult
    from ratio.schema import CaseRecord, Evidence, Flag, SourceSpan
    from ratio.testing import FakeEmbedder, FakeLLM

    record = CaseRecord.model_validate_json((GOLD_DIR / "mock_record.json").read_text(encoding="utf-8"))
    judgment = record.documents_of_type("judgment")[0]
    forged = SourceSpan(doc_id=judgment.id, start=0, end=5, text="FORGE")  # not the document's text
    real = record.passages_of(judgment.id)[21].span
    pair = ReusePair(id="pair", kind="verbatim", judgment=forged, indictment=real, flag_id="flag")
    flag = Flag(
        id="flag", case_id=record.case_id, module="reuse", standard_id="reasoning_reuse", standard_label="x",
        status="verbatim_reuse", message="m", evidence=(Evidence(role="judgment", span=forged),),
    )  # fmt: skip
    monkeypatch.setattr(
        reuse, "run", lambda rec, ctx: ReuseResult(judgment_doc_id=judgment.id, indictment_doc_id=None, pairs=(pair,), flags=(flag,))
    )
    ctx = AnalysisContext(llm=FakeLLM(lambda *_: {"labels": []}), embedder=FakeEmbedder(), config=CONFIG)
    analysis = analyze(record, ctx)
    assert analysis.reuse.flags == () and analysis.reuse.pairs == ()
    assert analysis.dropped_flags == 1 and analysis.dropped_reasons[0].startswith("reuse/reasoning_reuse")


def test_a_finding_whose_source_fails_the_check_loses_its_colour_and_status():
    from ratio.context import AnalysisContext
    from ratio.paths import GOLD_DIR
    from ratio.pipeline import analyze
    from ratio.schema import CaseRecord
    from ratio.testing import FakeEmbedder, FakeLLM

    record = CaseRecord.model_validate_json((GOLD_DIR / "mock_record.json").read_text(encoding="utf-8"))
    forged_events = []
    for event in record.events:  # every arrest mention now quotes text that is not in its document
        if event.type == "arrest":
            text = "X" + event.span.text[1:]
            event = event.model_copy(update={"span": event.span.model_copy(update={"text": text})})
        forged_events.append(event)
    forged = record.model_copy(update={"events": tuple(forged_events)})

    def no_labels(system, user, schema, purpose):
        return {"labels": []} if purpose == "labels" else {"note": "", "responding": []}

    analysis = analyze(forged, AnalysisContext(llm=FakeLLM(no_labels), embedder=FakeEmbedder(), config=CONFIG))
    gc35 = next(i for i in analysis.clock.intervals if i.benchmark_id == "gc35_48h")
    assert gc35.status == "cannot_compute" and gc35.flag_id is None
    assert all(e.span.text[0] != "X" for e in gc35.evidence)
    assert not any(t.type == "arrest" for t in analysis.clock.timeline)
    assert analysis.dropped_flags >= 1 and any("gc35_48h" in reason for reason in analysis.dropped_reasons)
