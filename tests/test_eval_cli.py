"""The eval replays the demo, lists what it found and missed, and fails on any unsourced flag."""

import json

import eval.run_eval as run_eval
from ratio.paths import GOLD_DIR
from ratio.results import CaseAnalysis, ReuseResult
from ratio.schema import CaseRecord, Evidence, Flag, SourceSpan

MOCK = CaseRecord.model_validate_json((GOLD_DIR / "mock_record.json").read_text(encoding="utf-8"))


def test_eval_replays_the_demo_with_full_provenance_and_recall(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_eval, "OUT_DIR", tmp_path)
    assert run_eval.main([]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC demo case (in-sample" in out and "replayed from cache" in out
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["provenance"]["shown_with_exact_spans"] == 1.0
    assert report["provenance"]["before_enforcement"] == 1.0
    assert report["flags"]["missed"] == [] and report["flags"]["violations"] == []
    assert report["synthetic"] and report["in_sample"] and report["replayed"]


def test_a_shown_flag_without_its_exact_source_fails_provenance():
    judgment = MOCK.documents_of_type("judgment")[0]
    forged = SourceSpan(doc_id=judgment.id, start=0, end=5, text="FORGE")
    flag = Flag(
        id="f", case_id=MOCK.case_id, module="reuse", standard_id="reasoning_reuse", standard_label="x",
        status="verbatim_reuse", message="m", evidence=(Evidence(role="judgment", span=forged),),
    )  # fmt: skip
    analysis = CaseAnalysis(case_id=MOCK.case_id, reuse=ReuseResult(judgment_doc_id=judgment.id, indictment_doc_id=None, flags=(flag,)))
    assert run_eval.provenance(MOCK, analysis)["shown_with_exact_spans"] == 0.0


def test_the_eval_fails_when_any_flag_had_to_be_dropped(tmp_path, monkeypatch):
    from ratio.modules import reuse

    real_run = reuse.run

    def forging_run(record, ctx):
        result = real_run(record, ctx)
        judgment = record.documents_of_type("judgment")[0]
        forged = SourceSpan(doc_id=judgment.id, start=0, end=5, text="FORGE")
        flag = Flag(
            id="forged", case_id=record.case_id, module="reuse", standard_id="reasoning_reuse", standard_label="x",
            status="verbatim_reuse", message="m", evidence=(Evidence(role="judgment", span=forged),),
        )  # fmt: skip
        return result.model_copy(update={"flags": (*result.flags, flag)})

    monkeypatch.setattr(reuse, "run", forging_run)
    monkeypatch.setattr(run_eval, "OUT_DIR", tmp_path)
    assert run_eval.main([]) == 1
