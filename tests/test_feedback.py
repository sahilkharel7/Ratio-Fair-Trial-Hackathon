"""Lawyers' corrections: what a decision must carry, which decision is in force, what the summary and
the regression check measure, how the store keeps them, and how the report and the eval use them."""

import json

import pytest
from pydantic import ValidationError

import eval.run_eval as run_eval
from ratio import __main__ as cli
from ratio.config import load_config
from ratio.extraction.build import load_case
from ratio.feedback import MissedIssue, Review, case_findings, current, message, regression, summarise
from ratio.history import load_alias_decisions, load_history
from ratio.paths import DEMO_CASE_DIR, MINILM_DIR
from ratio.pipeline import analysis_context, analyze, analyze_judges, demo_llm, ingest
from ratio.report import draft_report, md
from ratio.results import CaseAnalysis
from ratio.schema import Evidence, Flag, SourceSpan
from ratio.store import CaseStore


def needs_embedder(test):
    skip = pytest.mark.skipif(not (MINILM_DIR / "modules.json").exists(), reason="run scripts/fetch_models.py once")
    return pytest.mark.embed(skip(test))


def make_flag(flag_id: str = "f1", module: str = "reuse", status: str = "verbatim_reuse") -> Flag:
    span = SourceSpan(doc_id="case-1/judgment.txt", start=0, end=4, text="The ")
    return Flag(
        id=flag_id, case_id="case-1", module=module, standard_id="s", standard_label="Standard",
        status=status, message="Ratio's message.", evidence=(Evidence(role="judgment", span=span),),
    )  # fmt: skip


def review(decision: str, flag: Flag | None = None, **extra) -> Review:
    flag = flag or make_flag()
    return Review(case_id="case-1", flag_id=flag.id, decision=decision, flag=flag, **extra)


@pytest.fixture
def store(tmp_path):
    return CaseStore(tmp_path / "ratio.db")


# --- what a decision must carry -------------------------------------------------------------


def test_rejecting_or_rewording_needs_a_reason():
    with pytest.raises(ValidationError, match="reject a finding needs a reason"):
        review("rejected", note="  ")
    with pytest.raises(ValidationError, match="edit a finding needs a reason"):
        review("edited", edited_message="New wording.")
    assert review("accepted").note == ""  # accepting needs none


def test_rewording_needs_new_wording_and_only_rewording_carries_it():
    with pytest.raises(ValidationError, match="needs the reviewer's wording"):
        review("edited", note="clearer", edited_message=" ")
    with pytest.raises(ValidationError, match="same as Ratio's"):
        review("edited", note="clearer", edited_message="Ratio's message. ")
    with pytest.raises(ValidationError, match="only an edited finding"):
        review("accepted", edited_message="New wording.")


def test_the_snapshot_must_be_of_the_finding_decided():
    with pytest.raises(ValidationError, match="snapshot"):
        Review(case_id="case-1", flag_id="other", decision="accepted", flag=make_flag())


def test_a_missed_issue_needs_a_standard_and_a_description():
    with pytest.raises(ValidationError):
        MissedIssue(case_id="case-1", module="absence", standard=" ", note="x")
    with pytest.raises(ValidationError):
        MissedIssue(case_id="case-1", module="absence", standard="Art. 14(3)(f)", note="")


# --- which decision is in force -------------------------------------------------------------


def test_the_latest_decision_is_in_force_and_reopening_withdraws_it():
    accepted, rejected = review("accepted"), review("rejected", note="Quoted statute, not reasoning.")
    assert current([accepted, rejected])["f1"] == rejected
    assert current([accepted, rejected, review("reopened")]) == {}
    assert current([review("reopened"), accepted])["f1"] == accepted


def test_the_reworded_message_replaces_ratios_only_when_edited():
    flag = make_flag()
    edited = review("edited", note="clearer", edited_message="Reviewer's wording.")
    assert message(flag, edited) == "Reviewer's wording."
    assert message(flag, review("rejected", note="no")) == message(flag, None) == "Ratio's message."


# --- what the summary and the regression check measure --------------------------------------


def test_summary_counts_decisions_in_force_per_module_and_reports_stale_ones():
    shown = [make_flag("a", "reuse"), make_flag("b", "reuse"), make_flag("c", "clock", "needs_review")]
    gone = make_flag("old", "reuse")
    reviews = [
        review("accepted", shown[0]),
        review("rejected", shown[1], note="Restated charge."),
        review("accepted", gone),  # its finding is no longer produced
        review("accepted", shown[2]),
        review("reopened", shown[2]),  # withdrawn: not reviewed
    ]
    missed = [MissedIssue(case_id="case-1", module="absence", standard="Art. 14(3)(f)", note="No interpreter."),
              MissedIssue(case_id="elsewhere", module="absence", standard="x", note="y")]  # fmt: skip
    summary = summarise({"case-1": shown}, reviews, missed)
    rows = {row.module: row for row in summary.modules}
    assert (rows["reuse"].findings, rows["reuse"].reviewed, rows["reuse"].accepted, rows["reuse"].rejected) == (2, 2, 1, 1)
    assert rows["reuse"].kept_share == 0.5
    assert rows["clock"].reviewed == 0 and rows["clock"].kept_share is None
    assert rows["absence"].missed == 1  # the other case is not among the cases summarised
    assert [r.flag_id for r in summary.stale] == ["old"]
    assert [r.note for r in summary.rejections] == ["Restated charge."]
    assert (summary.reviewed, summary.findings) == (2, 3)


def test_regression_separates_kept_findings_lost_from_rejected_findings_fixed():
    kept, lost, unfixed, fixed = (make_flag(i) for i in ("kept", "lost", "unfixed", "fixed"))
    reviews = [
        review("accepted", kept),
        review("edited", lost, note="clearer", edited_message="New."),
        review("rejected", unfixed, note="no"),
        review("rejected", fixed, note="no"),
    ]
    check = regression([kept, unfixed], reviews)
    assert check.kept_still_shown == ("kept",) and check.kept_lost == ("lost",)
    assert check.rejected_still_shown == ("unfixed",) and check.rejected_gone == ("fixed",)
    assert not check.ok
    assert regression([kept, lost], reviews).ok


# --- the store ------------------------------------------------------------------------------


def test_the_store_keeps_every_decision_in_order_and_through_a_reanalysis(store):
    from tests.test_store import make_record

    store.save_case(make_record("case-1"))
    first = store.add_review(review("accepted"))
    store.add_review(review("rejected", note="Statute quote."))
    assert first.created_at  # stamped by the store
    store.save_case(make_record("case-1"))  # re-ingesting a case drops its analysis, never its decisions
    assert [r.decision for r in store.reviews("case-1")] == ["accepted", "rejected"]
    assert store.reviews("case-2") == () and len(store.reviews()) == 2


def test_missed_issues_get_an_id_and_can_be_withdrawn(store):
    issue = store.add_missed_issue(MissedIssue(case_id="case-1", module="absence", standard="Art. 14(3)(f)", note="No interpreter."))
    other = store.add_missed_issue(MissedIssue(case_id="case-2", module="clock", standard="Art. 9(3)", note="Renewal late."))
    assert issue.id is not None and issue.created_at
    assert store.missed_issues("case-1") == (issue,) and len(store.missed_issues()) == 2
    store.withdraw_missed_issue(issue.id)
    assert store.missed_issues() == (other,)


# --- the report, the CLI and the eval on the replayed demo ------------------------------------


@pytest.fixture(scope="module")
def demo():
    config = load_config()
    llm = demo_llm(config)
    record, _ = ingest(load_case(DEMO_CASE_DIR), llm, config)
    analysis = analyze(record, analysis_context(config, llm))
    history = [r for r in load_history() if r.case_id != record.case_id]
    judges = analyze_judges([*history, record], {record.case_id: analysis}, load_alias_decisions(), config)
    return record, analysis, judges, history, config


def demo_reviews(record, flags) -> list[Review]:
    absence = next(f for f in flags if f.module == "absence")
    reuse = next(f for f in flags if f.module == "reuse")
    pattern = next(f for f in flags if f.module == "judges")
    return [
        Review(case_id=record.case_id, flag_id=absence.id, decision="accepted", reviewer="Reviewer A", flag=absence, created_at="2026-10-09T10:00:00+00:00"),
        Review(case_id=record.case_id, flag_id=reuse.id, decision="edited", note="Name the passage.", flag=reuse,
               edited_message="The court's finding on the articles repeats the indictment *word for word*.", created_at="2026-10-09T10:01:00+00:00"),
        Review(case_id=record.case_id, flag_id=pattern.id, decision="rejected", note="Too few public cases.", flag=pattern, created_at="2026-10-09T10:02:00+00:00"),
    ]  # fmt: skip


@needs_embedder
def test_the_report_shows_each_decision_and_the_reviewers_wording(demo):
    record, analysis, judges, history, config = demo
    flags = case_findings(analysis, judges)
    assert len(flags) == 14 and flags[-1].module == "judges"
    reviews = demo_reviews(record, flags)
    missed = [MissedIssue(case_id=record.case_id, module="absence", standard="ICCPR Art. 14(3)(f)", note="No interpreter at hearing 2.",
                          reviewer="Reviewer A", created_at="2026-10-09T10:03:00+00:00")]  # fmt: skip
    stale = Review(case_id=record.case_id, flag_id="gone", decision="accepted", flag=make_flag("gone"))
    report = draft_report(record, analysis, judges, config, history=history, reviews=[*reviews, stale], missed=missed)
    notes = config.messages.notes
    assert "**Reviewed: 3 of 14 findings (1 accepted, 1 reworded, 1 rejected). Issues recorded that Ratio did not flag: 1.**" in report
    assert "Earlier decisions on findings this analysis no longer produces: 1." in report
    edited = md(reviews[1].edited_message)
    assert edited in report.split("## Annex")[0]  # the body shows the reviewer's wording
    assert f"*{notes['report_ratio_wording']}:* {md(reviews[1].flag.message)}" in report  # Ratio's is kept in the annex
    assert "*(Accepted by the reviewer)*" in report and "*(Rejected by the reviewer)*" in report
    assert "**Rejected by the reviewer** (2026-10-09). Reason: Too few public cases." in report
    assert "**Accepted by the reviewer** (2026-10-09, Reviewer A)." in report
    assert f"## 6. {notes['report_missed']}" in report
    assert "- **Rights coverage, ICCPR Art. 14(3)(f):** No interpreter at hearing 2. (Reviewer A, 2026-10-09)" in report
    other_case = [r.model_copy(update={"case_id": "another-case"}) for r in reviews]
    assert "Reviewed:" not in draft_report(record, analysis, judges, config, history=history, reviews=other_case)


@needs_embedder
def test_the_feedback_command_summarises_and_exports_decisions(demo, tmp_path, monkeypatch, capsys):
    record, analysis, judges, _history, _config = demo
    monkeypatch.setenv("RATIO_DB", str(tmp_path / "ratio.db"))
    store = CaseStore()
    for case in load_history():
        store.save_case(case)
    store.save_case(record)
    store.save_analysis(analysis)
    for item in demo_reviews(record, case_findings(analysis, judges)):
        store.add_review(item)
    assert cli.main(["feedback", "--json", str(tmp_path / "feedback.json")]) == 0
    out = capsys.readouterr().out
    assert "3 of 14 findings reviewed, over 1 analysed cases" in out
    assert "Judge profile      1/1   reviewed; 0 accepted, 0 reworded, 1 rejected (kept 0%)" in out
    assert "rejected (venn-2025, Pattern indicator (prompt for review)): Too few public cases." in out
    export = json.loads((tmp_path / "feedback.json").read_text(encoding="utf-8"))
    assert [r["decision"] for r in export["reviews"]] == ["accepted", "edited", "rejected"]
    assert export["summary"]["cases"] == 1 and export["missed_issues"] == []


@needs_embedder
def test_the_eval_compares_fresh_findings_with_reviewers_decisions(demo, tmp_path, monkeypatch, capsys):
    record, analysis, judges, _history, _config = demo
    monkeypatch.setenv("RATIO_DB", str(tmp_path / "ratio.db"))
    monkeypatch.setattr(run_eval, "OUT_DIR", tmp_path)
    store = CaseStore()
    for item in demo_reviews(record, case_findings(analysis, judges)):
        store.add_review(item)
    assert run_eval.main(["--reviews"]) == 0
    out = capsys.readouterr().out
    assert "Against reviewers' decisions on this case: 2 kept by a reviewer, 1 rejected" in out
    assert "still produced although a reviewer rejected it" in out
    result = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert len(result["reviews"]["kept_still_shown"]) == 2

    lost = Review(case_id=record.case_id, flag_id="no-longer-produced", decision="accepted", flag=make_flag("no-longer-produced"))
    store.add_review(lost)
    assert run_eval.main(["--reviews"]) == 1  # a finding a reviewer kept disappeared: a regression
    assert "regression: a finding a reviewer kept is no longer produced (no-longer-produced)" in capsys.readouterr().out
    assert run_eval.main([]) == 0  # without --reviews the store is not read


def test_an_analysis_without_a_judge_has_only_its_own_findings():
    assert case_findings(CaseAnalysis(case_id="case-1"), None) == ()
