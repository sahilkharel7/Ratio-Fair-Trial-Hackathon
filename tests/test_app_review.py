"""The Review page, driven headless with AppTest on the replayed demo: every finding can be decided,
a rejection without a reason is refused, decisions are saved append-only and change what the Case
page, the page itself and the report show, and missed issues can be recorded and withdrawn."""

import pytest
from streamlit.testing.v1 import AppTest

from ratio.paths import MINILM_DIR, REPO_ROOT
from ratio.store import CaseStore

APP = REPO_ROOT / "app" / "main.py"
PAGE = "views/review.py"
pytestmark = [
    pytest.mark.embed,
    pytest.mark.skipif(not (MINILM_DIR / "modules.json").exists(), reason="run scripts/fetch_models.py once"),
]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RATIO_DB", str(tmp_path_factory.mktemp("review") / "ratio.db"))
        at = AppTest.from_file(str(APP), default_timeout=180)
        at.run()
        at.button(key="load_demo").click().run()
        assert not at.exception
        at.switch_page(PAGE).run()
        assert not at.exception and not at.error
        yield at


def store() -> CaseStore:
    return CaseStore()


def findings():
    from ratio.feedback import case_findings
    from ratio.history import load_all_alias_decisions
    from ratio.config import default_config
    from ratio.pipeline import analyze_judges

    cases = store()
    case_id = cases.last_case()
    records = cases.all_records()
    analyses = {r.case_id: a for r in records if (a := cases.load_analysis(r.case_id)) is not None}
    judges = analyze_judges(records, analyses, load_all_alias_decisions(), default_config())
    return case_id, case_findings(analyses[case_id], judges)


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_every_finding_has_a_decision_form_and_nothing_is_reviewed_yet(app):
    _, flags = findings()
    assert len(flags) == 14
    for flag in flags:
        assert app.button(key=f"save-{flag.id}") is not None
    assert metric(app, "Findings") == "14" and metric(app, "Reviewed") == "0"
    assert any("SYNTHETIC DATA" in el.proto.body for el in app.get("html"))


def test_a_rejection_without_a_reason_is_refused(app):
    case_id, flags = findings()
    flag = flags[0]
    app.radio(key=f"decision-{flag.id}").set_value("rejected")
    app.button(key=f"save-{flag.id}").click().run()
    assert any("needs a reason" in e.value for e in app.error)
    assert store().reviews(case_id) == ()


def test_accept_reword_reject_and_reopen_are_saved_in_order(app):
    case_id, flags = findings()
    accepted, reworded, rejected = flags[0], flags[5], flags[-1]
    app.text_input(key="ratio_reviewer").input("Reviewer A").run()
    app.radio(key=f"decision-{accepted.id}").set_value("accepted")
    app.button(key=f"save-{accepted.id}").click().run()
    assert not app.error

    app.radio(key=f"decision-{reworded.id}").set_value("edited")
    app.text_area(key=f"note-{reworded.id}").input("Name the passage.")
    app.text_area(key=f"wording-{reworded.id}").input("The judgment repeats the indictment's account word for word.")
    app.button(key=f"save-{reworded.id}").click().run()
    assert not app.error
    assert any("The judgment repeats the indictment" in m.value for m in app.markdown)

    app.radio(key=f"decision-{rejected.id}").set_value("rejected")
    app.text_area(key=f"note-{rejected.id}").input("Too few cases to draw on.")
    app.button(key=f"save-{rejected.id}").click().run()
    assert not app.error
    assert (metric(app, "Reviewed"), metric(app, "Accepted"), metric(app, "Reworded"), metric(app, "Rejected")) == ("3", "1", "1", "1")

    app.button(key=f"reopen-{accepted.id}").click().run()
    assert metric(app, "Reviewed") == "2"
    saved = store().reviews(case_id)
    assert [r.decision for r in saved] == ["accepted", "edited", "rejected", "reopened"]
    assert all(r.reviewer == "Reviewer A" for r in saved)
    assert saved[1].edited_message == "The judgment repeats the indictment's account word for word."


def test_the_filter_shows_only_findings_not_reviewed(app):
    _, flags = findings()
    app.segmented_control(key="review_filter").set_value("Not reviewed").run()
    shown = {b.key for b in app.button if (b.key or "").startswith("save-")}
    assert len(shown) == 12
    app.segmented_control(key="review_filter").set_value("Reviewed").run()
    assert len({b.key for b in app.button if (b.key or "").startswith("save-")}) == 2
    app.segmented_control(key="review_filter").set_value("All").run()


def test_a_missed_issue_is_recorded_and_withdrawn(app):
    case_id, _ = findings()
    app.button(key="missed-save").click().run()
    assert any("Not saved" in e.value for e in app.error)  # blank standard and description
    app.selectbox(key="missed-module").set_value("absence")
    app.text_input(key="missed-standard").input("ICCPR Art. 14(3)(f)")
    app.text_area(key="missed-note").input("No interpreter at hearing 2.")
    app.button(key="missed-save").click().run()
    [issue] = store().missed_issues(case_id)
    assert issue.module == "absence" and issue.reviewer == "Reviewer A"
    assert any("No interpreter at hearing 2" in m.value for m in app.markdown)
    app.button(key=f"withdraw-{issue.id}").click().run()
    assert store().missed_issues(case_id) == ()


def test_the_case_page_counts_the_decisions_in_force(app):
    app.switch_page("views/case.py").run()
    assert not app.exception
    labels = [link.proto.label for link in app.get("page_link")]
    assert "Review: 2 of 14 findings reviewed by a lawyer" in labels
