"""Extraction metrics against the hand-checked timeline, expected-flag recall, and the negative controls."""

import datetime as dt
import json

from ratio.evaluation import event_violations, extraction_metrics, flag_recall
from ratio.expected import Anchor, ExpectedFlags, GoldTimeline
from ratio.gold import resolve_anchor
from ratio.paths import GOLD_DIR
from ratio.results import CaseAnalysis, ClockResult, ReuseResult
from ratio.schema import CaseRecord, Evidence, Flag

MOCK = CaseRecord.model_validate_json((GOLD_DIR / "mock_record.json").read_text(encoding="utf-8"))
GOLD = GoldTimeline.model_validate(json.loads((GOLD_DIR / "gold_timeline.json").read_text(encoding="utf-8")))
EXPECTED = ExpectedFlags.model_validate(json.loads((GOLD_DIR / "expected_flags.json").read_text(encoding="utf-8")))


def with_events(events) -> CaseRecord:
    return MOCK.model_copy(update={"events": tuple(events)})


def test_the_gold_record_scores_perfectly():
    metrics = extraction_metrics(MOCK, GOLD)
    assert metrics.event_recall == 1.0
    assert metrics.date_accuracy == 1.0
    assert metrics.conflicting_types == ()


def test_a_wrong_date_lowers_date_accuracy_and_is_listed():
    events = [
        e.model_copy(update={"parsed_date": dt.datetime(2025, 2, 18)}) if e.type == "first_appearance" else e
        for e in MOCK.events
    ]
    metrics = extraction_metrics(with_events(events), GOLD)
    assert metrics.date_accuracy < 1.0
    assert any("first_appearance" in item for item in metrics.wrong_dates)


def test_a_missing_event_lowers_recall_and_is_listed():
    metrics = extraction_metrics(with_events(e for e in MOCK.events if e.type != "counsel_access"), GOLD)
    assert metrics.event_recall < 1.0
    assert any("counsel_access" in item for item in metrics.missed)


def test_a_missing_repeated_event_is_missed_not_misdated():
    events = [e for e in MOCK.events if not (e.type == "detention_extension" and e.parsed_date.month == 5)]
    metrics = extraction_metrics(with_events(events), GOLD)
    assert any("detention_extension on 2025-05-14" in item for item in metrics.missed)
    assert metrics.wrong_dates == ()


def test_two_dates_for_one_arrest_are_a_conflict():
    arrest = next(e for e in MOCK.events if e.type == "arrest")
    other = arrest.model_copy(update={"id": "other", "parsed_date": dt.datetime(2025, 2, 13)})
    metrics = extraction_metrics(with_events([*MOCK.events, other]), GOLD)
    assert "arrest" in metrics.conflicting_types


def test_prosecutor_appearance_typed_as_first_appearance_is_a_violation():
    anchor = next(a for item in EXPECTED.must_not_flag if item.kind == "event" for a in item.anchors)
    span = resolve_anchor(MOCK, anchor)
    template = next(e for e in MOCK.events if e.type == "first_appearance")
    wrong = template.model_copy(update={"id": "wrong", "span": span, "quote_span": span, "date_span": None})
    assert event_violations(with_events([*MOCK.events, wrong]), EXPECTED) == ("no_prosecutor_as_first_appearance",)
    assert event_violations(MOCK, EXPECTED) == ()


# --- expected flags: recall, misses and must_not_flag violations -------------------------------


def flag_on(module: str, standard_id: str, status: str, *quotes: tuple[str, str], review_status="needs_legal_review") -> Flag:
    spans = [resolve_anchor(MOCK, Anchor(doc=doc, quote=quote)) for doc, quote in quotes]
    return Flag(
        id=f"{standard_id}-{status}",
        case_id=MOCK.case_id,
        module=module,
        standard_id=standard_id,
        standard_label=standard_id,
        status=status,
        message="m",
        evidence=tuple(Evidence(role="mention", span=span) for span in spans),
        review_status=review_status,
    )


def analysis_with(*flags: Flag) -> CaseAnalysis:
    reuse_flags = tuple(f for f in flags if f.module == "reuse")
    clock_flags = tuple(f for f in flags if f.module == "clock")
    return CaseAnalysis(
        case_id=MOCK.case_id,
        clock=ClockResult(timeline=(), intervals=(), flags=clock_flags),
        reuse=ReuseResult(judgment_doc_id=None, indictment_doc_id=None, flags=reuse_flags),
    )


def test_nothing_shown_means_every_expected_output_is_missed():
    recall = flag_recall(MOCK, CaseAnalysis(case_id=MOCK.case_id), EXPECTED)
    assert recall.found == () and recall.violations == ()
    assert set(recall.missed) == {item.id for item in EXPECTED.expected} and recall.recall == 0.0


def test_match_any_needs_one_anchor_and_status_must_agree():
    red = flag_on("clock", "gc35_48h", "exceeds_benchmark", ("indictment.txt", "in the early morning of 14 February 2025"), review_status="confirmed")
    amber = flag_on("clock", "gc35_48h", "needs_review", ("indictment.txt", "in the early morning of 14 February 2025"))
    assert "gc35_first_appearance" in flag_recall(MOCK, analysis_with(red), EXPECTED).found
    assert "gc35_first_appearance" in flag_recall(MOCK, analysis_with(amber), EXPECTED).missed


def test_flag_on_legitimate_quotation_or_an_answered_argument_is_a_violation():
    statute = flag_on("reuse", "reasoning_reuse", "verbatim_reuse", ("judgment.txt", "Whoever disseminates information that he knows to be false"))
    answered = flag_on(
        "reuse", "unaddressed_defense_argument", "unaddressed_argument",
        ("notes/hearing_3.txt", "the first article was published before the amendment to Article 214 entered into force"),
    )  # fmt: skip
    recall = flag_recall(MOCK, analysis_with(statute, answered), EXPECTED)
    assert set(recall.violations) == {"no_reuse_statute_quote", "no_unaddressed_retroactivity"}


def test_a_pure_miss_lowers_recall_but_not_date_accuracy():
    metrics = extraction_metrics(with_events(e for e in MOCK.events if e.type != "counsel_access"), GOLD)
    assert metrics.event_recall < 1.0 and metrics.date_accuracy == 1.0


def test_a_misdated_mention_that_is_not_the_first_annotation_is_a_wrong_date_not_a_miss():
    judgment_arrest = next(e for e in MOCK.events if e.type == "arrest" and e.span.doc_id.endswith("judgment.txt"))
    others = [e for e in MOCK.events if e.type != "arrest"]
    misdated = judgment_arrest.model_copy(update={"parsed_date": dt.datetime(2025, 2, 13)})
    metrics = extraction_metrics(with_events([*others, misdated]), GOLD)
    assert not any("arrest" in item for item in metrics.missed)
    assert any(item.startswith("arrest on 2025-02-14") for item in metrics.wrong_dates)


def test_a_must_not_flag_item_is_violated_by_a_flag_on_any_of_its_anchors():
    from ratio.expected import ExpectedItem

    two_anchors = ExpectedItem(
        id="no_reuse_quotation", module="reuse", standard_id="reasoning_reuse", kind="flag", status="any",
        anchors=(Anchor(doc="judgment.txt", quote="The accused is charged with disseminating false information"),
                 Anchor(doc="judgment.txt", quote="Whoever disseminates information that he knows to be false")),
    )  # fmt: skip
    expected = EXPECTED.model_copy(update={"must_not_flag": (two_anchors,)})
    statute = flag_on("reuse", "reasoning_reuse", "verbatim_reuse", ("judgment.txt", "Whoever disseminates information that he knows to be false"))
    assert flag_recall(MOCK, analysis_with(statute), expected).violations == ("no_reuse_quotation",)


def test_an_anchor_that_does_not_resolve_is_an_eval_error_not_a_pass():
    import pytest

    from ratio.expected import ExpectedItem
    from ratio.gold import GoldError

    ambiguous = ExpectedItem(id="x", module="reuse", standard_id="reasoning_reuse", kind="flag", status="any", anchors=(Anchor(doc="judgment.txt", quote="The accused"),))
    with pytest.raises(GoldError, match="exactly once"):
        flag_recall(MOCK, analysis_with(), EXPECTED.model_copy(update={"must_not_flag": (ambiguous,)}))
