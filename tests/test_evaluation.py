"""Extraction metrics against the hand-checked timeline, and the event-level negative control."""

import datetime as dt
import json

from ratio.evaluation import event_violations, extraction_metrics
from ratio.expected import ExpectedFlags, GoldTimeline
from ratio.gold import resolve_anchor
from ratio.paths import GOLD_DIR
from ratio.schema import CaseRecord

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
