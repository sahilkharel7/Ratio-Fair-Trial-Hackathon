"""Evaluation metrics. Extraction quality is measured against the hand-checked timeline; flag
recall and provenance are measured against expected_flags.json (see eval/run_eval.py)."""

from __future__ import annotations

from dataclasses import dataclass

from ratio.expected import ExpectedFlags, GoldTimeline
from ratio.gold import GoldError, resolve_anchor
from ratio.schema import CaseRecord, SourceSpan

# Event types that happen once per case; two different dates for one of them is a conflict.
SINGLETON_EVENTS = frozenset({"arrest", "first_appearance", "counsel_access", "charge", "trial_start", "verdict"})
# Event types that repeat (several extensions, several hearings): matched on their date only.
REPEATING_EVENTS = frozenset({"detention_extension", "hearing"})


@dataclass(frozen=True)
class ExtractionMetrics:
    gold_events: int
    matched: int
    correct_dates: int
    extracted_events: int
    missed: tuple[str, ...]
    wrong_dates: tuple[str, ...]
    conflicting_types: tuple[str, ...]

    @property
    def event_recall(self) -> float:
        return self.matched / self.gold_events if self.gold_events else 1.0

    @property
    def date_accuracy(self) -> float:
        return self.correct_dates / self.gold_events if self.gold_events else 1.0


def _anchor_span(record: CaseRecord, anchor) -> SourceSpan | None:
    try:
        return resolve_anchor(record, anchor)
    except GoldError:
        return None


def extraction_metrics(record: CaseRecord, gold: GoldTimeline) -> ExtractionMetrics:
    """A gold event is found if an extracted event of its type overlaps its anchor or has its date."""
    extracted = [event for event in record.events if event.type != "hearing"]
    matched = correct = 0
    missed: list[str] = []
    wrong: list[str] = []
    for gold_event in gold.events:
        anchor = _anchor_span(record, gold_event.anchor)
        same_date = [
            e for e in extracted if e.type == gold_event.type and e.parsed_date and e.parsed_date.date() == gold_event.date
        ]
        overlapping = [e for e in extracted if e.type == gold_event.type and anchor is not None and e.span.overlaps(anchor)]
        candidates = same_date if gold_event.type in REPEATING_EVENTS else same_date + overlapping
        label = f"{gold_event.type} on {gold_event.date.isoformat()}"
        if not candidates:
            missed.append(label)
            continue
        matched += 1
        if any(event.parsed_date is not None and event.parsed_date.date() == gold_event.date for event in candidates):
            correct += 1
        else:
            found = sorted({e.parsed_date.date().isoformat() for e in candidates if e.parsed_date} or {"no parsed date"})
            wrong.append(f"{label}: extracted {', '.join(found)}")
    conflicts = sorted(
        event_type
        for event_type in SINGLETON_EVENTS
        if len({e.parsed_date.date() for e in extracted if e.type == event_type and e.parsed_date}) > 1
    )
    return ExtractionMetrics(
        gold_events=len(gold.events),
        matched=matched,
        correct_dates=correct,
        extracted_events=len(extracted),
        missed=tuple(missed),
        wrong_dates=tuple(wrong),
        conflicting_types=tuple(conflicts),
    )


def event_violations(record: CaseRecord, expected: ExpectedFlags) -> tuple[str, ...]:
    """must_not_flag items of kind 'event': an extracted event of that type must not sit on the anchor."""
    violations = []
    for item in expected.must_not_flag:
        if item.kind != "event":
            continue
        anchors = [span for span in (_anchor_span(record, a) for a in item.anchors) if span is not None]
        if any(event.type == item.status and any(event.span.overlaps(a) for a in anchors) for event in record.events):
            violations.append(item.id)
    return tuple(violations)
