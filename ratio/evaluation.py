"""Evaluation metrics. Extraction quality is measured against the hand-checked timeline; flag
recall and provenance are measured against expected_flags.json (see eval/run_eval.py)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ratio.expected import ExpectedFlags, ExpectedItem, GoldTimeline
from ratio.gold import GoldError, resolve_anchor
from ratio.results import CaseAnalysis
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
        """Share of the events found whose date is right (a missed event counts against recall only)."""
        return self.correct_dates / self.matched if self.matched else 1.0


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
        anchors = [span for span in (_anchor_span(record, a) for a in gold_event.anchors) if span is not None]
        same_date = [
            e for e in extracted if e.type == gold_event.type and e.parsed_date and e.parsed_date.date() == gold_event.date
        ]
        overlapping = [e for e in extracted if e.type == gold_event.type and any(e.span.overlaps(a) for a in anchors)]
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


@dataclass(frozen=True)
class FlagRecall:
    """Expected outputs found and missed, and must_not_flag items that were flagged."""

    found: tuple[str, ...]
    missed: tuple[str, ...]
    violations: tuple[str, ...]

    @property
    def recall(self) -> float:
        total = len(self.found) + len(self.missed)
        return len(self.found) / total if total else 1.0


def _covers(spans: Sequence[SourceSpan], anchors: Sequence[SourceSpan], match: str) -> bool:
    hits = [any(span.overlaps(anchor) for span in spans) for anchor in anchors]
    return all(hits) if match == "all" else any(hits) or not hits


def _shown(item: ExpectedItem, record: CaseRecord, analysis: CaseAnalysis) -> bool:
    """Whether the analysis shows this output: same module, standard and status, over its anchors."""
    anchors = [_anchor_span(record, anchor) for anchor in item.anchors]
    if any(anchor is None for anchor in anchors):
        raise GoldError(f"{item.id}: an anchor does not occur exactly once in the case documents")
    absence = analysis.absence
    if item.kind == "flag":
        return any(
            flag.module == item.module
            and flag.standard_id == item.standard_id
            and item.status in ("any", flag.status)
            and _covers(flag.spans, anchors, item.match)
            for flag in analysis.all_flags()
        )
    if item.kind == "status" and absence is not None:
        assessment = next((a for a in absence.assessments if a.rubric_id == item.standard_id), None)
        if assessment is None or assessment.status != item.status:
            return False
        flag = next((f for f in absence.flags if f.id == assessment.flag_id), None)
        return _covers(flag.spans if flag else (), anchors, item.match)
    if item.kind == "follow_up" and absence is not None:
        follow_up = next((u for u in absence.follow_ups if u.rubric_id == item.standard_id), None)
        return follow_up is not None and _covers([e.span for e in follow_up.context], anchors, item.match)
    return False  # events are checked on the record (event_violations)


def flag_recall(record: CaseRecord, analysis: CaseAnalysis, expected: ExpectedFlags) -> FlagRecall:
    found = tuple(item.id for item in expected.expected if _shown(item, record, analysis))
    missed = tuple(item.id for item in expected.expected if item.id not in found)
    forbidden = (item.model_copy(update={"match": "any"}) for item in expected.must_not_flag if item.kind != "event")
    violations = tuple(item.id for item in forbidden if _shown(item, record, analysis))  # any anchor flagged is a violation
    return FlagRecall(found=found, missed=missed, violations=violations)


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
