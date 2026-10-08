"""Module 2: Procedural Clock (ICCPR Arts. 9(3), 14(3)(c)).

Builds a timeline from the extracted events and measures each benchmark interval in code; the
model never does arithmetic. Every mention of an event in every document is a candidate date, and
an interval covers every reading of those dates (day-level dates widen it by a day each side).
An interval is red only if even its shortest reading exceeds a confirmed benchmark and its dates
need no review, so a single stray mention can neither erase nor create a red flag.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass

from ratio.config import Benchmark
from ratio.context import AnalysisContext
from ratio.display import duration_text
from ratio.messages import render
from ratio.results import ClockResult, Interval, IntervalStatus, TimelineEvent, TimelineState
from ratio.schema import CaseRecord, Event, Evidence, Flag, stable_id

REPEATING = frozenset({"detention_extension", "hearing"})
DAY_LEVEL = frozenset({"date", "datetime"})
DAY = dt.timedelta(days=1)


def _usable(event: Event) -> bool:
    return event.parsed_date is not None and event.precision in DAY_LEVEL


def _bounds(event: Event) -> tuple[dt.datetime, dt.datetime]:
    """Earliest and latest moment the event can have happened, given its precision."""
    moment = event.parsed_date
    if event.precision == "datetime":
        return moment, moment
    start = dt.datetime.combine(moment.date(), dt.time())
    return start, start + DAY


def _timeline_event(event_type: str, mentions: list[Event]) -> TimelineEvent:
    usable = [e for e in mentions if _usable(e)]
    days = sorted({e.parsed_date.date() for e in usable})
    reasons = tuple(dict.fromkeys(reason for e in mentions for reason in e.review_reasons))
    state: TimelineState
    if not usable:
        state = "undated"
    elif len(days) > 1:
        state = "conflict"
    elif reasons:
        state = "needs_review"
    else:
        state = "ok"
    first = min(usable, key=lambda e: e.parsed_date) if usable else None
    return TimelineEvent(
        id=_timeline_id(mentions),
        type=event_type,
        date=first.parsed_date if first else None,
        precision=first.precision if first else "unknown",
        state=state,
        mentions=tuple(Evidence(role="mention", span=e.span) for e in mentions),
        candidate_dates=tuple(day.isoformat() for day in days),
        review_reasons=reasons,
    )


def _timeline_id(mentions: list[Event]) -> str:
    return stable_id(*(e.id for e in mentions), "timeline")


def _groups(events: tuple[Event, ...]) -> dict[tuple, list[Event]]:
    """Mentions of one real-world event: one group per type, or per type and date for repeating types."""
    groups: dict[tuple, list[Event]] = defaultdict(list)
    for event in events:
        key = (event.type, event.parsed_date.date()) if event.type in REPEATING and event.parsed_date else (event.type,)
        groups[key].append(event)
    return groups


def build_timeline(events: tuple[Event, ...]) -> tuple[TimelineEvent, ...]:
    timeline = [_timeline_event(key[0], mentions) for key, mentions in _groups(events).items()]
    return tuple(sorted(timeline, key=lambda t: (t.date is None, t.date or dt.datetime.max, t.type)))


def timeline_ids(events: tuple[Event, ...]) -> dict[str, str]:
    """Record event id -> id of the timeline event that merges it."""
    return {event.id: _timeline_id(mentions) for mentions in _groups(events).values() for event in mentions}


def _usable_of(events: tuple[Event, ...], event_type: str) -> tuple[list[Event], list[Event]]:
    """Day-level mentions of a type: (those needing no review, all of them)."""
    usable = [e for e in events if e.type == event_type and _usable(e)]
    return [e for e in usable if not e.needs_review], usable


def _hours(delta: dt.timedelta) -> float:
    return round(delta.total_seconds() / 3600, 2)


def _first_after(starts: list[Event], ends: list[Event]) -> tuple[float, float] | None:
    """Bounds in hours when each start reading is paired with the first end reading that can follow it."""
    lows, highs = [], []
    for start in starts:
        later = [end for end in ends if _bounds(end)[1] >= _bounds(start)[0]]  # an end before the start is impossible
        if later:
            first = min(later, key=lambda end: _bounds(end)[0])
            lows.append(_bounds(first)[0] - _bounds(start)[1])
            highs.append(_bounds(first)[1] - _bounds(start)[0])
    return (max(_hours(min(lows)), 0.0), _hours(max(highs))) if lows else None


@dataclass(frozen=True)
class _Measure:
    low: float | None
    high: float | None
    all_low: float | None  # lowest reading over every usable mention, including those marked for review
    starts: list[Event]
    ends: list[Event]
    review: bool
    note: str


def _conflict_note(benchmark: Benchmark, all_starts: list[Event], all_ends: list[Event]) -> str:
    conflicts = [
        f"{event_type.replace('_', ' ')} ({', '.join(sorted({e.parsed_date.date().isoformat() for e in group}))})"
        for event_type, group in ((benchmark.from_event, all_starts), (benchmark.to_event, all_ends))
        if len({e.parsed_date.date() for e in group}) > 1
    ]
    return f"Dates conflict in the record for {'; '.join(conflicts)}." if conflicts else ""


def _measure(benchmark: Benchmark, record: CaseRecord) -> _Measure:
    clean_starts, all_starts = _usable_of(record.events, benchmark.from_event)
    clean_ends, all_ends = _usable_of(record.events, benchmark.to_event)
    starts, ends = clean_starts or all_starts, clean_ends or all_ends
    review = (not clean_starts and bool(all_starts)) or (not clean_ends and bool(all_ends))
    bounds = _first_after(starts, ends) if starts and ends else None
    if bounds is None:
        missing = benchmark.from_event if not starts else benchmark.to_event
        return _Measure(None, None, None, starts, ends, False, f"No usable date for {missing.replace('_', ' ')}.")
    every = _first_after(all_starts, all_ends)
    ends = [end for end in ends if any(_bounds(end)[1] >= _bounds(start)[0] for start in starts)]
    return _Measure(bounds[0], bounds[1], every[0] if every else bounds[0], starts, ends, review, _conflict_note(benchmark, all_starts, all_ends))


def _status(benchmark: Benchmark, measure: _Measure) -> IntervalStatus:
    threshold, low, high = benchmark.threshold_hours, measure.low, measure.high
    if low is None or high is None:
        return "cannot_compute"
    if threshold is None:
        return "measured"
    if benchmark.review_status != "confirmed" or measure.review:
        return "needs_review"
    if high <= threshold:
        return "within_benchmark"
    if low > threshold and measure.all_low is not None and measure.all_low <= threshold:
        return "needs_review"  # a reading marked for review would put the interval within the benchmark
    return "exceeds_benchmark" if low > threshold else "may_exceed"


def _flag(record: CaseRecord, benchmark: Benchmark, interval: Interval, ctx: AnalysisContext) -> Flag:
    labels = ctx.config.messages.event_labels
    template = "clock_exceeds" if interval.status == "exceeds_benchmark" else "clock_needs_review"
    source = ctx.config.benchmarks.sources[benchmark.citation.instrument]
    return Flag(
        id=interval.flag_id,
        case_id=record.case_id,
        module="clock",
        standard_id=benchmark.id,
        standard_label=f"{benchmark.provision}: {benchmark.name}",
        status="exceeds_benchmark" if interval.status == "exceeds_benchmark" else "needs_review",
        message=render(
            ctx.config.messages,
            template,
            from_label=labels[benchmark.from_event],
            to_label=labels[benchmark.to_event],
            duration=duration_text(interval.min_hours, interval.max_hours),
            threshold=f"{benchmark.threshold_hours:g}-hour",
            citation_note=benchmark.flag_note,
        ).strip(),
        evidence=interval.evidence,
        citation=f"{source.symbol}, para. {benchmark.citation.paras}: “{benchmark.citation.quote}”",
        review_status=benchmark.review_status,
    )


def _interval(benchmark: Benchmark, record: CaseRecord, merged: dict[str, str]) -> Interval:
    measure = _measure(benchmark, record)
    low, high, starts, ends, note = measure.low, measure.high, measure.starts, measure.ends, measure.note
    status = _status(benchmark, measure)
    flagged = status in {"exceeds_benchmark", "may_exceed", "needs_review"}
    return Interval(
        id=stable_id(record.case_id, "clock", benchmark.id),
        benchmark_id=benchmark.id,
        benchmark_name=benchmark.name,
        from_event_id=merged[starts[0].id] if starts else None,
        to_event_id=merged[ends[0].id] if ends else None,
        min_hours=low,
        max_hours=high,
        threshold_hours=benchmark.threshold_hours,
        status=status,
        citation=benchmark.citation.paras,
        review_status=benchmark.review_status,
        evidence=tuple(Evidence(role="from_event", span=e.span) for e in starts)
        + tuple(Evidence(role="to_event", span=e.span) for e in ends),
        note=note,
        flag_id=stable_id(record.case_id, "clock", benchmark.id, status) if flagged else None,
    )


def run(record: CaseRecord, ctx: AnalysisContext) -> ClockResult:
    merged = timeline_ids(record.events)
    intervals = tuple(_interval(benchmark, record, merged) for benchmark in ctx.config.benchmarks.benchmarks)
    by_id = {benchmark.id: benchmark for benchmark in ctx.config.benchmarks.benchmarks}
    flags = tuple(_flag(record, by_id[i.benchmark_id], i, ctx) for i in intervals if i.flag_id is not None)
    return ClockResult(timeline=build_timeline(record.events), intervals=intervals, flags=flags)
