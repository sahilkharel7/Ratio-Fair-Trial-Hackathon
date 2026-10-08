"""Hard rule 2: a flag is shown only if every span it rests on is found, character for
character, in the original document text. Flags that fail are dropped and counted."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ratio.results import AbsenceResult, ClockResult, ReuseResult
from ratio.schema import CaseRecord, Flag, FollowUp, SourceSpan

DocResolver = Callable[[str], str | None]


def resolver_for(records: Iterable[CaseRecord]) -> DocResolver:
    """Look up document text by id across several cases (judge indicators span cases)."""
    texts = {doc.id: doc.text for record in records for doc in record.documents}
    return texts.get


def span_is_valid(span: SourceSpan, resolve: DocResolver) -> bool:
    text = resolve(span.doc_id)
    if text is None or not (0 <= span.start < span.end <= len(text)):
        return False
    return text[span.start : span.end] == span.text


@dataclass(frozen=True)
class EnforcementResult:
    kept: tuple[Flag, ...]
    dropped: tuple[tuple[Flag, str], ...]

    @property
    def rate(self) -> float:
        """Share of flags whose spans were all found (1.0 when there were no flags)."""
        total = len(self.kept) + len(self.dropped)
        return 1.0 if total == 0 else len(self.kept) / total


def enforce(flags: Iterable[Flag], resolve: DocResolver) -> EnforcementResult:
    kept: list[Flag] = []
    dropped: list[tuple[Flag, str]] = []
    for flag in flags:
        if not flag.evidence:
            dropped.append((flag, "flag has no evidence span"))
            continue
        bad = [span for span in flag.spans if not span_is_valid(span, resolve)]
        if bad:
            where = ", ".join(f"{span.doc_id}[{span.start}:{span.end}]" for span in bad)
            dropped.append((flag, f"span not found in source: {where}"))
        else:
            kept.append(flag)
    return EnforcementResult(kept=tuple(kept), dropped=tuple(dropped))


def filter_follow_up(follow_up: FollowUp, resolve: DocResolver) -> FollowUp:
    """Follow-ups need no evidence, but any context span they show must still be real."""
    context = tuple(item for item in follow_up.context if span_is_valid(item.span, resolve))
    return follow_up.model_copy(update={"context": context})


# --- whole module results: nothing on screen may rest on a span that failed the check --------------


def _valid(items, resolve: DocResolver) -> tuple:
    return tuple(item for item in items if span_is_valid(item.span, resolve))


def _kept(flags: tuple[Flag, ...], resolve: DocResolver, dropped: list[str]) -> tuple[tuple[Flag, ...], set[str]]:
    outcome = enforce(flags, resolve)
    dropped.extend(f"{flag.module}/{flag.standard_id}: {reason}" for flag, reason in outcome.dropped)
    return outcome.kept, {flag.id for flag in outcome.kept}


def check_absence(result: AbsenceResult, resolve: DocResolver, dropped: list[str]) -> AbsenceResult:
    """A guarantee whose flag fails the check falls back to no evidence."""
    flags, kept = _kept(result.flags, resolve, dropped)
    assessments = []
    for assessment in result.assessments:
        update = {
            "supporting": _valid(assessment.supporting, resolve),
            "contradicting": _valid(assessment.contradicting, resolve),
            "context": _valid(assessment.context, resolve),
        }
        if assessment.follow_up is not None:
            update["follow_up"] = filter_follow_up(assessment.follow_up, resolve)
        if assessment.flag_id is not None and assessment.flag_id not in kept:
            update |= {"status": "no_evidence", "flag_id": None}
        assessments.append(assessment.model_copy(update=update))
    follow_ups = tuple(filter_follow_up(follow_up, resolve) for follow_up in result.follow_ups)
    return result.model_copy(update={"assessments": tuple(assessments), "flags": flags, "follow_ups": follow_ups})


def check_clock(result: ClockResult, resolve: DocResolver, dropped: list[str]) -> ClockResult:
    """An interval whose flag fails the check is shown as not measurable, never in colour."""
    flags, kept = _kept(result.flags, resolve, dropped)
    intervals = []
    for interval in result.intervals:
        update: dict = {"evidence": _valid(interval.evidence, resolve)}
        if interval.flag_id is not None and interval.flag_id not in kept:
            update |= {"status": "cannot_compute", "flag_id": None, "note": "Its dates failed the check against the source text."}
        intervals.append(interval.model_copy(update=update))
    timeline = tuple(
        event.model_copy(update={"mentions": _valid(event.mentions, resolve)}) for event in result.timeline if _valid(event.mentions, resolve)
    )
    return result.model_copy(update={"flags": flags, "intervals": tuple(intervals), "timeline": timeline})


def check_reuse(result: ReuseResult, resolve: DocResolver, dropped: list[str]) -> ReuseResult:
    """Pairs and argument checks go with their flags; every span shown is checked."""
    flags, kept = _kept(result.flags, resolve, dropped)
    arguments = tuple(
        check.model_copy(update={"responding": tuple(s for s in check.responding if span_is_valid(s, resolve))})
        for check in result.arguments
        if (check.flag_id is None or check.flag_id in kept) and span_is_valid(check.argument, resolve)
    )
    return result.model_copy(
        update={
            "flags": flags,
            "pairs": tuple(pair for pair in result.pairs if pair.flag_id in kept),
            "excluded": _valid(result.excluded, resolve),
            "arguments": arguments,
        }
    )
