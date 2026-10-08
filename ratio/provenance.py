"""Hard rule 2: a flag is shown only if every span it rests on is found, character for
character, in the original document text. Flags that fail are dropped and counted."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

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
