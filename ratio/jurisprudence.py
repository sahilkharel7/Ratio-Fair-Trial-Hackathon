"""The jurisprudence linker: which entries of the corpus (config/jurisprudence.yaml) to show next to a
finding, or next to a question for the monitor.

Entries are chosen by the standard a finding concerns, and by its status where an entry is limited to
some statuses. They are never chosen by the case's facts, and never by a model. General Comment
paragraphs come first, then the Committee decisions they cite, in corpus order. A question for the
monitor is not a finding, so it is shown General Comment paragraphs only. Whether an entry applies to
the case is for the reviewing lawyer: the app and the report say so wherever entries are shown.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ratio.config import Jurisprudence, JurisprudenceEntry, SourceRef
from ratio.results import CaseAnalysis
from ratio.schema import Flag, FollowUp


@dataclass(frozen=True)
class Reference:
    entry: JurisprudenceEntry
    source: SourceRef

    @property
    def citation(self) -> str:
        entry = self.entry
        if entry.kind == "general_comment":
            return f"{self.source.symbol}, {entry.pinpoint}"
        return f"{entry.case}, communication No. {entry.communication} ({self.source.symbol}, {entry.pinpoint})"

    @property
    def text(self) -> str:
        """The General Comment's own words, or what it cites the decision for."""
        return self.entry.quote if self.entry.kind == "general_comment" else self.entry.note

    @property
    def shown(self) -> str:
        """As displayed: a General Comment's words in quotation marks (its own become single ones)."""
        if self.entry.kind != "general_comment":
            return self.entry.note
        return "“" + self.entry.quote.replace("“", "‘").replace("”", "’") + "”"


def for_standard(standard_id: str, status: str | None, corpus: Jurisprudence, *, finding: bool = True) -> tuple[Reference, ...]:
    chosen = [
        entry
        for entry in corpus.entries
        if standard_id in entry.standards
        and (finding or entry.kind == "general_comment")
        and (not entry.statuses or status in entry.statuses)
    ]
    chosen.sort(key=lambda entry: entry.kind != "general_comment")  # stable: corpus order within each kind
    return tuple(Reference(entry, corpus.sources[entry.source]) for entry in chosen)


def for_finding(flag: Flag, corpus: Jurisprudence) -> tuple[Reference, ...]:
    return for_standard(flag.standard_id, flag.status, corpus)


def for_follow_up(follow_up: FollowUp, corpus: Jurisprudence) -> tuple[Reference, ...]:
    return for_standard(follow_up.rubric_id, None, corpus, finding=False)


def follow_ups(analysis: CaseAnalysis) -> tuple[FollowUp, ...]:
    return analysis.absence.follow_ups if analysis.absence is not None else ()


def cited_entries(flags: Sequence[Flag], questions: Sequence[FollowUp], corpus: Jurisprudence) -> tuple[str, ...]:
    """Ids of the corpus entries shown anywhere for a case, in corpus order."""
    shown = {ref.entry.id for flag in flags for ref in for_finding(flag, corpus)}
    shown |= {ref.entry.id for question in questions for ref in for_follow_up(question, corpus)}
    return tuple(entry.id for entry in corpus.entries if entry.id in shown)
