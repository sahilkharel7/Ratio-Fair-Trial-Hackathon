"""Lawyers' corrections of Ratio's findings, and what they measure.

A reviewer accepts a finding, rewrites its wording, or rejects it with a reason, and can record an
issue Ratio did not flag. Decisions are append-only (the store keeps every one, so the history can be
audited) and keyed on the finding's stable id, so they survive a re-analysis that produces the same
finding. Each decision keeps a snapshot of the finding it was made on: a decision whose finding is no
longer produced is reported as stale, never dropped without a word.

The measures below are prompts for improving Ratio, computed on the cases reviewed so far, not claims
about its accuracy in general. Nothing here calls a model; the reviewer's wording is theirs and is
shown under their decision, never as Ratio's.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from ratio.results import CaseAnalysis, JudgeReport
from ratio.schema import Flag, Frozen, ModuleName

Decision = Literal["accepted", "edited", "rejected", "reopened"]
MODULES: tuple[ModuleName, ...] = ("absence", "clock", "renewal", "reuse", "judges")


class Review(Frozen):
    """One reviewer decision on one finding. ``reopened`` withdraws the previous decision."""

    case_id: str = Field(min_length=1)  # the case the finding was reviewed in (a judge pattern has no case of its own)
    flag_id: str = Field(min_length=1)
    decision: Decision
    note: str = ""
    edited_message: str | None = None
    reviewer: str | None = None
    created_at: str = ""  # set by the store
    flag: Flag  # the finding as it was when the decision was made

    @model_validator(mode="after")
    def _complete(self) -> Review:
        if self.flag.id != self.flag_id:
            raise ValueError("the snapshot must be of the finding the decision is about")
        if self.decision in ("rejected", "edited") and not self.note.strip():
            raise ValueError(f"a decision to {'reject' if self.decision == 'rejected' else 'edit'} a finding needs a reason")
        if self.decision == "edited":
            if not (self.edited_message or "").strip():
                raise ValueError("an edited finding needs the reviewer's wording")
            if self.edited_message.strip() == self.flag.message.strip():
                raise ValueError("the edited wording is the same as Ratio's; accept the finding instead")
        elif self.edited_message is not None:
            raise ValueError("only an edited finding carries the reviewer's wording")
        return self


class MissedIssue(Frozen):
    """An issue the reviewer found in the record that Ratio did not flag."""

    id: int | None = None  # set by the store
    case_id: str = Field(min_length=1)
    module: ModuleName
    standard: str = Field(min_length=1)
    note: str = Field(min_length=1)
    reviewer: str | None = None
    created_at: str = ""

    @model_validator(mode="after")
    def _not_blank(self) -> MissedIssue:
        if not self.standard.strip() or not self.note.strip():
            raise ValueError("a missed issue needs the standard it concerns and a description")
        return self


def case_findings(analysis: CaseAnalysis, judges: JudgeReport | None) -> tuple[Flag, ...]:
    """Every finding shown for a case: its own, then the patterns of its presiding judge."""
    profile = next((p for p in judges.profiles if analysis.case_id in p.case_ids), None) if judges else None
    return analysis.all_flags() + (profile.flags if profile else ())


def current(reviews: Iterable[Review]) -> dict[str, Review]:
    """The decision in force for each finding: the latest one, unless it was reopened. ``reviews`` are
    in the order they were made (the store returns them so)."""
    latest: dict[str, Review] = {}
    for review in reviews:
        if review.decision == "reopened":
            latest.pop(review.flag_id, None)
        else:
            latest[review.flag_id] = review
    return latest


def message(flag: Flag, review: Review | None) -> str:
    """The wording to show: the reviewer's when they edited the finding, else Ratio's."""
    return review.edited_message if review is not None and review.decision == "edited" else flag.message


class ModuleFeedback(Frozen):
    module: ModuleName
    findings: int = Field(ge=0)
    reviewed: int = Field(ge=0)
    accepted: int = Field(ge=0)
    edited: int = Field(ge=0)
    rejected: int = Field(ge=0)
    missed: int = Field(ge=0)

    @property
    def kept_share(self) -> float | None:
        """Share of the reviewed findings the reviewer kept (accepted or reworded)."""
        return (self.accepted + self.edited) / self.reviewed if self.reviewed else None


class FeedbackSummary(Frozen):
    cases: int = Field(ge=0)
    modules: tuple[ModuleFeedback, ...]
    stale: tuple[Review, ...] = ()  # decisions in force on findings the current analyses no longer produce
    rejections: tuple[Review, ...] = ()

    @property
    def reviewed(self) -> int:
        return sum(m.reviewed for m in self.modules)

    @property
    def findings(self) -> int:
        return sum(m.findings for m in self.modules)


def summarise(
    findings: Mapping[str, Sequence[Flag]], reviews: Sequence[Review], missed: Sequence[MissedIssue]
) -> FeedbackSummary:
    """Per module: findings shown, decisions in force on them, and missed issues, over the cases in
    ``findings`` (case id -> the findings its current analysis shows)."""
    rows: list[ModuleFeedback] = []
    stale: list[Review] = []
    in_force: dict[str, list[Review]] = {module: [] for module in MODULES}
    for case_id, flags in findings.items():
        shown = {flag.id for flag in flags}
        for review in current(r for r in reviews if r.case_id == case_id).values():
            (in_force[review.flag.module] if review.flag_id in shown else stale).append(review)
    for module in MODULES:
        decided = in_force[module]
        count = lambda decision: sum(r.decision == decision for r in decided)  # noqa: E731
        rows.append(
            ModuleFeedback(
                module=module,
                findings=sum(flag.module == module for flags in findings.values() for flag in flags),
                reviewed=len(decided),
                accepted=count("accepted"),
                edited=count("edited"),
                rejected=count("rejected"),
                missed=sum(issue.module == module and issue.case_id in findings for issue in missed),
            )
        )
    rejections = tuple(r for module in MODULES for r in in_force[module] if r.decision == "rejected")
    return FeedbackSummary(cases=len(findings), modules=tuple(rows), stale=tuple(stale), rejections=rejections)


class Regression(Frozen):
    """A fresh analysis compared with the decisions in force: what a change to Ratio did to findings a
    lawyer already judged."""

    kept_still_shown: tuple[str, ...] = ()  # accepted or edited, and still produced
    kept_lost: tuple[str, ...] = ()  # accepted or edited, and no longer produced: a regression
    rejected_still_shown: tuple[str, ...] = ()  # rejected, and still produced: not yet fixed
    rejected_gone: tuple[str, ...] = ()  # rejected, and no longer produced: fixed

    @property
    def ok(self) -> bool:
        return not self.kept_lost


def regression(flags: Sequence[Flag], reviews: Sequence[Review]) -> Regression:
    shown = {flag.id for flag in flags}
    buckets: dict[str, list[str]] = {"kept_still_shown": [], "kept_lost": [], "rejected_still_shown": [], "rejected_gone": []}
    for review in current(reviews).values():
        kept = "kept" if review.decision in ("accepted", "edited") else "rejected"
        state = ("still_shown" if review.flag_id in shown else "lost" if kept == "kept" else "gone")
        buckets[f"{kept}_{state}"].append(review.flag_id)
    return Regression(**{key: tuple(ids) for key, ids in buckets.items()})
