"""Module outputs. Each module returns one of these; the app and the eval only read them."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import Field, model_validator

from ratio.schema import (
    DatePrecision,
    EventType,
    Evidence,
    Flag,
    FollowUp,
    Frozen,
    ReviewStatus,
    SourceSpan,
)

GuaranteeStatus = Literal["evidence_of_compliance", "evidence_of_violation", "no_evidence"]
ObservationLabel = Literal["supports", "contradicts"]
TimelineState = Literal["ok", "needs_review", "conflict", "model_date_only", "undated"]
IntervalStatus = Literal[
    "exceeds_benchmark",  # red: confirmed benchmark, both dates parsed in code, lower bound above threshold
    "may_exceed",  # amber: the bounds straddle the threshold
    "needs_review",  # amber: a date needs review or conflicts
    "within_benchmark",
    "measured",  # benchmark has no confirmed threshold: shown, never coloured
    "cannot_compute",  # an event or a code-parsed date is missing
]
ExclusionReason = Literal[
    "statute_quote",
    "charge_recital",
    "party_position",
    "non_reasoning_section",
    "header_or_signature",
]


# --- Module 1: Absence Detector -------------------------------------------------------------


class LabeledObservation(Frozen):
    observation_id: str
    part_id: str
    indicator_id: str
    label: ObservationLabel
    span: SourceSpan
    hearing_date: dt.date | None = None
    model_note: str | None = None


class PartAssessment(Frozen):
    part_id: str
    label: str
    required: bool
    status: GuaranteeStatus
    hearings_missing: tuple[dt.date, ...] = ()


class GuaranteeAssessment(Frozen):
    rubric_id: str
    name: str
    provision: str
    citation: str
    review_status: ReviewStatus
    status: GuaranteeStatus
    parts: tuple[PartAssessment, ...]
    supporting: tuple[LabeledObservation, ...] = ()
    contradicting: tuple[LabeledObservation, ...] = ()
    context: tuple[Evidence, ...] = ()
    follow_up: FollowUp | None = None
    flag_id: str | None = None


class AbsenceResult(Frozen):
    assessments: tuple[GuaranteeAssessment, ...]
    flags: tuple[Flag, ...] = ()
    follow_ups: tuple[FollowUp, ...] = ()


# --- Module 2: Procedural Clock -------------------------------------------------------------


class TimelineEvent(Frozen):
    """One real-world event, merged from every mention of it across the case documents."""

    id: str
    type: EventType
    date: dt.datetime | None
    precision: DatePrecision
    state: TimelineState
    mentions: tuple[Evidence, ...] = Field(min_length=1)
    candidate_dates: tuple[str, ...] = ()
    review_reasons: tuple[str, ...] = ()


class Interval(Frozen):
    """A measured benchmark interval. Red (exceeds_benchmark) only for a confirmed threshold."""

    id: str
    benchmark_id: str
    benchmark_name: str
    from_event_id: str | None  # the TimelineEvent the interval starts from
    to_event_id: str | None  # the TimelineEvent it ends at
    min_hours: float | None
    max_hours: float | None
    threshold_hours: float | None
    status: IntervalStatus
    citation: str
    review_status: ReviewStatus
    evidence: tuple[Evidence, ...] = ()
    note: str = ""
    flag_id: str | None = None

    @model_validator(mode="after")
    def _red_only_when_confirmed(self) -> Interval:
        if self.status == "exceeds_benchmark" and (self.review_status != "confirmed" or self.threshold_hours is None):
            raise ValueError("only a confirmed benchmark with a threshold can be exceeded (hard rule 5)")
        return self


class ClockResult(Frozen):
    timeline: tuple[TimelineEvent, ...]
    intervals: tuple[Interval, ...]
    flags: tuple[Flag, ...] = ()


# --- Module 3: Reasoning Reuse Detector -----------------------------------------------------


class CharRange(Frozen):
    """Absolute character offsets [start, end) in a document's text."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)


class ExcludedPassage(Frozen):
    passage_id: str
    span: SourceSpan
    reason: ExclusionReason


class ReusePair(Frozen):
    id: str
    kind: Literal["verbatim", "paraphrase"]
    judgment: SourceSpan
    indictment: SourceSpan
    jaccard: float | None = None
    containment: float | None = None
    cosine: float | None = None
    judgment_ranges: tuple[CharRange, ...] = ()
    indictment_ranges: tuple[CharRange, ...] = ()
    matches_charge_particulars: bool = False
    flag_id: str | None = None


class ArgumentCheck(Frozen):
    """Whether the judgment's reasoning responds to one defence argument from the notes."""

    argument_id: str
    argument: SourceSpan
    addressed: bool
    responding: tuple[SourceSpan, ...] = ()
    passages_checked: int = Field(ge=1)
    model_note: str | None = None
    flag_id: str | None = None


class ReuseResult(Frozen):
    """``score`` is the share of reasoning characters traceable to the indictment; None when the
    case has no judgment, no indictment, or no reasoning to measure."""

    judgment_doc_id: str | None
    indictment_doc_id: str | None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    verbatim_chars: int = Field(default=0, ge=0)
    paraphrase_chars: int = Field(default=0, ge=0)
    reasoning_chars: int = Field(default=0, ge=0)
    reasoning_passage_ids: tuple[str, ...] = ()
    pairs: tuple[ReusePair, ...] = ()
    excluded: tuple[ExcludedPassage, ...] = ()
    arguments: tuple[ArgumentCheck, ...] = ()
    flags: tuple[Flag, ...] = ()


# --- Module 4: Judicial History Tracker -----------------------------------------------------


class RateEstimate(Frozen):
    k: int = Field(ge=0)
    n: int = Field(ge=0)
    rate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class Indicator(Frozen):
    rate_id: str
    label: str
    charge_type: str
    judge: RateEstimate
    baseline: RateEstimate
    shown: bool
    hidden_reason: str | None = None
    pattern: bool = False
    difference_ci: tuple[float, float] | None = None
    k_compared: int = Field(default=0, ge=0)
    evidence: tuple[Evidence, ...] = ()
    flag_id: str | None = None


class CaseSignal(Frozen):
    """Per-case outputs of the other modules, linked from the judge page (not attributed to the judge)."""

    case_id: str
    title: str
    reuse_score: float | None = None
    flag_ids: tuple[str, ...] = ()


class AliasCandidate(Frozen):
    raw_name: str
    case_id: str
    court: str
    candidate_judge_id: str | None
    score: float
    reason: str


class JudgeProfile(Frozen):
    judge_id: str
    display_name: str
    court: str
    charge_types: tuple[str, ...]
    case_ids: tuple[str, ...]
    name_variants: tuple[str, ...]
    indicators: tuple[Indicator, ...] = ()
    case_signals: tuple[CaseSignal, ...] = ()
    flags: tuple[Flag, ...] = ()


class JudgeReport(Frozen):
    profiles: tuple[JudgeProfile, ...] = ()
    manual_confirmations: tuple[AliasCandidate, ...] = ()


# --- One analysed case ----------------------------------------------------------------------


class CaseAnalysis(Frozen):
    case_id: str
    absence: AbsenceResult | None = None
    clock: ClockResult | None = None
    reuse: ReuseResult | None = None
    dropped_flags: int = Field(default=0, ge=0)
    dropped_reasons: tuple[str, ...] = ()
    llm_model: str | None = None
    llm_mode: str | None = None

    def all_flags(self) -> tuple[Flag, ...]:
        results = (self.absence, self.clock, self.reuse)
        return tuple(flag for result in results if result is not None for flag in result.flags)
