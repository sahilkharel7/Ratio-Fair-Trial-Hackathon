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
    notes_missing: tuple[str, ...] = ()  # ids of notes with no hearing date that hold no evidence for this part


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
    unlabelled_notes: int = Field(default=0, ge=0)  # shortlisted notes the model never labelled


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
    passage_containment: float | None = None  # share of the judgment passage's 5-grams found in the whole indictment
    matches_charge_particulars: bool = False
    flag_id: str | None = None


class ArgumentCheck(Frozen):
    """Whether the judgment's reasoning responds to one defence argument from the notes."""

    argument_id: str
    argument: SourceSpan
    addressed: bool
    checked: bool = True  # False when the model's answer could not be used; never flagged then
    responding: tuple[SourceSpan, ...] = ()
    passages_checked: int = Field(ge=0)
    passages_total: int = Field(default=0, ge=0)
    model_note: str | None = None
    flag_id: str | None = None


class ReuseResult(Frozen):
    """``score`` is the share of reasoning characters traceable to the indictment; None when the
    case has no judgment, no indictment, or no reasoning to measure."""

    judgment_doc_id: str | None
    indictment_doc_id: str | None
    indictment_doc_ids: tuple[str, ...] = ()
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    score_note: str | None = None  # why there is no score
    verbatim_chars: int = Field(default=0, ge=0)
    paraphrase_chars: int = Field(default=0, ge=0)
    charge_wording_chars: int = Field(default=0, ge=0)  # restated charge: shown, not scored
    reasoning_chars: int = Field(default=0, ge=0)
    reasoning_passage_ids: tuple[str, ...] = ()
    pairs: tuple[ReusePair, ...] = ()
    excluded: tuple[ExcludedPassage, ...] = ()
    arguments: tuple[ArgumentCheck, ...] = ()
    flags: tuple[Flag, ...] = ()


# --- Module 5: Detention renewals ---------------------------------------------------------


class RenewalPair(Frozen):
    """A grounds passage of a later order that repeats a passage of an earlier one."""

    id: str
    kind: Literal["verbatim", "paraphrase"]
    later: SourceSpan
    earlier: SourceSpan
    later_ranges: tuple[CharRange, ...] = ()  # the repeated 5-gram text (verbatim only)
    earlier_ranges: tuple[CharRange, ...] = ()
    containment: float | None = None
    cosine: float | None = None


class OrderSummary(Frozen):
    """One detention order, in date order. ``share_repeated`` is None for the first order, for an
    undated order and for an order with no grounds to measure."""

    doc_id: str
    title: str
    date: dt.date | None
    until: dt.date | None
    date_span: SourceSpan | None = None
    until_span: SourceSpan | None = None
    days_since_previous: int | None = None
    grounds_passage_ids: tuple[str, ...] = ()
    grounds_chars: int = Field(default=0, ge=0)
    repeated_chars: int = Field(default=0, ge=0)
    share_repeated: float | None = Field(default=None, ge=0, le=1)
    passages_compared: int = Field(default=0, ge=0)  # grounds passages long enough to compare
    passages_new: int = Field(default=0, ge=0)  # of those, the ones no earlier order contains
    pairs: tuple[RenewalPair, ...] = ()
    excluded: tuple[ExcludedPassage, ...] = ()
    flag_id: str | None = None


class OrderGap(Frozen):
    """Whole days between the end date of one order and the date of the next one in the record."""

    id: str
    earlier_doc_id: str
    later_doc_id: str
    until: dt.date
    next_date: dt.date
    days: int = Field(gt=0)
    evidence: tuple[Evidence, ...] = Field(min_length=2)
    flag_id: str | None = None


class RenewalResult(Frozen):
    orders: tuple[OrderSummary, ...] = ()
    gaps: tuple[OrderGap, ...] = ()
    notes: tuple[str, ...] = ()  # what could not be measured, and why
    flags: tuple[Flag, ...] = ()


# --- The State's strongest reply (ratio/steelman.py) -----------------------------------------


class StateArgument(Frozen):
    """One argument the local model made for the State, on a reviewed ground, resting on an exact quote
    of the record. ``argument`` is model text: shown only as unverified, after the block list."""

    ground_id: str
    argument: str = Field(min_length=1)
    span: SourceSpan
    passage_label: str  # the passage id the model was shown, e.g. "E3"


class StateReply(Frozen):
    flag_id: str
    standard_id: str
    arguments: tuple[StateArgument, ...] = ()
    unsupported_grounds: tuple[str, ...] = ()  # grounds offered that no quoted passage supported
    passages_shown: int = Field(default=0, ge=0)
    dropped: tuple[str, ...] = ()  # model arguments that failed a check, and why
    checked: bool = True  # False when the model gave no usable answer
    note: str | None = None


class SteelmanResult(Frozen):
    replies: tuple[StateReply, ...] = ()
    model: str | None = None

    def for_flag(self, flag_id: str) -> StateReply | None:
        return next((reply for reply in self.replies if reply.flag_id == flag_id), None)


# --- Module 4: Judicial History Tracker -----------------------------------------------------


class RateEstimate(Frozen):
    """k of n cases, with a Wilson interval at ``confidence``."""

    k: int = Field(ge=0)
    n: int = Field(gt=0)
    rate: float = Field(ge=0, le=1)
    ci_low: float = Field(ge=0, le=1)
    ci_high: float = Field(ge=0, le=1)
    confidence: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _consistent(self) -> RateEstimate:
        if self.k > self.n or not self.ci_low <= self.rate <= self.ci_high:
            raise ValueError("a rate needs k <= n and an interval that contains it")
        return self


class CaseOutcome(Frozen):
    """One case counted in a judge's rate, and the coded ruling that decided its outcome."""

    case_id: str
    title: str
    counted: bool  # in the numerator
    ruling: Evidence


class Indicator(Frozen):
    """One rate for one judge and charge type. Below the minimum case count it carries no rate at all,
    so nothing can display one; ``pattern`` needs a difference interval that excludes zero."""

    rate_id: str
    label: str
    charge_type: str
    judge_cases: int = Field(ge=0)
    baseline_cases: int = Field(ge=0)
    baseline_judges: int = Field(ge=0)
    shown: bool
    message: str
    judge: RateEstimate | None = None
    baseline: RateEstimate | None = None
    difference_ci: tuple[float, float] | None = None
    difference_confidence: float | None = Field(default=None, gt=0, lt=1)
    pattern: bool = False
    outcomes: tuple[CaseOutcome, ...] = ()
    flag_id: str | None = None

    @model_validator(mode="after")
    def _hidden_means_no_numbers(self) -> Indicator:
        numbers = (self.judge, self.baseline, self.difference_ci, self.difference_confidence)
        if self.shown and any(value is None for value in numbers):
            raise ValueError("a shown indicator needs both rates and the difference interval")
        if not self.shown and (any(value is not None for value in numbers) or self.outcomes or self.pattern):
            raise ValueError("a hidden indicator must not carry rates, cases or a pattern (minimum case count)")
        if self.pattern and (self.flag_id is None or self.difference_ci[0] <= 0 <= self.difference_ci[1]):
            raise ValueError("a pattern needs a flag and a difference interval that excludes zero")
        return self


class Descriptive(Frozen):
    """Coded values shown as plain numbers and never compared with a baseline (sentence lengths)."""

    code: str
    label: str
    charge_type: str
    cases: int = Field(ge=0)
    shown: bool
    message: str
    values: tuple[float, ...] = ()
    evidence: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def _hidden_means_no_values(self) -> Descriptive:
        if not self.shown and (self.values or self.evidence):
            raise ValueError("hidden values must not be carried")
        return self


class CaseSignal(Frozen):
    """Per-case outputs of the other modules, linked from the judge page (not attributed to the judge)."""

    case_id: str
    title: str
    reuse_score: float | None = None
    flags: int = Field(default=0, ge=0)


class AliasCandidate(Frozen):
    """A name that may belong to a registered judge. Its rulings count nowhere, neither in that judge's
    rates nor in any baseline, until a person decides in alias_decisions.yaml."""

    case_id: str
    raw_name: str
    court: str
    synthetic: bool
    candidate_judge_id: str | None
    candidate_name: str | None
    score: float = Field(ge=0, le=100)
    reason: str


class DataNote(Frozen):
    """A coded ruling, a decision or a case that could not be used, and why. ``synthetic`` is the kind of
    data it concerns, so the judge page never shows one kind under the other's label; None only for a
    note that names no case content."""

    text: str
    synthetic: bool | None = None


class JudgeProfile(Frozen):
    judge_id: str
    display_name: str
    court: str
    synthetic: bool
    charge_types: tuple[str, ...]
    case_ids: tuple[str, ...]
    name_variants: tuple[str, ...]
    k_compared: int = Field(default=0, ge=0)  # indicators shown, over which alpha is split
    indicators: tuple[Indicator, ...] = ()
    descriptive: tuple[Descriptive, ...] = ()
    case_signals: tuple[CaseSignal, ...] = ()
    pending: tuple[AliasCandidate, ...] = ()  # names that may be this judge's, awaiting a decision
    flags: tuple[Flag, ...] = ()


class JudgeReport(Frozen):
    profiles: tuple[JudgeProfile, ...] = ()
    manual_confirmations: tuple[AliasCandidate, ...] = ()
    notes: tuple[DataNote, ...] = ()
    dropped_flags: int = Field(default=0, ge=0)
    dropped_reasons: tuple[str, ...] = ()

    def profile(self, judge_id: str) -> JudgeProfile:
        for profile in self.profiles:
            if profile.judge_id == judge_id:
                return profile
        raise KeyError(judge_id)


# --- One analysed case ----------------------------------------------------------------------


class CaseAnalysis(Frozen):
    case_id: str
    absence: AbsenceResult | None = None
    clock: ClockResult | None = None
    reuse: ReuseResult | None = None
    renewal: RenewalResult | None = None
    steelman: SteelmanResult | None = None
    dropped_flags: int = Field(default=0, ge=0)
    dropped_reasons: tuple[str, ...] = ()
    llm_model: str | None = None
    llm_mode: str | None = None

    def all_flags(self) -> tuple[Flag, ...]:
        results = (self.absence, self.clock, self.renewal, self.reuse)
        return tuple(flag for result in results if result is not None for flag in result.flags)
