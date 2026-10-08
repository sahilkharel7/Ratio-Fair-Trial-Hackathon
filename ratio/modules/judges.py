"""Module 4: Judicial History Tracker.

Rates per judge come from hand-coded rulings (rulings.yaml: a code from ruling_codes.yaml resting
on an exact quote), so no model output ever feeds statistics about a named person. Each case counts
once per rate. A judge is compared with the baseline of every other judge of the same court for the
same charge type, with no wider pool. An indicator is hidden when the judge or the baseline has
fewer cases than the minimum. "Pattern that warrants review" needs a Newcombe interval for the
difference that excludes zero at 1 - alpha/k, where k is the number of indicators shown for the
judge; otherwise the rates are "not distinguishable from the baseline at this sample size". No
output characterises the judge (hard rule 3), and every one states its sample size.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ratio.config import RateDef, RatioConfig
from ratio.messages import render
from ratio.modules.judge_registry import RegisteredJudge, Registry, build_registry, key_text
from ratio.modules.judge_stats import family_confidence, newcombe, wilson
from ratio.results import (
    CaseAnalysis,
    CaseOutcome,
    CaseSignal,
    DataNote,
    Descriptive,
    Indicator,
    JudgeProfile,
    JudgeReport,
    RateEstimate,
)
from ratio.schema import AliasDecision, CaseRecord, Evidence, Flag, Ruling, stable_id

STANDARD = "judicial_pattern"
ORDERED = ("first", "last")  # rates decided by one ruling of the case, whoever made it

RulingPicker = Callable[[CaseRecord], list[Ruling]]
Owner = Callable[[Ruling], object]


class OrderUnknown(ValueError):
    """Undated rulings make it impossible to tell which ruling of a case came first (or last)."""


def _chronological(rulings: Sequence[Ruling], record: CaseRecord) -> list[Ruling]:
    order = {doc.id: index for index, doc in enumerate(record.documents)}
    return sorted(rulings, key=lambda r: (r.date or dt.date.max, order.get(r.span.doc_id, len(order)), r.span.start))


def _in_document_order(relevant: Sequence[Ruling]) -> list[Ruling] | None:
    """The rulings in the order the document gives them, when that order can stand for time: all in one
    document, with the dated ones in date order. None otherwise."""
    if len({r.span.doc_id for r in relevant}) != 1:
        return None
    ordered = sorted(relevant, key=lambda r: r.span.start)
    dates = [r.date for r in ordered if r.date is not None]
    return ordered if dates == sorted(dates) else None


def _first_or_last(relevant: Sequence[Ruling], rate: RateDef, record: CaseRecord, owner: Owner) -> tuple[bool, Ruling]:
    """The earliest (or latest) ruling decides. Dates order the rulings; an undated one is placed by its
    position in the document when that order agrees with the dates. Otherwise, if an undated ruling could
    change the outcome or the judge, the order is unknown (OrderUnknown) and the case is left out."""
    pick = (lambda rulings: rulings[0]) if rate.combine == "first" else (lambda rulings: rulings[-1])
    undated = [r for r in relevant if r.date is None]
    if not undated:
        decider = pick(_chronological(relevant, record))
        return decider.code in rate.numerator, decider
    in_document = _in_document_order(relevant)
    if in_document is not None:
        decider = pick(in_document)
        return decider.code in rate.numerator, decider
    dated = _chronological([r for r in relevant if r.date], record)
    decider = pick(dated) if dated else pick(_chronological(undated, record))
    if len({(r.code in rate.numerator, owner(r)) for r in (decider, *undated)}) > 1:
        raise OrderUnknown(f"case {record.case_id}: the order of its rulings for '{rate.label}' is unknown")
    return decider.code in rate.numerator, decider


def case_outcome(rulings: Sequence[Ruling], rate: RateDef, record: CaseRecord, owner: Owner = lambda r: r.judge_name) -> tuple[bool, Ruling] | None:
    """Whether one case counts in the rate's numerator, and the ruling that decided it; None when the
    case has no ruling for the rate. Raises OrderUnknown for a 'first' or 'last' rate whose deciding
    ruling depends on where undated rulings fall; ``owner`` says whose ruling each one is."""
    relevant = [r for r in rulings if r.code in rate.denominator]
    if not relevant:
        return None
    if rate.combine in ORDERED:
        return _first_or_last(relevant, rate, record, owner)
    ordered = _chronological(relevant, record)
    hits = [r for r in ordered if r.code in rate.numerator]
    misses = [r for r in ordered if r.code not in rate.numerator]
    if rate.combine == "any":
        return (True, hits[0]) if hits else (False, ordered[0])
    return (False, misses[0]) if misses else (True, ordered[0])


@dataclass(frozen=True)
class _Count:
    """The cases counted for one rate on one side of the comparison."""

    outcomes: tuple[CaseOutcome, ...]
    judges: int

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def k(self) -> int:
        return sum(outcome.counted for outcome in self.outcomes)


@dataclass(frozen=True)
class _Draft:
    rate: RateDef
    charge_type: str
    judge: _Count
    baseline: _Count


def _own(registry: Registry, judge_id: str) -> RulingPicker:
    return lambda record: [r for r in record.rulings if registry.judge_of(record.case_id, r.judge_name) == judge_id]


def _others(registry: Registry, judge_id: str) -> RulingPicker:
    """Rulings of other registered judges; names awaiting confirmation count nowhere."""
    return lambda record: [r for r in record.rulings if registry.judge_of(record.case_id, r.judge_name) not in (None, judge_id)]


def _owner(registry: Registry, record: CaseRecord) -> Owner:
    return lambda ruling: registry.judge_of(record.case_id, ruling.judge_name)


def _decided(records: Sequence[CaseRecord], rate: RateDef, registry: Registry, judge_id: str, mine: bool) -> _Count:
    """A 'first' or 'last' rate: the case's own deciding ruling, over every judge's rulings, counts for the
    judge who made it, so a later judge is never credited with an earlier judge's decision. A case decided
    by a name awaiting confirmation counts nowhere; one whose order is unknown is left out (data notes)."""
    outcomes, judges = [], set()
    for record in records:
        try:
            found = case_outcome(record.rulings, rate, record, _owner(registry, record))
        except OrderUnknown:
            continue
        if found is None:
            continue
        counted, ruling = found
        decider = registry.judge_of(record.case_id, ruling.judge_name)
        if decider is None or (decider == judge_id) != mine:
            continue
        evidence = Evidence(role="ruling", span=ruling.span)
        outcomes.append(CaseOutcome(case_id=record.case_id, title=record.meta.title, counted=counted, ruling=evidence))
        judges.add(decider)
    return _Count(tuple(outcomes), len(judges))


def _count(records: Sequence[CaseRecord], rate: RateDef, pick: RulingPicker, registry: Registry) -> _Count:
    outcomes, judges = [], set()
    for record in records:
        found = case_outcome(pick(record), rate, record)
        if found is None:
            continue
        counted, ruling = found
        evidence = Evidence(role="ruling", span=ruling.span)
        outcomes.append(CaseOutcome(case_id=record.case_id, title=record.meta.title, counted=counted, ruling=evidence))
        judges.add(registry.judge_of(record.case_id, ruling.judge_name))
    return _Count(tuple(outcomes), len(judges))


def _charge_types(records: Sequence[CaseRecord]) -> list[str]:
    seen: dict[str, str] = {}
    for record in sorted(records, key=lambda r: r.case_id):
        seen.setdefault(key_text(record.meta.charge_type), record.meta.charge_type)
    return [seen[key] for key in sorted(seen)]


def _of_charge(records: Sequence[CaseRecord], charge_type: str) -> list[CaseRecord]:
    return [record for record in records if key_text(record.meta.charge_type) == key_text(charge_type)]


def _sides(judge: RegisteredJudge, rate: RateDef, own: list[CaseRecord], place: list[CaseRecord], registry: Registry) -> tuple[_Count, _Count]:
    """The judge's count and the baseline's, both from cases of one court, kind of data and charge type."""
    if rate.combine in ORDERED:
        return _decided(own, rate, registry, judge.judge_id, True), _decided(place, rate, registry, judge.judge_id, False)
    peers = [r for r in place if r.case_id not in judge.case_ids]  # each case on one side of the comparison only
    return _count(own, rate, _own(registry, judge.judge_id), registry), _count(peers, rate, _others(registry, judge.judge_id), registry)


def _drafts(judge: RegisteredJudge, records: Sequence[CaseRecord], registry: Registry, config: RatioConfig) -> list[_Draft]:
    place = [r for r in records if r.meta.synthetic == judge.synthetic and key_text(r.meta.court) == key_text(judge.court)]
    own = [r for r in place if r.case_id in judge.case_ids]
    drafts = []
    for charge_type in _charge_types(own):
        for rate in config.ruling_codes.rates:
            mine, baseline = _sides(judge, rate, _of_charge(own, charge_type), _of_charge(place, charge_type), registry)
            if mine.n:  # a rate with no case decided by this judge is left out, not shown as hidden
                drafts.append(_Draft(rate=rate, charge_type=charge_type, judge=mine, baseline=baseline))
    return drafts


def _estimate(k: int, n: int, confidence: float) -> RateEstimate:
    low, high = wilson(k, n, confidence)
    return RateEstimate(k=k, n=n, rate=k / n, ci_low=low, ci_high=high, confidence=confidence)


def _percent(value: float) -> str:
    return f"{round(100 * value)}%"


def _flag(draft: _Draft, judge: RegisteredJudge, rates: tuple[RateEstimate, RateEstimate], k: int, config: RatioConfig) -> Flag:
    standard = config.standard(STANDARD)
    judge_rate, baseline_rate = rates
    message = render(
        config.messages, "judge_pattern", label=draft.rate.label, judge_rate=_percent(judge_rate.rate), judge_n=judge_rate.n,
        baseline_rate=_percent(baseline_rate.rate), baseline_n=baseline_rate.n, k=k,
    )  # fmt: skip
    return Flag(
        id=stable_id(judge.judge_id, draft.rate.id, key_text(draft.charge_type), "pattern"),
        case_id=None,
        module="judges",
        standard_id=STANDARD,
        standard_label=standard.label,
        status="pattern_warrants_review",
        message=message,
        evidence=tuple(outcome.ruling for outcome in draft.judge.outcomes),
        citation=standard.citation,
        review_status=standard.review_status,
    )


def _indicator(draft: _Draft, judge: RegisteredJudge, k: int, config: RatioConfig) -> tuple[Indicator, Flag | None]:
    settings, messages = config.settings.judges, config.messages
    minimum = settings.min_case_count
    common = {
        "rate_id": draft.rate.id, "label": draft.rate.label, "charge_type": draft.charge_type, "judge_cases": draft.judge.n,
        "baseline_cases": draft.baseline.n, "baseline_judges": draft.baseline.judges,
    }  # fmt: skip
    if draft.judge.n < minimum:
        return Indicator(**common, shown=False, message=render(messages, "judge_hidden", n=draft.judge.n, min=minimum)), None
    if draft.baseline.n < minimum:
        return Indicator(**common, shown=False, message=render(messages, "judge_hidden_baseline", n=draft.baseline.n, min=minimum)), None
    rates = (_estimate(draft.judge.k, draft.judge.n, 1 - settings.alpha), _estimate(draft.baseline.k, draft.baseline.n, 1 - settings.alpha))
    confidence = family_confidence(settings.alpha, k)
    low, high = newcombe(draft.judge.k, draft.judge.n, draft.baseline.k, draft.baseline.n, confidence)
    flag = _flag(draft, judge, rates, k, config) if low > 0 or high < 0 else None
    message = flag.message if flag else render(messages, "judge_not_distinguishable", judge_n=draft.judge.n, baseline_n=draft.baseline.n)
    indicator = Indicator(
        **common, shown=True, message=message, judge=rates[0], baseline=rates[1], difference_ci=(low, high),
        difference_confidence=confidence, pattern=flag is not None, outcomes=draft.judge.outcomes,
        flag_id=None if flag is None else flag.id,
    )  # fmt: skip
    return indicator, flag


def _descriptive(judge: RegisteredJudge, own: Sequence[CaseRecord], registry: Registry, config: RatioConfig) -> tuple[Descriptive, ...]:
    """Sentence lengths and other numeric codes: plain numbers, never compared with a baseline."""
    labels = {code.id: code.label for code in config.ruling_codes.codes}
    minimum = config.settings.judges.min_case_count
    pick, items = _own(registry, judge.judge_id), []
    for charge_type in _charge_types(own):
        for code in config.ruling_codes.descriptive:
            rulings = [r for record in _of_charge(own, charge_type) for r in pick(record) if r.code == code and r.value is not None]
            if not rulings:
                continue
            cases = len({r.case_id for r in rulings})
            common = {"code": code, "label": labels[code], "charge_type": charge_type, "cases": cases}
            if cases < minimum:
                items.append(Descriptive(**common, shown=False, message=render(config.messages, "judge_hidden", n=cases, min=minimum)))
                continue
            items.append(
                Descriptive(
                    **common, shown=True, message=render(config.messages, "judge_descriptive", cases=cases),
                    values=tuple(r.value for r in rulings), evidence=tuple(Evidence(role="ruling", span=r.span) for r in rulings),
                )  # fmt: skip
            )
    return tuple(items)


def _signals(own: Sequence[CaseRecord], analyses: Mapping[str, CaseAnalysis]) -> tuple[CaseSignal, ...]:
    signals = []
    for record in own:
        analysis = analyses.get(record.case_id)
        if analysis is not None:
            score = analysis.reuse.score if analysis.reuse else None
            signals.append(CaseSignal(case_id=record.case_id, title=record.meta.title, reuse_score=score, flags=len(analysis.all_flags())))
    return tuple(signals)


def _profile(
    judge: RegisteredJudge, records: Sequence[CaseRecord], analyses: Mapping[str, CaseAnalysis], registry: Registry, config: RatioConfig
) -> JudgeProfile:
    minimum = config.settings.judges.min_case_count
    drafts = _drafts(judge, records, registry, config)
    k = sum(1 for draft in drafts if draft.judge.n >= minimum and draft.baseline.n >= minimum)
    built = [_indicator(draft, judge, k, config) for draft in drafts]
    own = sorted((r for r in records if r.case_id in judge.case_ids), key=lambda r: r.case_id)
    return JudgeProfile(
        judge_id=judge.judge_id,
        display_name=judge.display_name,
        court=judge.court,
        synthetic=judge.synthetic,
        charge_types=tuple(_charge_types(own)),
        case_ids=judge.case_ids,
        name_variants=judge.variants,
        k_compared=k,
        indicators=tuple(indicator for indicator, _ in built),
        descriptive=_descriptive(judge, own, registry, config),
        case_signals=_signals(own, analyses),
        pending=tuple(c for c in registry.pending if c.candidate_judge_id == judge.judge_id),
        flags=tuple(flag for _, flag in built if flag is not None),
    )


def _ruling_notes(record: CaseRecord, config: RatioConfig) -> list[DataNote]:
    """Coded rulings that cannot be used as written."""
    codes = {code.id: code for code in config.ruling_codes.codes}
    notes = []
    for ruling in record.rulings:
        code = codes.get(ruling.code)
        if code is None:
            text = f"case {record.case_id}: ruling code {ruling.code!r} is not in ruling_codes.yaml, so the ruling was not counted"
        elif code.numeric and ruling.value is None:
            text = f"case {record.case_id}: the {ruling.code} ruling has no value, so it is not shown"
        elif not code.numeric and ruling.value is not None:
            text = f"case {record.case_id}: the {ruling.code} ruling has a value, but the code is not numeric, so the value is not used"
        else:
            continue
        notes.append(DataNote(text=text, synthetic=record.meta.synthetic))
    return notes


def _order_notes(record: CaseRecord, registry: Registry, config: RatioConfig) -> list[DataNote]:
    notes = []
    for rate in (r for r in config.ruling_codes.rates if r.combine in ORDERED):
        try:
            case_outcome(record.rulings, rate, record, _owner(registry, record))
        except OrderUnknown:
            text = (
                f"case {record.case_id}: the order of its rulings for '{rate.label}' is unknown (some are undated and the "
                "document order does not settle it), so the case is left out of that rate"
            )
            notes.append(DataNote(text=text, synthetic=record.meta.synthetic))
    return notes


def _data_notes(records: Sequence[CaseRecord], registry: Registry, config: RatioConfig) -> tuple[DataNote, ...]:
    ordered = sorted(records, key=lambda r: r.case_id)
    return tuple(note for record in ordered for note in _ruling_notes(record, config) + _order_notes(record, registry, config))


def run(records: Sequence[CaseRecord], analyses: Mapping[str, CaseAnalysis], decisions: Sequence[AliasDecision], config: RatioConfig) -> JudgeReport:
    """Every judge named in the records' coded rulings, compared within court and charge type."""
    ids = [record.case_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("each case may appear only once in the judicial history")
    registry = build_registry(records, decisions, config.settings.judges)
    profiles = tuple(_profile(judge, records, analyses, registry, config) for judge in registry.judges)
    return JudgeReport(profiles=profiles, manual_confirmations=registry.pending, notes=registry.notes + _data_notes(records, registry, config))
