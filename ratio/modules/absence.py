"""Module 1: Absence Detector (ICCPR Art. 14(3)(a)-(g)).

For each rubric guarantee: shortlist the observations most similar to it (embeddings), have the
model label each one supports / contradicts / unrelated while citing a rubric indicator, then
apply the status rules in code:
  any part contradicted                          -> evidence of violation (a flag)
  every required part supported (at every
  monitored hearing for per-hearing parts)       -> evidence of compliance (a flag)
  otherwise                                      -> no evidence: a follow-up question, not a finding
Context facts (e.g. the defendant's language) are shown with the follow-up and never count.
"""

from __future__ import annotations

import datetime as dt
import re

import numpy as np

from ratio.config import RatioConfig, RubricItem, RubricPart
from ratio.context import AnalysisContext
from ratio.embeddings import cosine_matrix
from ratio.messages import filter_model_note, render
from ratio.modules.absence_prompts import LABEL_SYSTEM, LabelReply, label_user_prompt
from ratio.results import AbsenceResult, GuaranteeAssessment, GuaranteeStatus, LabeledObservation, PartAssessment
from ratio.schema import CaseRecord, Evidence, Flag, FollowUp, Observation, stable_id

MAX_CONTEXT_NOTES = 1
_OBSERVATION_ID = re.compile(r"\bO\d+\b")


def monitored_hearings(record: CaseRecord) -> tuple[dt.date, ...]:
    return tuple(sorted({doc.date for doc in record.documents if doc.type == "monitoring_note" and doc.date}))


def _citation(config: RatioConfig, item: RubricItem) -> str:
    source = config.rubric.sources[item.citation.instrument]
    return f"{item.provision}; {source.symbol}, para. {item.citation.paras}"


def _top(indices, similarity: np.ndarray, k: int, floor: float) -> set[int]:
    ranked = sorted(indices, key=lambda i: (-float(similarity[i]), i))
    return {i for i in ranked[:k] if similarity[i] >= floor}


def _shortlist(
    item: RubricItem, observations: tuple[Observation, ...], vectors: np.ndarray, hearings: tuple[dt.date, ...], ctx: AnalysisContext
) -> list[Observation]:
    """Per part: notes with a part keyword, plus the most similar notes overall (or per hearing)."""
    settings = ctx.config.settings.absence
    scores: dict[int, float] = {}
    for part in item.parts:
        queries = [f"{item.name}: {part.label}"] + [i.text for i in part.compliance + part.violation]
        similarity = cosine_matrix(vectors, ctx.embedder.encode(queries)).max(axis=1)
        chosen = {i for i, obs in enumerate(observations) if any(k in obs.text.lower() for k in part.keywords)}
        if part.scope == "per_hearing" and hearings:
            for hearing in hearings:
                same_hearing = [i for i, obs in enumerate(observations) if obs.hearing_date == hearing]
                chosen |= _top(same_hearing, similarity, settings.per_hearing_top_k, settings.shortlist_min_similarity)
        else:
            chosen |= _top(range(len(observations)), similarity, settings.shortlist_top_k, settings.shortlist_min_similarity)
        for i in chosen:
            scores[i] = max(scores.get(i, -1.0), float(similarity[i]))
    kept = sorted(scores, key=lambda i: (-scores[i], i))[: settings.max_shortlist]
    return [observations[i] for i in sorted(kept)]  # document order keeps prompts deterministic


def _grounded(part: RubricPart, obs: Observation, previous: dict[str, str]) -> bool:
    """A label counts only if the note (or the sentence before it) mentions one of the part's keywords."""
    if not part.keywords:
        return True
    text = f"{previous.get(obs.id, '')} {obs.text}".lower()
    return any(keyword in text for keyword in part.keywords)


def _previous_sentences(observations: tuple[Observation, ...]) -> dict[str, str]:
    """The sentence before each observation in the same note, to resolve words like 'the request'."""
    previous: dict[str, str] = {}
    for before, current in zip(observations, observations[1:]):
        if before.span.doc_id == current.span.doc_id:
            previous[current.id] = before.text
    return previous


def _label(
    item: RubricItem, shortlist: list[Observation], previous: dict[str, str], ctx: AnalysisContext
) -> tuple[LabeledObservation, ...]:
    numbered = [(f"O{n}", obs) for n, obs in enumerate(shortlist, start=1)]
    prompt = label_user_prompt(item, [(obs_id, obs, previous.get(obs.id)) for obs_id, obs in numbered])
    reply = ctx.llm.complete_json(system=LABEL_SYSTEM, user=prompt, schema=LabelReply, purpose="labels")
    by_id = dict(numbered)
    licensed = {i.id: (part, "supports") for part in item.parts for i in part.compliance}
    licensed |= {i.id: (part, "contradicts") for part in item.parts for i in part.violation}
    block_list = ctx.config.messages.block_list
    labels: dict[tuple[str, str], LabeledObservation] = {}
    for entry in reply.labels:
        found = _OBSERVATION_ID.search(entry.observation)  # models sometimes copy the whole "O3 (date): text" line
        obs = by_id.get(found.group(0)) if found else None
        part_and_label = licensed.get(entry.indicator_id.strip())
        if obs is None or entry.label == "unrelated" or part_and_label is None or part_and_label[1] != entry.label:
            continue  # a label counts only with a matching indicator of the right kind
        part, label = part_and_label
        if not _grounded(part, obs, previous):
            continue  # the note never mentions what this part is about
        labels.setdefault(
            (obs.id, part.id),
            LabeledObservation(
                observation_id=obs.id,
                part_id=part.id,
                indicator_id=entry.indicator_id.strip(),
                label=label,
                span=obs.span,
                hearing_date=obs.hearing_date,
                model_note=filter_model_note(entry.note, block_list),
            ),
        )
    return tuple(labels.values())


def _mentions_context(item: RubricItem, obs: Observation) -> bool:
    """A context note must mention what the context is about (e.g. a language), not only resemble it."""
    keywords = [keyword for indicator in item.context for keyword in indicator.keywords]
    return not keywords or any(keyword in obs.text.lower() for keyword in keywords)


def _context(item: RubricItem, observations, vectors, labels, ctx: AnalysisContext) -> tuple[Evidence, ...]:
    if not item.context or not observations:
        return ()
    labelled = {label.observation_id for label in labels}
    similarity = cosine_matrix(vectors, ctx.embedder.encode([i.text for i in item.context])).max(axis=1)
    threshold = ctx.config.settings.absence.context_min_similarity
    ranked = sorted(range(len(observations)), key=lambda i: -float(similarity[i]))
    chosen = [
        i
        for i in ranked
        if similarity[i] >= threshold and observations[i].id not in labelled and _mentions_context(item, observations[i])
    ]
    return tuple(Evidence(role="context", span=observations[i].span) for i in chosen[:MAX_CONTEXT_NOTES])


def _part_status(part: RubricPart, labels, hearings: tuple[dt.date, ...]) -> PartAssessment:
    mine = [label for label in labels if label.part_id == part.id]
    if any(label.label == "contradicts" for label in mine):
        return PartAssessment(part_id=part.id, label=part.label, required=part.required, status="evidence_of_violation")
    supporting = [label for label in mine if label.label == "supports"]
    if not supporting:
        return PartAssessment(
            part_id=part.id, label=part.label, required=part.required, status="no_evidence",
            hearings_missing=hearings if part.scope == "per_hearing" else (),
        )  # fmt: skip
    if part.scope == "case" or not hearings:
        return PartAssessment(part_id=part.id, label=part.label, required=part.required, status="evidence_of_compliance")
    all_hearings = {i.id for i in part.compliance if i.covers_all_hearings}
    covered = set(hearings) if any(label.indicator_id in all_hearings for label in supporting) else {
        label.hearing_date for label in supporting
    }
    missing = tuple(h for h in hearings if h not in covered)
    status: GuaranteeStatus = "no_evidence" if missing else "evidence_of_compliance"
    return PartAssessment(part_id=part.id, label=part.label, required=part.required, status=status, hearings_missing=missing)


def _overall(parts: tuple[PartAssessment, ...]) -> GuaranteeStatus:
    if any(part.status == "evidence_of_violation" for part in parts):
        return "evidence_of_violation"
    if all(part.status == "evidence_of_compliance" for part in parts if part.required):
        return "evidence_of_compliance"
    return "no_evidence"


def _follow_up(record: CaseRecord, item: RubricItem, parts, context) -> FollowUp:
    uncovered = [part.label for part in parts if part.required and part.status == "no_evidence" and not part.hearings_missing]
    uncovered += [f"{part.label}: hearing of {h.isoformat()}" for part in parts if part.required for h in part.hearings_missing]
    return FollowUp(
        id=stable_id(record.case_id, "absence", item.id, "follow_up"),
        case_id=record.case_id,
        rubric_id=item.id,
        question=item.follow_up,
        uncovered=tuple(uncovered),
        context=context,
    )


def _flag(record: CaseRecord, item: RubricItem, status: GuaranteeStatus, labels, ctx: AnalysisContext) -> Flag:
    wanted = "contradicts" if status == "evidence_of_violation" else "supports"
    chosen = [label for label in labels if label.label == wanted]
    role = "contradicting" if wanted == "contradicts" else "supporting"
    template = "absence_violation" if wanted == "contradicts" else "absence_compliance"
    return Flag(
        id=stable_id(record.case_id, "absence", item.id, status),
        case_id=record.case_id,
        module="absence",
        standard_id=item.id,
        standard_label=f"{item.provision}: {item.name}",
        status=status,
        message=render(ctx.config.messages, template, provision=item.provision, name=item.name),
        evidence=tuple(Evidence(role=role, span=label.span) for label in chosen),
        citation=_citation(ctx.config, item),
        review_status=item.review_status,
        model_note=next((label.model_note for label in chosen if label.model_note), None),
    )


def _assess(record, item, labels, context, hearings, ctx) -> tuple[GuaranteeAssessment, Flag | None, FollowUp | None]:
    parts = tuple(_part_status(part, labels, hearings) for part in item.parts)
    status = _overall(parts)
    flag = _flag(record, item, status, labels, ctx) if status != "no_evidence" else None
    follow_up = _follow_up(record, item, parts, context) if status == "no_evidence" else None
    assessment = GuaranteeAssessment(
        rubric_id=item.id,
        name=item.name,
        provision=item.provision,
        citation=_citation(ctx.config, item),
        review_status=item.review_status,
        status=status,
        parts=parts,
        supporting=tuple(label for label in labels if label.label == "supports"),
        contradicting=tuple(label for label in labels if label.label == "contradicts"),
        context=context,
        follow_up=follow_up,
        flag_id=flag.id if flag else None,
    )
    return assessment, flag, follow_up


def run(record: CaseRecord, ctx: AnalysisContext) -> AbsenceResult:
    observations = record.observations
    hearings = monitored_hearings(record)
    vectors = ctx.embedder.encode([obs.text for obs in observations]) if observations else np.zeros((0, 1), dtype=np.float32)
    previous = _previous_sentences(observations)
    assessments, flags, follow_ups = [], [], []
    for item in ctx.config.rubric.items:
        shortlist = _shortlist(item, observations, vectors, hearings, ctx) if observations else []
        labels = _label(item, shortlist, previous, ctx) if shortlist else ()
        context = _context(item, observations, vectors, labels, ctx)
        assessment, flag, follow_up = _assess(record, item, labels, context, hearings, ctx)
        assessments.append(assessment)
        flags.extend([flag] if flag else [])
        follow_ups.extend([follow_up] if follow_up else [])
    return AbsenceResult(assessments=tuple(assessments), flags=tuple(flags), follow_ups=tuple(follow_ups))
