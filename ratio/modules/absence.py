"""Module 1: Absence Detector (ICCPR Art. 14(3)(a)-(g)).

For each rubric guarantee: shortlist the observations most similar to it (embeddings), have the
model label each one supports / contradicts / unrelated while citing a rubric indicator, then
apply the status rules in code:
  any part contradicted                          -> evidence of violation (a flag)
  every required part supported (at every
  monitored hearing for per-hearing parts)       -> evidence of compliance (a flag)
  otherwise                                      -> no evidence: a follow-up question, not a finding
Context facts (e.g. the defendant's language) are shown with the follow-up and never count.

Hearings are keyed by the note's hearing date, or by the note itself when it has none, so an
undated note is never silently treated as covered. The shortlist is labelled in batches that fit
the model's context window; notes the model skips are asked about once more, and a guarantee is
never marked compliant while some of its notes stay unlabelled.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence

import numpy as np

from ratio.config import RatioConfig, RubricIndicator, RubricItem, RubricPart
from ratio.context import AnalysisContext, estimated_tokens, system_with_schema
from ratio.embeddings import cosine_matrix
from ratio.messages import filter_model_note, render
from ratio.modules.absence_prompts import LABEL_SYSTEM, LabelReply, ObservationLabel, label_user_prompt
from ratio.results import AbsenceResult, GuaranteeAssessment, GuaranteeStatus, LabeledObservation, PartAssessment
from ratio.schema import CaseRecord, Evidence, Flag, FollowUp, Observation, stable_id

MAX_CONTEXT_NOTES = 1
LABELS_PER_BATCH = 30  # a reply with 30 labels fits the labels token budget with room to spare
ASK_ROUNDS = 2  # the model is asked once more about notes it left out of its reply
_OBSERVATION_ID = re.compile(r"\bO\d+\b")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")

HearingKey = dt.date | str  # a note's hearing date, or the note's document id when it has none


def monitored_hearings(record: CaseRecord) -> tuple[dt.date, ...]:
    return tuple(sorted({doc.date for doc in record.documents if doc.type == "monitoring_note" and doc.date}))


def hearing_key(obs: Observation | LabeledObservation) -> HearingKey:
    return obs.hearing_date or obs.span.doc_id


def hearing_keys(observations: Sequence[Observation]) -> tuple[HearingKey, ...]:
    dated = sorted({obs.hearing_date for obs in observations if obs.hearing_date})
    undated = sorted({obs.span.doc_id for obs in observations if not obs.hearing_date})
    return (*dated, *undated)


def _citation(config: RatioConfig, item: RubricItem) -> str:
    source = config.rubric.sources[item.citation.instrument]
    return f"{item.provision}; {source.symbol}, para. {item.citation.paras}"


def _top(indices, similarity: np.ndarray, k: int, floor: float) -> list[int]:
    ranked = sorted(indices, key=lambda i: (-float(similarity[i]), i))
    return [i for i in ranked[:k] if similarity[i] >= floor]


def _shortlist(
    item: RubricItem, observations: tuple[Observation, ...], vectors: np.ndarray, keys: tuple[HearingKey, ...], ctx: AnalysisContext
) -> list[Observation]:
    """Per part: notes with a part keyword, plus the most similar notes overall (or per hearing)."""
    settings = ctx.config.settings.absence
    scores: dict[int, float] = {}
    floor: set[int] = set()  # each part's best note at each hearing: kept even above the cap
    for part in item.parts:
        queries = [f"{item.name}: {part.label}"] + [i.text for i in part.compliance + part.violation]
        similarity = cosine_matrix(vectors, ctx.embedder.encode(queries)).max(axis=1)
        keyword_hits = {i for i, obs in enumerate(observations) if any(k in obs.text.lower() for k in part.keywords)}
        chosen = set(keyword_hits)
        if part.scope == "per_hearing":
            for key in keys:
                same = [i for i, obs in enumerate(observations) if hearing_key(obs) == key]
                chosen |= set(_top(same, similarity, settings.per_hearing_top_k, settings.shortlist_min_similarity))
        else:
            chosen |= set(_top(range(len(observations)), similarity, settings.shortlist_top_k, settings.shortlist_min_similarity))
        groups = keys if part.scope == "per_hearing" else (None,)
        for key in groups:
            group = [i for i in chosen if key is None or hearing_key(observations[i]) == key]
            if group:  # a keyword note first, then the most similar note
                floor.add(min(group, key=lambda i: (i not in keyword_hits, -float(similarity[i]), i)))
        for i in chosen:
            scores[i] = max(scores.get(i, -1.0), float(similarity[i]))
    rest = sorted((i for i in scores if i not in floor), key=lambda i: (-scores[i], i))
    kept = floor | set(rest[: max(0, settings.max_shortlist - len(floor))])
    return [observations[i] for i in sorted(kept)]  # document order keeps prompts deterministic


def _previous_sentences(observations: tuple[Observation, ...]) -> dict[str, str]:
    """The sentence before each observation in the same note, shown to the model to resolve 'the request'."""
    previous: dict[str, str] = {}
    for before, current in zip(observations, observations[1:]):
        if before.span.doc_id == current.span.doc_id:
            previous[current.id] = before.text
    return previous


def _same_paragraph(previous: dict[str, str], observations: tuple[Observation, ...], record: CaseRecord) -> dict[str, str]:
    """The previous sentences that sit in the same paragraph: only these may ground a label."""
    kept: dict[str, str] = {}
    for before, current in zip(observations, observations[1:]):
        if current.id in previous:
            gap = record.document(current.span.doc_id).text[before.span.end : current.span.start]
            if not _PARAGRAPH_BREAK.search(gap):
                kept[current.id] = previous[current.id]
    return kept


def _grounded(part: RubricPart, indicator: RubricIndicator, obs: Observation, before: str, context_words: set[str]) -> bool:
    """A label counts only if the note mentions what the part is about. The sentence before it in
    the same paragraph may supply the topic, unless the topic word is a context fact (a language)
    or the indicator claims something about every hearing."""
    if not part.keywords:
        return True
    if any(keyword in obs.text.lower() for keyword in part.keywords):
        return True
    if indicator.covers_all_hearings:
        return False
    return any(keyword in before.lower() for keyword in part.keywords if keyword not in context_words)


def _batches(item: RubricItem, shortlist: list[Observation], previous: dict[str, str], ctx: AnalysisContext) -> list[list[Observation]]:
    """Split the shortlist so each prompt, plus its reply budget, fits the model's context window."""
    settings = ctx.config.settings.llm
    system = system_with_schema(LABEL_SYSTEM, LabelReply)

    def fits(batch: list[Observation]) -> bool:
        prompt = label_user_prompt(item, [(f"O{n}", obs, previous.get(obs.id)) for n, obs in enumerate(batch, start=1)])
        return estimated_tokens(system, prompt) + settings.num_predict.labels <= settings.num_ctx

    batches: list[list[Observation]] = []
    current: list[Observation] = []
    for obs in shortlist:
        if current and (len(current) == LABELS_PER_BATCH or not fits([*current, obs])):
            batches.append(current)
            current = []
        current.append(obs)
    return [*batches, current] if current else batches


def _ask(item: RubricItem, batch: list[Observation], previous: dict[str, str], ctx: AnalysisContext) -> list[tuple[Observation, ObservationLabel]]:
    numbered = [(f"O{n}", obs) for n, obs in enumerate(batch, start=1)]
    prompt = label_user_prompt(item, [(obs_id, obs, previous.get(obs.id)) for obs_id, obs in numbered])
    reply = ctx.llm.complete_json(system=LABEL_SYSTEM, user=prompt, schema=LabelReply, purpose="labels")
    by_id = dict(numbered)
    answers = []
    for entry in reply.labels:
        found = _OBSERVATION_ID.search(entry.observation)  # models sometimes copy the whole "O3 (date): text" line
        if found and found.group(0) in by_id:
            answers.append((by_id[found.group(0)], entry))
    return answers


def _replies(item: RubricItem, shortlist: list[Observation], previous: dict[str, str], ctx: AnalysisContext):
    """Every label the model gave, and the notes it still had not labelled after being asked again."""
    answers: list[tuple[Observation, ObservationLabel]] = []
    pending = list(shortlist)
    for _ in range(ASK_ROUNDS):
        if not pending:
            break
        for batch in _batches(item, pending, previous, ctx):
            answers += _ask(item, batch, previous, ctx)
        answered = {obs.id for obs, _ in answers}
        pending = [obs for obs in pending if obs.id not in answered]
    return answers, pending


def _context_resemblance(item: RubricItem, answers, vectors: dict[str, np.ndarray], ctx: AnalysisContext) -> dict[tuple[str, str], bool]:
    """(observation, indicator) -> True when the note is closer to a context fact than to the indicator."""
    if not item.context or not answers:
        return {}
    indicators = {i.id: i.text for part in item.parts for i in part.compliance + part.violation}
    cited = sorted({entry.indicator_id.strip() for _, entry in answers} & indicators.keys())
    if not cited:
        return {}
    indicator_vectors = dict(zip(cited, ctx.embedder.encode([indicators[i] for i in cited])))
    context_vectors = ctx.embedder.encode([i.text for i in item.context])
    resemblance = {}
    for obs, entry in answers:
        indicator_id = entry.indicator_id.strip()
        if indicator_id in indicator_vectors:
            note = vectors[obs.id][None, :]
            to_context = float(cosine_matrix(note, context_vectors).max())
            to_indicator = float(cosine_matrix(note, indicator_vectors[indicator_id][None, :])[0, 0])
            resemblance[(obs.id, indicator_id)] = to_context > to_indicator
    return resemblance


def _label(
    item: RubricItem, shortlist: list[Observation], previous: dict[str, str], grounding: dict[str, str],
    vectors: dict[str, np.ndarray], ctx: AnalysisContext,
) -> tuple[tuple[LabeledObservation, ...], int]:  # fmt: skip
    answers, unlabelled = _replies(item, shortlist, previous, ctx)
    licensed = {i.id: (part, "supports", i) for part in item.parts for i in part.compliance}
    licensed |= {i.id: (part, "contradicts", i) for part in item.parts for i in part.violation}
    context_words = {keyword for indicator in item.context for keyword in indicator.keywords}
    like_context = _context_resemblance(item, answers, vectors, ctx)
    block_list = ctx.config.messages.block_list
    labels: dict[tuple[str, str], LabeledObservation] = {}
    for obs, entry in answers:
        indicator_id = entry.indicator_id.strip()
        part, label, indicator = licensed.get(indicator_id, (None, None, None))
        if part is None or entry.label != label:
            continue  # a label counts only with a matching indicator of the right kind
        if not _grounded(part, indicator, obs, grounding.get(obs.id, ""), context_words):
            continue  # the note never mentions what this part is about
        if like_context.get((obs.id, indicator_id)):
            continue  # a context fact (e.g. the defendant's language) is never evidence
        current = labels.get((obs.id, part.id))
        if current is None or (current.label == "supports" and label == "contradicts"):
            labels[(obs.id, part.id)] = LabeledObservation(
                observation_id=obs.id,
                part_id=part.id,
                indicator_id=indicator_id,
                label=label,
                span=obs.span,
                hearing_date=obs.hearing_date,
                model_note=filter_model_note(entry.note, block_list),
            )
    return tuple(labels.values()), len(unlabelled)


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


def _part_status(part: RubricPart, labels, keys: tuple[HearingKey, ...]) -> PartAssessment:
    base = {"part_id": part.id, "label": part.label, "required": part.required}
    mine = [label for label in labels if label.part_id == part.id]
    if any(label.label == "contradicts" for label in mine):
        return PartAssessment(**base, status="evidence_of_violation")
    supporting = [label for label in mine if label.label == "supports"]
    if part.scope == "case":
        return PartAssessment(**base, status="evidence_of_compliance" if supporting else "no_evidence")
    all_hearings = {i.id for i in part.compliance if i.covers_all_hearings}
    covered = set(keys) if any(label.indicator_id in all_hearings for label in supporting) else {hearing_key(label) for label in supporting}
    missing = [key for key in keys if key not in covered]
    status: GuaranteeStatus = "evidence_of_compliance" if supporting and not missing else "no_evidence"
    return PartAssessment(
        **base,
        status=status,
        hearings_missing=tuple(key for key in missing if isinstance(key, dt.date)),
        notes_missing=tuple(key for key in missing if isinstance(key, str)),
    )


def _overall(parts: tuple[PartAssessment, ...], unlabelled: int) -> GuaranteeStatus:
    if any(part.status == "evidence_of_violation" for part in parts):
        return "evidence_of_violation"
    required = [part for part in parts if part.required]
    if required and not unlabelled and all(part.status == "evidence_of_compliance" for part in required):
        return "evidence_of_compliance"
    return "no_evidence"


def _follow_up(record: CaseRecord, item: RubricItem, parts, context, unlabelled: int) -> FollowUp:
    uncovered = []
    for part in (part for part in parts if part.required and part.status == "no_evidence"):
        if not part.hearings_missing and not part.notes_missing:
            uncovered.append(part.label)
        uncovered += [f"{part.label}: hearing of {h.isoformat()}" for h in part.hearings_missing]
        uncovered += [f"{part.label}: {record.document(doc_id).title} (no hearing date)" for doc_id in part.notes_missing]
    if unlabelled:
        uncovered.append(f"{unlabelled} shortlisted notes were not labelled by the model")
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


def _assess(record, item, labels, unlabelled, context, keys, ctx) -> tuple[GuaranteeAssessment, Flag | None, FollowUp | None]:
    parts = tuple(_part_status(part, labels, keys) for part in item.parts)
    status = _overall(parts, unlabelled)
    flag = _flag(record, item, status, labels, ctx) if status != "no_evidence" else None
    follow_up = _follow_up(record, item, parts, context, unlabelled) if status == "no_evidence" else None
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
        unlabelled_notes=unlabelled,
    )
    return assessment, flag, follow_up


def run(record: CaseRecord, ctx: AnalysisContext) -> AbsenceResult:
    observations = record.observations
    keys = hearing_keys(observations)
    vectors = ctx.embedder.encode([obs.text for obs in observations]) if observations else np.zeros((0, 1), dtype=np.float32)
    by_id = {obs.id: vectors[row] for row, obs in enumerate(observations)}
    previous = _previous_sentences(observations)
    grounding = _same_paragraph(previous, observations, record)
    assessments, flags, follow_ups = [], [], []
    for item in ctx.config.rubric.items:
        shortlist = _shortlist(item, observations, vectors, keys, ctx) if observations else []
        labels, unlabelled = _label(item, shortlist, previous, grounding, by_id, ctx) if shortlist else ((), 0)
        context = _context(item, observations, vectors, labels, ctx)
        assessment, flag, follow_up = _assess(record, item, labels, unlabelled, context, keys, ctx)
        assessments.append(assessment)
        flags.extend([flag] if flag else [])
        follow_ups.extend([follow_up] if follow_up else [])
    return AbsenceResult(assessments=tuple(assessments), flags=tuple(flags), follow_ups=tuple(follow_ups))
