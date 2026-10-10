"""Triangulation: past public cases whose passages on one right resemble the open case's.

Retrieval is local: MiniLM meaning search, plus the indicators of the chosen right that both sides
mention. The local Ollama model only picks and explains facts, naming passages by id, so every fact
opens its exact source span on both sides. Similarity is a pointer for the lawyer, never a finding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field, create_model

from ratio.config import default_config
from ratio.extraction.segment import sentence_spans
from ratio.llm import LLMError, OllamaClient

POINTER_LABEL = (
    "Similarity is a pointer, not a finding: passages are ranked by local meaning search and by the indicators "
    "of the chosen right that both cases mention. Whether a past case applies is for the reviewing lawyer."
)
METHOD = "Score = 0.75 × closest-passage similarity (local MiniLM) + 0.25 × share of this case's indicators the past case also mentions"
COSINE_WEIGHT, OVERLAP_WEIGHT = 0.75, 0.25  # retrieval heuristics, not legal thresholds
CASE_PASSAGES, SIMILAR_CASES, PASSAGES_PER_CASE = 3, 3, 2
PASSAGE_CHARS = 700  # a passage is whole sentences, up to this many characters
QUERY_CHARS = 300  # the corpus search accepts at most 300 characters
MIN_PASSAGE_CHARS = 40
CITES = "Cites the right"

SYSTEM = (
    "You compare an open case with past public cases for a reviewing lawyer. Use only the supplied passages. "
    "Pick the 3 to 5 facts that matter most for the named right. For each fact, give the id of the open-case "
    "passage and the id of the past-case passage it rests on, say why it is relevant, and whether the two cases "
    "are similar or different on it. Never say whether a right was violated, never predict an outcome, never "
    "characterise a person, and never invent facts or ids."
)


@dataclass(frozen=True)
class Right:
    id: str
    label: str
    query: str
    indicators: tuple[tuple[str, str], ...]  # (label, pattern): search aids, not legal tests


RIGHTS: tuple[Right, ...] = (
    Right(
        "14(2)",
        "Presumption of innocence (ICCPR Art. 14(2); General Comment 32, para. 30)",
        "presumption of innocence: officials publicly calling the accused guilty before the verdict, the accused "
        "shown in a cage, handcuffs or prison clothes in court, the burden of proof put on the defence, the court "
        "treating guilt as established before the verdict, a prejudicial media campaign by the authorities",
        (
            ("Officials call the accused guilty before the verdict", r"publicly (?:declared|called|named|described)|(?:minister|official|prosecutor|police)\w* (?:said|stated|declared|announced)[^.]{0,80}guilt"),
            ("Accused shown in a cage, handcuffs or prison clothes", r"\bcage\b|handcuff|shackle|prison (?:uniform|clothes|clothing)|restrained in (?:the )?court"),
            ("Burden of proof put on the defence", r"burden of pro(?:of|ving)|(?:must|had to|required to) prove (?:his|her|their|that)|prove (?:his|her|their) innocence"),
            ("Guilt treated as established before the verdict", r"guilt (?:was|is|had been) (?:established|proven)|already guilty|before (?:the )?(?:verdict|judgment)[^.]{0,60}guilt"),
            ("Prejudicial media campaign by the authorities", r"media campaign|press conference|state (?:media|television)|televised confession"),
            (CITES, r"presumption of innocence|presumed innocent|14\s*\(\s*2\s*\)|article 14,? para(?:graph)?\.? 2"),
        ),
    ),
    Right(
        "9(3)",
        "Prompt appearance before a judge (ICCPR Art. 9(3))",
        "brought promptly before a judge after arrest, days in police custody before a first appearance, pre-trial detention",
        (
            ("Delay before a judge", r"brought (?:promptly )?before a (?:judge|court)|first appearance|police custody|48 hours"),
            ("Pre-trial detention", r"pre-?trial detention|remand\w*|detention order|bail"),
            (CITES, r"9\s*\(\s*3\s*\)|article 9,? para(?:graph)?\.? 3"),
        ),
    ),
    Right(
        "14(1)",
        "Fair and public hearing by an independent tribunal (ICCPR Art. 14(1))",
        "independence and impartiality of the court, public hearing, closed trial, military court, judgment reasoning",
        (
            ("Independence or impartiality", r"independen\w*|impartial\w*|military court"),
            ("Closed hearing", r"public hearing|closed (?:hearing|trial)|in camera|behind closed doors"),
            (CITES, r"14\s*\(\s*1\s*\)|article 14,? para(?:graph)?\.? 1"),
        ),
    ),
    Right(
        "14(3)(b)",
        "Time, facilities and counsel to prepare the defence (ICCPR Art. 14(3)(b))",
        "adequate time and facilities to prepare the defence, access to a lawyer, confidential communication with counsel",
        (
            ("Access to counsel", r"access to (?:a |his |her )?(?:lawyer|counsel)|communicat\w* with (?:his |her )?counsel"),
            ("Time to prepare", r"time and facilities|prepar\w* (?:his|her|the) defen[cs]e|case file"),
            (CITES, r"14\s*\(\s*3\s*\)\s*\(\s*b\s*\)"),
        ),
    ),
    Right(
        "14(3)(c)",
        "Trial without undue delay (ICCPR Art. 14(3)(c))",
        "trial without undue delay, years of proceedings, repeated adjournments",
        (
            ("Delay of the trial", r"undue delay|without delay|delay\w* (?:of|in) (?:the )?(?:trial|proceedings)"),
            ("Adjournments", r"adjourn\w*|postpone\w*"),
            (CITES, r"14\s*\(\s*3\s*\)\s*\(\s*c\s*\)"),
        ),
    ),
    Right(
        "14(3)(e)",
        "Examination of witnesses (ICCPR Art. 14(3)(e))",
        "the defence could not call or cross-examine witnesses; witness statements read without the witness",
        (
            ("Witnesses", r"witness\w*|cross-?examin\w*"),
            (CITES, r"14\s*\(\s*3\s*\)\s*\(\s*e\s*\)"),
        ),
    ),
    Right(
        "14(3)(f)",
        "Free assistance of an interpreter (ICCPR Art. 14(3)(f))",
        "the accused did not understand the language of the court and had no interpreter",
        (
            ("Interpretation", r"interpret\w*|translat\w*|(?:did|does) not (?:speak|understand)"),
            (CITES, r"14\s*\(\s*3\s*\)\s*\(\s*f\s*\)"),
        ),
    ),
    Right(
        "14(3)(g)",
        "Not to be compelled to confess (ICCPR Art. 14(3)(g))",
        "confession obtained under coercion or torture, forced to sign a statement, self-incrimination",
        (
            ("Coerced confession", r"confess\w*|coerc\w*|forced to (?:sign|testify)|tortur\w*|self-incrimin\w*"),
            (CITES, r"14\s*\(\s*3\s*\)\s*\(\s*g\s*\)"),
        ),
    ),
)


def rights() -> list[dict]:
    return [{"id": right.id, "label": right.label} for right in RIGHTS]


def get_right(right_id: str) -> Right:
    for right in RIGHTS:
        if right.id == right_id:
            return right
    raise ValueError("Choose one of the listed rights.")


def indicators(right: Right, text: str) -> list[str]:
    return [label for label, pattern in right.indicators if re.search(pattern, text, re.IGNORECASE)]


def _span(doc, start: int, end: int) -> dict:
    return {"doc_id": doc.id, "start": start, "end": end, "text": doc.text[start:end], "document": doc.title}


def case_passages(record, max_chars: int = PASSAGE_CHARS) -> list[dict]:
    """Whole sentences of every document, grouped into passages of at most max_chars, with exact offsets."""
    passages = []
    for doc in record.documents:
        start = end = None
        for s, e in sentence_spans(doc.text):
            if start is not None and e - start > max_chars:
                passages.append(_span(doc, start, end))
                start = None
            start = s if start is None else start
            end = e
        if start is not None:
            passages.append(_span(doc, start, end))
    return [p for p in passages if len(p["text"].strip()) >= MIN_PASSAGE_CHARS]


def source_span(passage: dict) -> dict:
    return {key: passage[key] for key in ("doc_id", "start", "end", "text")}


def _score(cosine: float, overlap: float) -> float:
    return round(COSINE_WEIGHT * cosine + OVERLAP_WEIGHT * overlap, 4)


def _focus_span(right: Right, passage: dict) -> dict:
    """The sentences of a passage that mention the right's indicators (the whole passage if none do)."""
    text = passage["text"]
    matched = [(s, e) for s, e in sentence_spans(text) if e - s >= MIN_PASSAGE_CHARS and indicators(right, text[s:e])]
    if not matched:
        return source_span(passage)
    start, end = matched[0][0], matched[-1][1]
    if end - start > PASSAGE_CHARS:
        end = matched[0][1]
    return {"doc_id": passage["doc_id"], "start": passage["start"] + start, "end": passage["start"] + end, "text": text[start:end]}


def _rank_case_passages(right: Right, passages: list[dict], embed) -> list[dict]:
    focused = [{**p, "span": _focus_span(right, p)} for p in passages]
    vectors = np.asarray(embed([right.query] + [p["span"]["text"] for p in focused]))
    query, rows = vectors[0], vectors[1:]
    scored = []
    for passage, vector in zip(focused, rows):
        found = [label for label in indicators(right, passage["span"]["text"]) if label != CITES]
        cosine = round(float(np.dot(query, vector)), 4)
        scored.append({**passage, "cosine": cosine, "indicators": found, "score": _score(cosine, min(1, len(found)))})
    scored.sort(key=lambda p: (-p["score"], p["doc_id"], p["start"]))
    return scored[:CASE_PASSAGES]


def _similar_cases(right: Right, case_top: list[dict], research, embed) -> list[dict]:
    candidates = {}
    queries = [right.query] + [p["span"]["text"] for p in case_top]
    for query in queries:
        for hit in research.search(query[:QUERY_CHARS], mode="meaning", limit=12)["hits"]:
            if not hit["document"].get("private"):
                candidates.setdefault((hit["document"]["id"], hit["start"], hit["end"]), hit)
    if not candidates:
        return []
    hits = list(candidates.values())
    vectors = np.asarray(embed([p["span"]["text"] for p in case_top] + [h["text"] for h in hits]))
    similarity = vectors[len(case_top):] @ vectors[: len(case_top)].T  # one row per hit
    case_indicators = {label for p in case_top for label in p["indicators"]}
    by_document: dict[str, dict] = {}
    for hit, row in zip(hits, similarity):
        best = int(np.argmax(row))
        entry = {
            "reference_span": {"doc_id": "precedent:" + hit["document"]["id"], "start": hit["start"], "end": hit["end"], "text": hit["text"]},
            "cosine": round(float(row[best]), 4),
            "closest_case_passage": best,
            "indicators": indicators(right, hit["text"]),
        }
        by_document.setdefault(hit["document"]["id"], {"reference": hit["document"], "passages": []})["passages"].append(entry)
    cases = []
    for item in by_document.values():
        passages = sorted(item["passages"], key=lambda e: -e["cosine"])[:PASSAGES_PER_CASE]
        shared = sorted(case_indicators & {label for e in passages for label in e["indicators"]})
        overlap = len(shared) / len(case_indicators) if case_indicators else 0.0
        cases.append({
            "reference": item["reference"],
            "passages": passages,
            "cosine": passages[0]["cosine"],
            "shared_indicators": shared,
            "cites_right": any(CITES in e["indicators"] for e in passages),
            "score": _score(passages[0]["cosine"], overlap),
            "facts": [],
        })
    cases.sort(key=lambda c: (-c["score"], c["reference"]["id"]))
    return cases[:SIMILAR_CASES]


def _explain(right: Right, case_top: list[dict], cases: list[dict], llm) -> list[dict]:
    """The local model picks 3–5 facts by passage id; returns the cases with their facts attached."""
    case_ids = {f"c{i + 1}": p for i, p in enumerate(case_top)}
    reference_ids = {
        f"r{ci + 1}{'abcdefgh'[pi]}": (ci, entry)
        for ci, case in enumerate(cases)
        for pi, entry in enumerate(case["passages"])
    }
    fact = create_model(
        "TriangulationFact",
        case_passage=(Literal[tuple(case_ids)], ...),
        reference_passage=(Literal[tuple(reference_ids)], ...),
        fact=(str, Field(min_length=1, max_length=200)),
        why_relevant=(str, Field(min_length=1, max_length=260)),
        comparison=(Literal["similar", "different"], ...),
        explanation=(str, Field(min_length=1, max_length=260)),
    )
    schema = create_model("TriangulationFacts", facts=(list[fact], Field(min_length=1, max_length=5)))
    user = json.dumps({
        "right": right.label,
        "open_case_passages": [{"id": k, "text": p["span"]["text"]} for k, p in case_ids.items()],
        "past_case_passages": [
            {"id": k, "case": cases[ci]["reference"]["title"], "text": e["reference_span"]["text"]}
            for k, (ci, e) in reference_ids.items()
        ],
    })
    answer = llm.complete_json(system=SYSTEM, user=user, schema=schema, purpose="triangulation")
    facts: dict[int, list[dict]] = {}
    for item in answer.facts:
        ci, entry = reference_ids[item.reference_passage]
        facts.setdefault(ci, []).append({
            "fact": item.fact,
            "why_relevant": item.why_relevant,
            "comparison": item.comparison,
            "explanation": item.explanation,
            "case_span": case_ids[item.case_passage]["span"],
            "case_document": case_ids[item.case_passage]["document"],
            "reference_span": entry["reference_span"],
        })
    return [{**case, "facts": facts.get(ci, [])} for ci, case in enumerate(cases)]


def triangulate(library, research, case_id: str, right_id: str = "14(2)", *, llm=None) -> dict:
    right = get_right(right_id)
    record = library.cases.load_case(case_id)
    if record is None:
        raise KeyError("Case not found.")
    result = {"right": {"id": right.id, "label": right.label}, "label": POINTER_LABEL, "method": METHOD}
    passages = case_passages(record)
    if not passages:
        return {**result, "case_passages": [], "similar_cases": [], "note": "The case has no readable text."}
    case_top = _rank_case_passages(right, passages, research.embed)
    shown = [{**p["span"], "document": p["document"], "indicators": p["indicators"], "cosine": p["cosine"]} for p in case_top]
    cases = _similar_cases(right, case_top, research, research.embed)
    if not cases:
        return {**result, "case_passages": shown, "similar_cases": [], "note": "No public reference passage was retrieved on this computer."}
    try:
        cases = _explain(right, case_top, cases, llm or OllamaClient(default_config().settings.llm))
        note = "The local model picked and explained the facts (model text, unverified); each fact opens its source on both sides."
    except LLMError:
        note = "The local model is unavailable, so no facts were picked: the closest passages are shown without explanation."
    return {**result, "case_passages": shown, "similar_cases": cases, "note": note}
