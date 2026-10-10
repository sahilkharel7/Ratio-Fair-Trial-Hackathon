"""Questions about the open case, answered by the local Ollama model from retrieved passages only.

The open case's own passages are retrieved first, then public reference passages. The model names the
passages its answer rests on by id, so every citation opens an exact source span; an answer without a
supported citation becomes a refusal. Nothing is stored: the conversation lives in the browser tab.
"""

from __future__ import annotations

import json
import re
from typing import Literal

import numpy as np
from pydantic import Field, create_model

from ratio.config import default_config
from ratio.llm import LLMError, LLMResponseError, OllamaClient
from ratio.messages import filter_model_note
from ratio.triangulation import QUERY_CHARS, case_passages, source_span

REFUSAL = "The documents do not support an answer to this question."
UNAVAILABLE = "The local model is unavailable, so no answer was generated. Check that Ollama is running."
UNCHECKED = (
    "The local model's reply could not be checked against the sources, so it is not shown. "
    "Try a shorter or narrower question."
)
LABEL = "Model answer, unverified."
BANNER = "Local model only. Nothing leaves this computer."
CASE_TOP, MAX_PER_DOCUMENT, REFERENCE_TOP, MAX_CITATIONS = 6, 2, 3, 6
MAX_QUESTION = 500
MAX_ANSWER_CHARS = 1200  # shown; the model itself is never stopped mid-sentence (schema bound 4000)
COSINE_WEIGHT, WORD_WEIGHT = 0.75, 0.25  # the same retrieval heuristic as the Precedent library search
_STOPWORDS = frozenset("what when where which while about this that with from have were their there they them she her his him and the for why was who how does did into case".split())

SYSTEM = (
    "Answer the reviewing lawyer's question about the open case using only the supplied passages, in at most "
    "five sentences, and cite the ids of the passages your answer rests on. If the passages answer part of the "
    "question, answer that part and say what they do not cover. Set supported to false only when no passage "
    "is relevant. Never say whether a right was violated, never predict an outcome, never characterise a person, "
    "and never invent facts or ids."
)


def _reply(supported: bool, answer: str, citations: list[dict]) -> dict:
    return {"supported": supported, "answer": answer, "citations": citations, "label": LABEL, "banner": BANNER}


def _shorten(text: str) -> str:
    """At most MAX_ANSWER_CHARS, ending on a whole sentence where one fits."""
    if len(text) <= MAX_ANSWER_CHARS:
        return text
    cut = text[: MAX_ANSWER_CHARS - 1]
    end = cut.rfind(". ")
    return cut[: end + 1] if end >= MAX_ANSWER_CHARS // 4 else cut.rstrip() + "…"


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS}


def _case_sources(record, question: str, embed) -> list[dict]:
    """The best passages by meaning and wording, at most MAX_PER_DOCUMENT from any one document, so a broad
    question sees the charge, the facts and the decision rather than one long filing."""
    passages = case_passages(record)
    if not passages:
        return []
    vectors = np.asarray(embed([question] + [p["text"] for p in passages]))
    cosine = vectors[1:] @ vectors[0]
    asked = _words(question)
    score = [
        COSINE_WEIGHT * float(c) + WORD_WEIGHT * (len(asked & _words(p["text"])) / len(asked) if asked else 0.0)
        for c, p in zip(cosine, passages)
    ]
    ranked = sorted(range(len(passages)), key=lambda i: (-score[i], i))
    chosen, per_document = [], {}
    for i in ranked:  # first pass: spread across documents
        doc = passages[i]["doc_id"]
        if per_document.get(doc, 0) < MAX_PER_DOCUMENT and len(chosen) < CASE_TOP:
            per_document[doc] = per_document.get(doc, 0) + 1
            chosen.append(i)
    chosen += [i for i in ranked if i not in chosen][: CASE_TOP - len(chosen)]  # then fill with the next best
    return [{"kind": "case", "document": passages[i]["document"], "span": source_span(passages[i])} for i in chosen]


def _reference_sources(research, question: str) -> list[dict]:
    hits = research.search(question[:QUERY_CHARS], mode="meaning", limit=REFERENCE_TOP * 2)["hits"]
    return [
        {
            "kind": "reference",
            "reference": hit["document"],
            "span": {"doc_id": "precedent:" + hit["document"]["id"], "start": hit["start"], "end": hit["end"], "text": hit["text"]},
        }
        for hit in hits
        if not hit["document"].get("private")
    ][:REFERENCE_TOP]


def ask(library, research, case_id: str, question: str, *, llm=None) -> dict:
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION:
        raise ValueError(f"Ask a question of 1 to {MAX_QUESTION} characters.")
    question = question.strip()
    record = library.cases.load_case(case_id)
    if record is None:
        raise KeyError("Case not found.")
    found = _case_sources(record, question, research.embed) + _reference_sources(research, question)
    if not found:
        return _reply(False, REFUSAL, [])
    sources = {f"{'c' if s['kind'] == 'case' else 'r'}{i + 1}": s for i, s in enumerate(found)}
    schema = create_model(
        "CaseAnswer",
        supported=(bool, ...),
        answer=(str, Field(max_length=4000)),
        citations=(list[Literal[tuple(sources)]], Field(max_length=MAX_CITATIONS)),
    )
    user = json.dumps({
        "question": question,
        "passages": [
            {"id": key, "source": s["document"] if s["kind"] == "case" else s["reference"]["title"], "text": s["span"]["text"]}
            for key, s in sources.items()
        ],
    })
    try:
        answer = (llm or OllamaClient(default_config().settings.llm)).complete_json(
            system=SYSTEM, user=user, schema=schema, purpose="case_chat"
        )
    except LLMResponseError:  # ids outside the passages, or a reply that fails the schema twice
        return _reply(False, UNCHECKED, [])
    except LLMError:
        return _reply(False, UNAVAILABLE, [])
    cited = list(dict.fromkeys(answer.citations))
    text = filter_model_note(answer.answer, default_config().messages.block_list)  # hard rule 3
    if not cited or text is None:
        return _reply(False, REFUSAL, [])
    # A cited answer the model marks as partial is still shown (as not fully supported), with its sources.
    return _reply(bool(answer.supported), _shorten(text), [sources[key] for key in cited][:MAX_CITATIONS])
