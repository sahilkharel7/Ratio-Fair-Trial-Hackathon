"""Questions about the open case, answered by the local Ollama model from retrieved passages only.

The open case's own passages are retrieved first, then public reference passages. The model names the
passages its answer rests on by id, so every citation opens an exact source span; an answer without a
supported citation becomes a refusal. Nothing is stored: the conversation lives in the browser tab.
"""

from __future__ import annotations

import json
from typing import Literal

import numpy as np
from pydantic import Field, create_model

from ratio.config import default_config
from ratio.llm import LLMError, LLMResponseError, OllamaClient
from ratio.messages import filter_model_note
from ratio.triangulation import QUERY_CHARS, case_passages, source_span

REFUSAL = "The documents do not support an answer to this question."
UNAVAILABLE = "The local model is unavailable, so no answer was generated. Check that Ollama is running."
LABEL = "Model answer, unverified."
BANNER = "Local model only. Nothing leaves this computer."
CASE_TOP, REFERENCE_TOP, MAX_CITATIONS = 4, 3, 4
MAX_QUESTION = 500

SYSTEM = (
    "Answer the reviewing lawyer's question about the open case using only the supplied passages, and cite the "
    "ids of the passages your answer rests on. If the passages do not answer the question, set supported to "
    "false. Never say whether a right was violated, never predict an outcome, never characterise a person, and "
    "never invent facts or ids."
)


def _reply(supported: bool, answer: str, citations: list[dict]) -> dict:
    return {"supported": supported, "answer": answer, "citations": citations, "label": LABEL, "banner": BANNER}


def _case_sources(record, question: str, embed) -> list[dict]:
    passages = case_passages(record)
    if not passages:
        return []
    vectors = np.asarray(embed([question] + [p["text"] for p in passages]))
    order = np.argsort(-(vectors[1:] @ vectors[0]), kind="stable")[:CASE_TOP]
    return [{"kind": "case", "document": passages[i]["document"], "span": source_span(passages[i])} for i in order]


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
        answer=(str, Field(max_length=700)),
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
        return _reply(False, REFUSAL, [])
    except LLMError:
        return _reply(False, UNAVAILABLE, [])
    cited = list(dict.fromkeys(answer.citations))
    text = filter_model_note(answer.answer, default_config().messages.block_list)  # hard rule 3
    if not answer.supported or not cited or text is None:
        return _reply(False, REFUSAL, [])
    return _reply(True, text, [sources[key] for key in cited])
