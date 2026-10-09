"""Shared helpers for the corpus builder tests (extract, verify, embed, build-db).

Everything here is SYNTHETIC: invented documents, a scripted model and a deterministic embedder.
Nothing touches the network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from corpus_builder.extract import ExtractionFailed
from ratio.embeddings import EMBEDDING_DIM
from ratio.precedent_schema import PrecedentDoc

FIXTURES = Path(__file__).parent
VIEWS = "views_synthetic.txt"
REPORT = "report_synthetic.txt"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make_doc(precedent_id: str, text: str, *, kind: str = "ccpr_views", **overrides: object) -> PrecedentDoc:
    """A precedent built from fixture text; the title is invented and never part of the text."""
    fields: dict[str, object] = {
        "id": precedent_id,
        "kind": kind,
        "title": f"Synthetic precedent {precedent_id}",
        "body": "Synthetic deciding body",
        "url": f"https://example.invalid/{precedent_id}",
        "retrieved_at": "2099-01-01T00:00:00+00:00",
        "raw_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "attribution": "Synthetic source, for tests only",
        "text": text,
    }
    return PrecedentDoc.model_validate({**fields, **overrides})


def facet(facet_id: str, facts: list[str], finding_kind: str = "violation_found", finding: str | None = None) -> dict:
    """One facet of a model answer, as JSON."""
    return {
        "facet_id": facet_id,
        "facts": [{"quote": quote} for quote in facts],
        "finding_kind": finding_kind,
        "finding": None if finding is None else {"quote": finding},
    }


@dataclass(frozen=True)
class Call:
    system: str
    user: str
    schema: type
    purpose: str


@dataclass
class FakeLLM:
    """A scripted model: `answer` maps the user prompt to an answer dict, or raises."""

    answer: Callable[[str], Mapping] | Mapping
    model: str = "fake-model"
    window_chars: int = 200_000
    calls: list[Call] = field(default_factory=list)

    def complete_json(self, *, system: str, user: str, schema: type, purpose: str):  # noqa: ANN201 - returns schema
        self.calls.append(Call(system, user, schema, purpose))
        answer = self.answer(user) if callable(self.answer) else self.answer
        return schema.model_validate(answer)


def failing_llm(reason: str = "the answer stopped early (MAX_TOKENS)") -> FakeLLM:
    def fail(_user: str) -> Mapping:
        raise ExtractionFailed(reason)

    return FakeLLM(answer=fail)


class FakeEmbedder:
    """Deterministic unit vectors: the same text always gets the same vector."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.texts.extend(texts)
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            row = np.random.default_rng(seed).normal(size=EMBEDDING_DIM)
            rows.append(row / np.linalg.norm(row))
        return np.asarray(rows, dtype=np.float32).reshape(len(texts), EMBEDDING_DIM)


@dataclass(frozen=True)
class FakeSource:
    """The fields of corpus_builder.sources.Source that build_db reads."""

    id: str
    kind: str
    name: str
    attribution: str
    terms_url: str = "https://example.invalid/terms"
    terms_checked: str = "2099-01-01"
    allowed_prefixes: tuple[str, ...] = ("https://example.invalid/",)
