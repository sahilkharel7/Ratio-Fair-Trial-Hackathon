"""Deterministic test doubles for the LLM and the embedder (no Ollama, no model files needed),
and small SYNTHETIC case records built by the real loader."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from unidecode import unidecode

from ratio.extraction.build import build_base_record
from ratio.extraction.loader import CaseManifest, ManifestDocument
from ratio.schema import Argument, CaseRecord, DocType, Party, stable_id

TEST_MARKER = "SYNTHETIC: test document; every name in it is invented."


def make_record(documents: Sequence[tuple[str, DocType, str]], case_id: str = "test-case") -> CaseRecord:
    """A deterministic record from (path, type, text) triples, built exactly as an upload would be."""
    manifest = CaseManifest(
        case_id=case_id,
        title="Republic of Testland v. A. Example",
        court="Test District Court",
        charge_type="Test charge",
        data_provenance="synthetic",
        synthetic=True,
        documents=tuple(ManifestDocument(path=path, type=doc_type, title=path) for path, doc_type, _ in documents),
    )
    files = {path: f"{TEST_MARKER}\n{text}".encode("utf-8") for path, _, text in documents}
    return build_base_record(manifest, files)


def with_argument(record: CaseRecord, doc_path: str, quote: str, party: Party = "defense") -> CaseRecord:
    """The record plus the note sentence containing ``quote`` as a party argument."""
    doc_id = f"{record.case_id}/{doc_path}"
    sentence = next(obs for obs in record.observations if obs.span.doc_id == doc_id and quote in obs.text)
    argument = Argument(
        id=stable_id(sentence.id, party, "argument"),
        party=party,
        text=sentence.text,
        hearing_date=sentence.hearing_date,
        span=sentence.span,
    )
    return record.model_copy(update={"arguments": record.arguments + (argument,)})

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and or of to in on at for by with from as is was were be been has had have that this these "
    "those it its he she his her him their them they which who whom the".split()
)


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts that share words get a high cosine similarity."""

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in (t for t in _TOKEN.findall(unidecode(text).lower()) if t not in _STOPWORDS):
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                matrix[row, int.from_bytes(digest[:4], "big") % self._dim] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


@dataclass(frozen=True)
class LLMCall:
    purpose: str
    system: str
    user: str
    schema_name: str


Responder = Callable[[str, str, type[BaseModel], str], BaseModel | dict]


class FakeLLM:
    """Scripted LLM: ``responder(system, user, schema, purpose)`` decides each reply."""

    def __init__(self, responder: Responder, model: str = "fake-llm") -> None:
        self._responder = responder
        self._model = model
        self._calls: list[LLMCall] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def calls(self) -> tuple[LLMCall, ...]:
        return tuple(self._calls)

    def complete_json(self, *, system: str, user: str, schema: type[BaseModel], purpose: str) -> BaseModel:
        self._calls.append(LLMCall(purpose=purpose, system=system, user=user, schema_name=schema.__name__))
        reply = self._responder(system, user, schema, purpose)
        return reply if isinstance(reply, schema) else schema.model_validate(reply)
