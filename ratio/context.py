"""Services injected into the modules, so tests can swap in fakes for the LLM and embedder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from ratio.config import RatioConfig

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """Returns a validated instance of ``schema`` for one structured-output request."""

    @property
    def model(self) -> str: ...

    def complete_json(self, *, system: str, user: str, schema: type[T], purpose: str) -> T: ...


class Embedder(Protocol):
    """Encodes texts as L2-normalised float32 vectors, shape (len(texts), dim)."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class AnalysisContext:
    llm: LLMClient
    embedder: Embedder
    config: RatioConfig
