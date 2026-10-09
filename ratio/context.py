"""Services injected into the modules, so tests can swap in fakes for the LLM and embedder."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from ratio.config import RatioConfig

T = TypeVar("T", bound=BaseModel)
CHARS_PER_TOKEN = 3  # conservative estimate, so prompts are refused before Ollama would truncate them


class ModelReplyError(RuntimeError):
    """The model answered, but the answer cannot be used (cut off at the reply length limit, or not
    the requested schema). Modules may recover from it; a missing model or recording is not one."""


def system_with_schema(system: str, schema: type[BaseModel]) -> str:
    """The system prompt as sent: the reply's JSON schema is given in the prompt as well as in ``format``."""
    return f"{system}\n\nReply with JSON only, matching this JSON schema:\n{json.dumps(schema.model_json_schema())}"


def estimated_tokens(*texts: str) -> int:
    return sum(len(text) for text in texts) // CHARS_PER_TOKEN


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
