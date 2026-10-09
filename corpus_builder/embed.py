"""Embed every verified fact quote, locally, with the app's MiniLM model.

The vector is of the sentence around the quote (MiniLM reads about 256 word pieces, so it never sees
a whole document), capped at CONTEXT_CHARS and centred on the quote. It is stored under the quote's
span, the key the corpus and the OpenSearch index use.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Protocol

import numpy as np

from corpus_builder.store import BuildStore
from corpus_builder.verify import check_span
from ratio.embeddings import EMBEDDING_DIM
from ratio.extraction.align import sentence_bounds

log = logging.getLogger(__name__)

CONTEXT_CHARS = 1_200

FactKey = tuple[str, str, int, int]  # (precedent id, facet id, quote start, quote end)


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def context_bounds(text: str, start: int, end: int, cap: int = CONTEXT_CHARS) -> tuple[int, int]:
    """The sentence(s) around [start, end), cut to at most `cap` characters centred on the quote."""
    low, high = sentence_bounds(text, start, end)
    if high - low <= cap:
        return low, high
    centred = (start + end) // 2 - cap // 2
    low = max(low, min(centred, high - cap))
    return low, min(high, low + cap)


def _fact_contexts(store: BuildStore) -> Iterator[tuple[FactKey, str]]:
    for precedent_id, (facets, _) in store.verified().items():
        doc = store.document(precedent_id)
        if doc is None:
            log.warning("%s: verified but no longer stored; skipped", precedent_id)
            continue
        for facet in facets:
            for quote in facet.facts:
                check_span(doc, quote.span)
                low, high = context_bounds(doc.text, quote.span.start, quote.span.end)
                yield (precedent_id, facet.facet_id, quote.span.start, quote.span.end), doc.text[low:high]


def _unit_rows(vectors: np.ndarray, count: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (count, EMBEDDING_DIM):
        raise ValueError(f"the embedder returned shape {matrix.shape}, expected ({count}, {EMBEDDING_DIM})")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(matrix)) or np.any(norms == 0):
        raise ValueError("the embedder returned a zero or non-finite vector")
    return matrix / norms


def embed_all(store: BuildStore, embedder: Embedder) -> int:
    """Embed every verified fact and store its vector; returns how many were stored."""
    items = list(_fact_contexts(store))
    if not items:
        return 0
    vectors = _unit_rows(embedder.encode([text for _, text in items]), len(items))
    for (key, _), vector in zip(items, vectors, strict=True):
        store.put_vector(*key, vector)
    return len(items)
