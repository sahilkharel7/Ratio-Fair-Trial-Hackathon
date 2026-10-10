"""Embed every verified fact quote, locally, with the app's MiniLM model.

The vector is of the sentence around the quote (MiniLM reads about 256 word pieces, so it never sees
a whole document), capped at CONTEXT_CHARS and centred on the quote. It is stored under the quote's
span, the key the corpus and the OpenSearch index use.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from typing import Protocol

import numpy as np

from corpus_builder.store import BuildStore
from corpus_builder.verify import check_span
from ratio.embeddings import EMBEDDING_DIM
from ratio.extraction.align import sentence_bounds
from ratio.extraction.chunking import chunk_spans

log = logging.getLogger(__name__)

CONTEXT_CHARS = 1_200
PASSAGE_CHARS = 800  # one search passage: sentences up to this length (MiniLM reads about 256 word pieces)
MIN_PASSAGE_CHARS = 120  # shorter pieces (headings, page numbers, stray citation lines) are not worth searching
PASSAGE_BATCH = 256
MAX_NOTE_SHARE = 0.5  # a passage mostly made of footnote or citation lines is not searched
_CITATION = re.compile(
    r"U\.\s?N\. Doc|\bparas?\.\s*\d|¶\s*\d|\bId\.|https?://|www\.|available at|Communication No|\bsupra\b|\bibid\b"
    r"|CCPR/C|A/HRC|General Comment No|\bArt\. \d|\bpp?\. \d",
    re.IGNORECASE,
)
# "58 Trial Monitor's Notes, October 27, 2020." or a short numbered line such as "101 Judgment at 13."
_NOTE = re.compile(r"^\s*\d{1,3}\s+\S(?:.*\b(?:19|20)\d\d\b|.{0,40}$)")
_US_DATE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? \d{1,2}, (?:19|20)\d\d\b")
_NUMBERED = re.compile(r"^\s*\d{1,3}\s+\S")
_REFERENCE_MARK = re.compile(r"(?<=[.;:!?)\"\u201d\u2019])\s?\d{1,3}$")
_SENTENCE_ENDS = (".", ";", ":", "!", "?", ")", '"', "\u201d", "\u2019")
_LINK_LINE = re.compile(r"^\s*(?:available at:?\s*)?(?:https?://|www\.)\S*\s*$", re.IGNORECASE)

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


def passage_spans(text: str) -> list[tuple[int, int]]:
    """The document cut into search passages on sentence boundaries, each a trimmed, non-trivial span.
    Footnotes are left out: each passage lies between footnote lines, and one still mostly made of
    citations is dropped. They match many queries and say nothing about the facts."""
    spans = []
    for low, high in _body_runs(text):
        for chunk in chunk_spans(text[low:high], max_chars=PASSAGE_CHARS, overlap_chars=0):
            start, end = low + chunk.start, low + chunk.end
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end - start >= MIN_PASSAGE_CHARS and note_share(text[start:end]) <= MAX_NOTE_SHARE:
                spans.append((start, end))
    return spans


def is_footnote_line(line: str, previous: str = "", previous_is_footnote: bool = False) -> bool:
    """A footnote ("106 It is unclear ...; see A/HRC/45/16, para. 51", "58 Notes, October 27, 2020.",
    "103 Id.") or a bare link line. A numbered line of the text itself ("2.1 The author ...") is not, nor
    is a short one that ends the previous line's sentence ("...violated Article" / "21 of the ICCPR.")."""
    if _LINK_LINE.match(line):
        return True
    if not _NUMBERED.match(line):
        return False
    if _CITATION.search(line) or _US_DATE.search(line):
        return True
    ended = _REFERENCE_MARK.sub("", previous.rstrip())  # "...were broken.110": a footnote mark after the full stop
    starts_anew = previous_is_footnote or not previous.strip() or ended.endswith(_SENTENCE_ENDS)
    return len(line.strip()) <= 45 and starts_anew


def _body_runs(text: str) -> Iterator[tuple[int, int]]:
    """[start, end) of each run of lines between footnote lines."""
    start = position = 0
    previous, previous_is_footnote = "", False
    for line in text.splitlines(keepends=True):
        end = position + len(line)
        footnote = is_footnote_line(line, previous, previous_is_footnote)
        if footnote:
            if position > start:
                yield start, position
            start = end
        if line.strip():
            previous, previous_is_footnote = line, footnote
        else:
            previous = ""  # a blank line: the next line starts a new block
        position = end
    if start < len(text):
        yield start, len(text)


def note_share(passage: str) -> float:
    """The share of a passage's characters on footnote or citation lines ("60 Id.", "U.N. Doc. ..., para. 4")."""
    lines = [line for line in passage.splitlines() if line.strip()]
    total = sum(map(len, lines))
    notes = sum(len(line) for line in lines if _CITATION.search(line) or _NOTE.match(line) or _US_DATE.search(line))
    return notes / total if total else 1.0


def embed_passages(store: BuildStore, embedder: Embedder, *, only: set[str] | None = None, redo: bool = False) -> int:
    """Embed the passages of each stored document (or only those named) whose passages are missing or were
    cut differently; returns the number of passages stored. Search over these finds a precedent by its
    wording, not only its facets."""
    done = {} if redo else store.passage_cuts()
    stored = 0
    for doc in store.documents():
        if only is not None and doc.id not in only:
            continue
        spans = passage_spans(doc.text)
        if done.get(doc.id, []) == spans:
            continue
        rows: list[tuple[int, int, np.ndarray]] = []
        for first in range(0, len(spans), PASSAGE_BATCH):
            batch = spans[first : first + PASSAGE_BATCH]
            vectors = _unit_rows(embedder.encode([doc.text[a:b] for a, b in batch]), len(batch))
            rows.extend((a, b, vec) for (a, b), vec in zip(batch, vectors, strict=True))
        store.put_passages(doc.id, rows)
        stored += len(rows)
        log.info("%s: %d passages embedded", doc.id, len(rows))
    return stored
