"""Split a document into model-sized chunks on sentence boundaries, keeping absolute offsets.

Neighbouring chunks overlap by up to ``overlap_chars`` so an event stated across a chunk edge
is still seen whole; duplicates from the overlap are removed after alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from ratio.extraction.segment import sentence_spans


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int


def chunk_spans(text: str, *, max_chars: int, overlap_chars: int) -> tuple[Chunk, ...]:
    sentences = sentence_spans(text)
    chunks: list[Chunk] = []
    first = 0
    while first < len(sentences):
        start = sentences[first][0]
        last = first
        while last + 1 < len(sentences) and sentences[last + 1][1] - start <= max_chars:
            last += 1
        chunks.append(Chunk(index=len(chunks), start=start, end=sentences[last][1]))
        if last + 1 >= len(sentences):
            break
        nxt = last + 1
        while nxt - 1 > first and sentences[last][1] - sentences[nxt - 1][0] <= overlap_chars:
            nxt -= 1
        if sentences[last + 1][1] - sentences[nxt][0] > max_chars:
            nxt = last + 1  # the next sentence cannot share a chunk with the overlap, so no overlap
        first = nxt
    return tuple(chunks)
