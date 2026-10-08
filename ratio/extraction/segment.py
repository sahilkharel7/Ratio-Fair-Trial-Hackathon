"""Deterministic segmentation of document text into sentences with exact character offsets.

Blocks are separated by blank lines. The first block of a multi-block document is its header
(caption lines, one segment per line). A final block that starts with "(signed)" is the
signature. A single-line block that looks like a heading ("III. STATEMENT OF FACTS") starts a
section; every other block is split into sentences.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from ratio.schema import PassageKind

_BLOCK = re.compile(r"[^\n](?:.|\n(?![ \t]*\n))*", re.MULTILINE)
_LINE = re.compile(r"[^\n]+")
_NUMBERED_HEADING = re.compile(r"^(?:[IVXLC]+|\d+|[A-Z])\.\s+\S")
_SIGNATURE = re.compile(r"^\(signed\)|^signed[:,]", re.IGNORECASE)
_LABEL_LINE = re.compile(r"^[A-Z][A-Za-z .()/-]{0,40}:\s+\S")
MAX_HEADER_LINE = 300
_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*(?=\s+[\"'“‘(\[]?[A-Z0-9])")
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "hon", "st", "no", "nos", "art", "arts", "para", "paras",
        "p", "pp", "sec", "cf", "etc", "e.g", "i.e", "v", "vs", "approx", "jan", "feb", "mar",
        "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "ibid",
    }
)  # fmt: skip
_CITATION = re.compile(
    r"\b(?:Article|Art\.|Section|Sec\.)\s+\d+[A-Za-z]?(?:\(\d+\))*"
    r"(?:\s+of\s+the\s+(?:[A-Z][\w-]*\s+){0,4}?(?:Code|Act|Law|Constitution|Covenant))?"
)


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    kind: PassageKind
    section: str | None


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _looks_like_header(block: str) -> bool:
    """Caption lines: short, and either 'Label: value' lines or lines without sentence endings."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines or any(len(line) > MAX_HEADER_LINE for line in lines):
        return False
    if any(_LABEL_LINE.match(line) for line in lines):
        return True
    return all(len(line) <= 120 and line[-1] not in ".!?" for line in lines)


def _is_heading(block: str) -> bool:
    line = block.strip()
    if "\n" in line or len(line) > 90 or line[-1] in ".!?;,":
        return False
    letters = [c for c in line if c.isalpha()]
    return bool(_NUMBERED_HEADING.match(line)) or (len(letters) > 2 and all(c.isupper() for c in letters))


def _is_abbreviation(text: str, dot: int, sentence_start: int) -> bool:
    if text[dot] != ".":
        return False
    begin = dot
    while begin > sentence_start and (text[begin - 1].isalnum() or text[begin - 1] == "."):
        begin -= 1
    word = text[begin:dot].lower()
    if not word:
        return False
    return (
        word in _ABBREVIATIONS
        or (len(word) == 1 and word.isalpha())
        or re.fullmatch(r"(?:[a-z]\.)+[a-z]", word) is not None
    )


def _inside_quotation(text: str, start: int, end: int) -> bool:
    chunk = text[start:end]
    return chunk.count('"') % 2 == 1 or chunk.count("“") > chunk.count("”")


def split_sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence spans inside [start, end), skipping abbreviations, initials and quoted text."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in _BOUNDARY.finditer(text, start, end):
        stop = match.end()
        if _is_abbreviation(text, match.start(), cursor) or _inside_quotation(text, cursor, stop):
            continue
        spans.append(_trim(text, cursor, stop))
        cursor = stop
    spans.append(_trim(text, cursor, end))
    return [span for span in spans if span[1] > span[0]]


def _line_segments(text: str, start: int, end: int, kind: PassageKind, section: str | None) -> list[Segment]:
    segments = []
    for match in _LINE.finditer(text, start, end):
        s, e = _trim(text, match.start(), match.end())
        if e > s:
            segments.append(Segment(s, e, kind, section))
    return segments


@functools.lru_cache(maxsize=128)
def segment_document(text: str) -> tuple[Segment, ...]:
    blocks = [_trim(text, m.start(), m.end()) for m in _BLOCK.finditer(text)]
    blocks = [(s, e) for s, e in blocks if e > s]
    segments: list[Segment] = []
    section: str | None = None
    for index, (start, end) in enumerate(blocks):
        block = text[start:end]
        if index == 0 and len(blocks) > 1 and _looks_like_header(block):
            segments.extend(_line_segments(text, start, end, "header", None))
        elif index == len(blocks) - 1 and _SIGNATURE.match(block):
            segments.extend(_line_segments(text, start, end, "signature", section))
        elif _is_heading(block):
            section = block.strip()
            segments.append(Segment(start, end, "heading", section))
        else:
            segments.extend(Segment(s, e, "body", section) for s, e in split_sentences(text, start, end))
    return tuple(segments)


def sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((segment.start, segment.end) for segment in segment_document(text))


def header_value(text: str, label: str) -> tuple[int, int] | None:
    """Offsets of the value of a ``Label: value`` line in the document's header block."""
    first = _BLOCK.search(text)
    if first is None or not _looks_like_header(first.group(0)):
        return None
    prefix = label.lower() + ":"
    offset = first.start()
    for line in first.group(0).split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            value = stripped[len(prefix) :].strip()
            if value:
                start = offset + line.index(value, line.lower().index(prefix) + len(prefix))
                return start, start + len(value)
        offset += len(line) + 1
    return None


def find_citations(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((m.start(), m.end()) for m in _CITATION.finditer(text))
