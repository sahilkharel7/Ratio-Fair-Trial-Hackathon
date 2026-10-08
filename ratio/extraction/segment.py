"""Deterministic segmentation of document text into sentences with exact character offsets.

Blocks are separated by blank lines; heading lines also start a new block, so text without
blank lines (PDF extraction) keeps its structure. The document opens with its caption: the
leading lines that are labels ("Presiding Judge: ...") or short lines without sentence endings,
up to the first numbered heading or sentence. A final block that starts with "(signed)" is the
signature. A heading ("III. STATEMENT OF FACTS", "4.1 Assessment", "Submissions of the Parties")
starts a section. Headings in the style of the document's first heading, and unnumbered capitals,
are chapters; other styles are sub-headings inside the current chapter ("A. The prosecution"
under "III. SUBMISSIONS OF THE PARTIES"). Every other line is body text, split into sentences.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from ratio.schema import PassageKind

_LINE = re.compile(r"[^\n]+")
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_NUMBERING = re.compile(r"^(?:(?P<roman>[IVXLC]+)\.|(?P<decimal>\d+(?:\.\d+)+)\.?|(?P<digit>\d+)\.|(?P<letter>[A-Z])\.|(?P<paren>\([a-z0-9]{1,4}\)))\s+(?=\S)")
_SIGNATURE = re.compile(r"^\(signed\)|^signed[:,]", re.IGNORECASE)
_LABEL_LINE = re.compile(r"^[A-Z][A-Za-z .()/-]{0,40}:\s+\S")
MAX_CAPTION_LINE = 120
MAX_HEADING_CHARS = 90
MAX_HEADING_WORDS = 12
MAX_QUOTATION_CHARS = 1500  # a quotation mark with no partner this close is a stray, not a quotation
_BOUNDARY = re.compile(r"[.!?][\"'”’»)\]]*(?=\s+[\"'“‘«(\[]?[A-Z0-9])")
_BULLET = re.compile(r"\n[ \t]*(?:[-*•▪◦]|\d{1,2}[.)])[ \t]+")  # a list item on its own line
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "hon", "st", "no", "nos", "art", "arts", "para", "paras",
        "p", "pp", "sec", "cf", "etc", "e.g", "i.e", "v", "vs", "approx", "jan", "feb", "mar",
        "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "ibid",
    }
)  # fmt: skip
_SMALL_WORDS = frozenset({"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"})
_CITATION = re.compile(
    r"(?:\b(?:Article|Art\.|Section|Sec\.|Rule|Regulation|Paragraph|Para\.|s\.)|§)\s*\d+[A-Za-z]?(?:\(\d+\))*"
    r"(?:\s+of\s+the\s+(?:[A-Z][\w-]*\s+){0,4}?(?:Code|Act|Law|Constitution|Covenant|Rules|Regulations))?"
)
_QUOTE_PAIRS = (("“", "”"), ("«", "»"), ("„", "“"))
_NUMBERED_STYLES = frozenset({"roman", "decimal", "digit", "letter", "paren"})
_CLOSING_SINGLE = re.compile(r"’(?![A-Za-z])|(?<![A-Za-z])’")


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    kind: PassageKind
    section: str | None
    chapter: str | None = None


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _lines(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    spans = (_trim(text, m.start(), m.end()) for m in _LINE.finditer(text, start, len(text) if end is None else end))
    return [(s, e) for s, e in spans if e > s]


def _caption_line(line: str) -> bool:
    """A caption line: 'Label: value', or a short line with no sentence ending that is not a numbered heading."""
    if _LABEL_LINE.match(line):
        return True
    return len(line) <= MAX_CAPTION_LINE and line[-1] not in ".!?" and not _NUMBERING.match(line)


def _title_case(words: list[str]) -> bool:
    significant = [w for w in words if w.lower() not in _SMALL_WORDS and any(c.isalpha() for c in w)]
    return bool(significant) and sum(w[0].isupper() for w in significant) >= max(1, round(0.75 * len(significant)))


def heading_style(line: str) -> str | None:
    """'roman', 'decimal', 'digit', 'letter', 'paren', 'caps' or 'title' for a heading line, else None."""
    line = line.strip()
    if not line or "\n" in line or len(line) > MAX_HEADING_CHARS or line[-1] in ";!?":
        return None
    numbering = _NUMBERING.match(line)
    title = line[numbering.end() :] if numbering else line
    words = title.split()
    letters = [c for c in title if c.isalpha()]
    caps = len(letters) > 2 and all(c.isupper() for c in letters)
    if len(words) > MAX_HEADING_WORDS:
        return None
    if line[-1] in ".,:" and not (caps or (numbering and _title_case(words))):
        return None  # "12. The Court notes that the accused was absent." is a numbered paragraph
    if numbering:
        return next(name for name, value in numbering.groupdict().items() if value)
    if caps:
        return "caps"
    return "title" if len(words) <= 10 and line[-1] not in ".," and _title_case(words) and len(letters) > 3 else None


def _next_letter(previous: str, title: str) -> bool:
    """Whether ``title`` is numbered with the letter after the one numbering ``previous`` (B. -> C.)."""
    return len(previous) > 1 and len(title) > 1 and previous[1] == "." and title[1] == "." and ord(title[0]) == ord(previous[0]) + 1


def _is_heading(block: str) -> bool:
    return heading_style(block) is not None


def _is_abbreviation(text: str, dot: int, sentence_start: int) -> bool:
    if text[dot] != ".":
        return False
    begin = dot
    while begin > sentence_start and (text[begin - 1].isalnum() or text[begin - 1] == "."):
        begin -= 1
    word = text[begin:dot].lower()
    if not word:
        return False
    paragraph_number = begin == sentence_start and (word.isdigit() or re.fullmatch(r"[ivxlc]+", word) is not None)
    return (
        word in _ABBREVIATIONS
        or paragraph_number  # "12. The Court notes ..." is one sentence
        or (len(word) == 1 and word.isalpha())
        or re.fullmatch(r"(?:[a-z]\.)+[a-z]", word) is not None
    )


def _inside_quotation(text: str, start: int, stop: int, limit: int) -> bool:
    """Whether [start, stop) leaves a quotation open that closes later in the block (within reach)."""
    chunk = text[start:stop]
    horizon = text[stop : min(limit, stop + MAX_QUOTATION_CHARS)]
    if chunk.count('"') % 2 == 1 and '"' in horizon:
        return True
    if chunk.count("‘") > len(_CLOSING_SINGLE.findall(chunk)) and _CLOSING_SINGLE.search(horizon):
        return True  # ‘…’ quotations; a ’ between two letters is an apostrophe ("accused’s")
    return any(chunk.count(open_) > chunk.count(close) and close in horizon for open_, close in _QUOTE_PAIRS)


def split_sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence spans inside [start, end), skipping abbreviations, initials and quoted text."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in _BOUNDARY.finditer(text, start, end):
        stop = match.end()
        if _is_abbreviation(text, match.start(), cursor) or _inside_quotation(text, cursor, stop, end):
            continue
        spans.append(_trim(text, cursor, stop))
        cursor = stop
    spans.append(_trim(text, cursor, end))
    return [span for span in spans if span[1] > span[0]]


def _list_items(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """A block split at its list items ("- ...", "2. ..." on new lines); a plain paragraph is one item."""
    cuts = [m.start() for m in _BULLET.finditer(text, start, end)]
    bounds = [start, *cuts, end]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def _blocks(text: str) -> list[tuple[int, int]]:
    bounds = [0, *(m.start() for m in _BLANK_LINE.finditer(text)), len(text)]
    spans = (_trim(text, a, b) for a, b in zip(bounds, bounds[1:]))
    return [(s, e) for s, e in spans if e > s]


def caption_lines(text: str) -> list[tuple[int, int]]:
    """The document's caption: leading lines of the first block that are labels or short unpunctuated lines."""
    blocks = _blocks(text)
    if not blocks:
        return []
    caption = []
    for start, end in _lines(text, *blocks[0]):
        if not _caption_line(text[start:end]):
            break
        caption.append((start, end))
    if len(blocks) == 1 and len(caption) == len(_lines(text, *blocks[0])):
        return []  # a one-block document of short lines is content, not a caption
    return caption


def _heading_line(text: str, lines: list[tuple[int, int]], index: int) -> bool:
    """A heading inside a multi-line block: short, after a finished line, before a line that starts anew."""
    start, end = lines[index]
    if heading_style(text[start:end]) is None or end - start > 70:
        return False
    previous_ends = index == 0 or text[lines[index - 1][1] - 1] in ".!?:" or heading_style(text[slice(*lines[index - 1])]) is not None
    following = text[lines[index + 1][0]] if index + 1 < len(lines) else ""
    return previous_ends and (not following or following.isupper() or following.isdigit() or following in "\"“«('‘")


def _pieces(text: str, start: int, end: int) -> list[tuple[int, int, bool]]:
    """A block cut at its heading lines: (start, end, is_heading) pieces."""
    lines = _lines(text, start, end)
    if len(lines) == 1:
        return [(start, end, heading_style(text[start:end]) is not None)]
    pieces: list[tuple[int, int, bool]] = []
    body_start = None
    for index, (s, e) in enumerate(lines):
        if _heading_line(text, lines, index):
            if body_start is not None:
                pieces.append((body_start, lines[index - 1][1], False))
                body_start = None
            pieces.append((s, e, True))
        elif body_start is None:
            body_start = s
    if body_start is not None:
        pieces.append((body_start, lines[-1][1], False))
    return pieces


@functools.lru_cache(maxsize=128)
def segment_document(text: str) -> tuple[Segment, ...]:
    blocks = _blocks(text)
    caption = caption_lines(text)
    segments = [Segment(s, e, "header", None) for s, e in caption]
    if caption and blocks:  # whatever follows the caption in the first block is content
        rest = _trim(text, caption[-1][1], blocks[0][1])
        blocks = ([rest] if rest[1] > rest[0] else []) + blocks[1:]
    section: str | None = None
    chapter: str | None = None
    chapter_style: str | None = None  # the first numbered style in the document marks its chapters
    previous_style: str | None = None
    previous_title = ""
    for index, (start, end) in enumerate(blocks):
        if index == len(blocks) - 1 and _SIGNATURE.match(text[start:end]):
            segments.extend(Segment(s, e, "signature", section, chapter) for s, e in _lines(text, start, end))
            continue
        for piece_start, piece_end, is_heading in _pieces(text, start, end):
            if is_heading:
                title = text[piece_start:piece_end].strip()
                style = heading_style(title)
                if style == "roman" and previous_style == "letter" and _next_letter(previous_title, title):
                    style = "letter"  # "C." after "B." is a letter, not the numeral 100
                if style in _NUMBERED_STYLES and chapter_style is None:
                    chapter_style = style
                if style == "caps" or chapter_style is None or style == chapter_style:
                    chapter = title
                section, previous_style, previous_title = title, style, title
                segments.append(Segment(piece_start, piece_end, "heading", section, chapter))
                continue
            for item_start, item_end in _list_items(text, piece_start, piece_end):
                segments.extend(Segment(s, e, "body", section, chapter) for s, e in split_sentences(text, item_start, item_end))
    return tuple(segments)


def sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((segment.start, segment.end) for segment in segment_document(text))


def header_value(text: str, label: str) -> tuple[int, int] | None:
    """Offsets of the value of a ``Label: value`` line in the document's caption."""
    prefix = label.lower() + ":"
    for start, end in caption_lines(text):
        line = text[start:end]
        if line.lower().startswith(prefix):
            value_start, value_end = _trim(text, start + len(prefix), end)
            if value_end > value_start:
                return value_start, value_end
    return None


def find_citations(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((m.start(), m.end()) for m in _CITATION.finditer(text))
