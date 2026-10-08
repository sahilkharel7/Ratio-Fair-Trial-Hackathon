"""Locate model quotes in the source text.

A quote is only used to find text. The evidence a module classifies and shows is always the
source sentence containing it (``sentence_bounds``), never the model's own wording. Order of
attempts: exact (inside the chunk window first), whitespace/quote-normalised, then fuzzy for
long quotes whose digits match exactly. Ambiguous or unmatched quotes return None.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass

from rapidfuzz import fuzz

from ratio.extraction.segment import sentence_spans
from ratio.schema import MatchKind

PLACEHOLDERS = frozenset({"", "unknown", "null", "none", "n/a", "na", "not stated", "unspecified", "-"})
_NEGATIONS = frozenset(
    {"no", "not", "never", "without", "denied", "refused", "absent", "unable", "nor", "none", "nobody", "nothing", "cannot"}
)
_WORD = re.compile(r"[A-Za-z']+")
_DIGITS = re.compile(r"\d+")
_CANONICAL = str.maketrans(
    {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", " ": " "}
)


@dataclass(frozen=True)
class Located:
    start: int
    end: int
    match: MatchKind
    score: float


def find_all(text: str, quote: str) -> tuple[int, ...]:
    if not quote:
        return ()
    positions: list[int] = []
    index = text.find(quote)
    while index != -1:
        positions.append(index)
        index = text.find(quote, index + 1)
    return tuple(positions)


def find_unique(text: str, quote: str) -> int | None:
    positions = find_all(text, quote)
    return positions[0] if len(positions) == 1 else None


def _clean_quote(quote: str) -> str:
    cleaned = quote.strip().strip("\"'“”‘’").strip()
    return cleaned.rstrip(".,;:").strip()


def _pick(
    positions: tuple[int, ...], length: int, window: tuple[int, int] | None, used: Collection[tuple[int, int]]
) -> int | None:
    free = [p for p in positions if (p, p + length) not in used]
    if window is not None:
        inside = [p for p in free if window[0] <= p and p + length <= window[1]]
        if inside:
            return inside[0]
    return free[0] if len(free) == 1 else None


def _canonical(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs, straighten quotes/dashes and fold Unicode (ligatures, accents,
    soft hyphens), keeping a map from each canonical character to its original offset."""
    chars: list[str] = []
    origin: list[int] = []
    previous_space = False
    for index, original in enumerate(text.translate(_CANONICAL)):
        for char in unicodedata.normalize("NFKD", original):
            if unicodedata.combining(char) or char == "\u00ad":
                continue
            if char.isspace():
                if previous_space:
                    continue
                char, previous_space = " ", True
            else:
                previous_space = False
            chars.append(char)
            origin.append(index)
    return "".join(chars), origin


def _snap_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    while 0 < start < len(text) and text[start - 1].isalnum() and text[start].isalnum():
        start -= 1
    while 0 < end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end += 1
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _fuzzy(text: str, quote: str, region: tuple[int, int] | None, threshold: float) -> Located | None:
    low, high = region if region is not None else (0, len(text))
    result = fuzz.partial_ratio_alignment(quote, text[low:high], score_cutoff=threshold)
    if result is None:
        return None
    start, end = _snap_to_words(text, low + result.dest_start, low + result.dest_end)
    while end > start and text[end - 1] in ".,;:":
        end -= 1
    if end <= start or _DIGITS.findall(quote) != _DIGITS.findall(text[start:end]):
        return None
    return Located(start, end, "fuzzy", float(result.score))


def locate_quote(
    text: str,
    quote: str | None,
    *,
    window: tuple[int, int] | None = None,
    used: Collection[tuple[int, int]] = (),
    min_chars: int = 12,
    fuzzy_min_chars: int = 30,
    fuzzy_threshold: float = 90.0,
) -> Located | None:
    if quote is None:
        return None
    cleaned = _clean_quote(quote)
    if len(cleaned) < min_chars or cleaned.lower() in PLACEHOLDERS:
        return None

    position = _pick(find_all(text, cleaned), len(cleaned), window, used)
    if position is not None:
        return Located(position, position + len(cleaned), "exact", 100.0)

    canonical_text, origin = _canonical(text)
    canonical_quote, _ = _canonical(cleaned)
    canonical_window = None
    if window is not None:
        canonical_window = (bisect.bisect_left(origin, window[0]), bisect.bisect_left(origin, window[1]))
    occurrences = find_all(canonical_text, canonical_quote)
    found = _pick(occurrences, len(canonical_quote), canonical_window, ())
    if found is not None:
        start, end = origin[found], origin[found + len(canonical_quote) - 1] + 1
        return Located(start, end, "normalized", 100.0)
    if len(occurrences) > 1:
        return None  # the quote occurs several times: never guess which one is meant

    if len(cleaned) >= fuzzy_min_chars:  # only inside the window: the model saw nothing else
        located = _fuzzy(text, cleaned, window, fuzzy_threshold)
        if located is not None and (located.start, located.end) not in used:
            return located
    return None


def sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand [start, end) to the full sentence(s) it overlaps."""
    overlapping = [(s, e) for s, e in sentence_spans(text) if s < end and start < e]
    if not overlapping:
        return _snap_to_words(text, start, end)
    return min(s for s, _ in overlapping), max(e for _, e in overlapping)


_NUMBER_ABBREVIATION = re.compile(r"\bno\.(?=\s+\S)|\bno\s+(?=\d)", re.IGNORECASE)  # "Decision No. 45", "Law No 12"


def _negation_words(text: str) -> set[str]:
    text = _NUMBER_ABBREVIATION.sub(" ", text.translate(_CANONICAL))
    words = {word.lower() for word in _WORD.findall(text)}
    return {word for word in words if word in _NEGATIONS or word.endswith("n't")}


def negation_mismatch(sentence: str, quote: str) -> bool:
    """True when the sentence holds a negation the quote dropped (the quote may reverse its meaning)."""
    return bool(_negation_words(sentence) - _negation_words(quote))
