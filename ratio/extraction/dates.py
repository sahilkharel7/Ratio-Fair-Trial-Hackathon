"""Parse dates in code with dateparser. The model's date is stored for comparison only.

Relative and year-less phrases resolve against a fixed base date (the document's date), never
the wall clock, so results are reproducible; callers mark such dates for review.
"""

from __future__ import annotations

import datetime as dt
import functools
import re
from dataclasses import dataclass

from ratio.schema import DatePrecision

_PLACEHOLDERS = frozenset({"", "unknown", "null", "none", "n/a", "na", "not stated", "unspecified", "-"})
_YEAR = re.compile(r"\b(?:1[89]\d{2}|2\d{3})\b")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_FIRST = re.compile(r"^\d{4}[-/.]\d{1,2}(?:[-/.]\d{1,2})?(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?$")
_MONTH_YEAR_NUMERIC = re.compile(r"^\d{1,2}[-/.]\d{4}$")
_RANGE = re.compile(r"\d\s*(?:-|\u2013|\u2014|to|and|until)\s*\d", re.IGNORECASE)
MAX_DATE_TEXT = 64
_SENTINEL_BASE = dt.datetime(2000, 1, 1)
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
_WRITTEN_DATE = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?(?:\s+day)?(?:\s+of)?\s+{_MONTH},?\s+\d{{4}}\b"  # 14 February 2025, 14th day of February, 2025
    rf"|\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"  # February 14, 2025
    r"|\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b"  # 14/02/2025, 14.02.2025, 14-02-2025
    r"|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",  # 2025-02-14, 2025/02/14
    re.IGNORECASE,
)
_NUMERIC_DATE = re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{4}$")
_ORDINAL_OF = re.compile(r"^(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+)?of\s+", re.IGNORECASE)
_DAY_AND_MONTH = re.compile(rf"{_MONTH}|\b\d{{1,2}}[./-]\d{{1,2}}\b", re.IGNORECASE)
_ALNUM_EDGE = "0-9A-Za-z"
_PERIOD_PRECISION: dict[str, DatePrecision] = {
    "time": "datetime",
    "day": "date",
    "week": "date",
    "month": "month",
    "year": "year",
}


@dataclass(frozen=True)
class ParsedDate:
    value: dt.datetime
    precision: DatePrecision


@functools.lru_cache(maxsize=64)
def _parser(base: dt.datetime, order: str = "DMY"):  # noqa: ANN202 - dateparser has no public type for this
    from dateparser.date import DateDataParser

    return DateDataParser(
        languages=["en"],
        settings={
            "DATE_ORDER": order,
            "PREFER_DAY_OF_MONTH": "first",
            "PREFER_MONTH_OF_YEAR": "first",
            "PREFER_DATES_FROM": "past",
            "RETURN_TIME_AS_PERIOD": True,
            "RETURN_AS_TIMEZONE_AWARE": False,
            "RELATIVE_BASE": base,
        },
    )


def parse_date_text(text: str | None, *, base: dt.datetime | None = None) -> ParsedDate | None:
    """Parse one date expression copied from the source.

    Returns None for placeholders, non-dates, ranges ("9-12 February 2025"), numeric month/year
    forms ("02/2025"), text without a written year when no base date is known (so no sentinel
    year can leak into a record), and year-less text that names no day and month (a time, a
    duration or a weekday). ISO-style dates are read year first; ordinals like "the 14th day
    of February, 2025" are accepted.
    """
    if text is None:
        return None
    cleaned = _ORDINAL_OF.sub(r"\1 ", " ".join(text.split()))  # "the 14th day of February" -> "14 February"
    if cleaned.lower() in _PLACEHOLDERS or len(cleaned) > MAX_DATE_TEXT:
        return None
    year_first = bool(_YEAR_FIRST.match(cleaned))
    numeric = bool(_NUMERIC_DATE.match(cleaned))
    if not (year_first or numeric) and (_RANGE.search(cleaned) or _MONTH_YEAR_NUMERIC.match(cleaned)):
        return None
    if not has_explicit_year(cleaned) and (base is None or not _DAY_AND_MONTH.search(cleaned)):
        return None  # no year and nothing to anchor a day: a time ("10:10"), a duration or a weekday
    order = "YMD" if year_first else "DMY"
    data = _parser(base or _SENTINEL_BASE, order).get_date_data(cleaned)
    if data is None or data.date_obj is None:
        return None
    precision = _PERIOD_PRECISION.get(data.period or "", "unknown")
    return ParsedDate(value=data.date_obj.replace(tzinfo=None), precision=precision)


def find_written_dates(text: str, start: int = 0, end: int | None = None) -> tuple[tuple[int, int], ...]:
    """Spans of full written dates (day, month and year) in text[start:end]."""
    return tuple((m.start(), m.end()) for m in _WRITTEN_DATE.finditer(text, start, len(text) if end is None else end))


def locate_date_text(text: str, start: int, end: int, candidate: str) -> tuple[int, int] | None:
    """Offsets of ``candidate`` in text[start:end] as a whole expression (never inside a longer
    word or number), allowing any whitespace between its words (line breaks, non-breaking spaces)."""
    words = candidate.split()
    if not words:
        return None
    body = r"\s+".join(re.escape(word) for word in words)
    pattern = re.compile(rf"(?<![{_ALNUM_EDGE}]){body}(?![{_ALNUM_EDGE}])", re.IGNORECASE)
    match = pattern.search(text, start, end)
    return (match.start(), match.end()) if match else None


def has_explicit_year(text: str | None) -> bool:
    return bool(text and _YEAR.search(text))


def parse_iso_date(text: str | None) -> dt.date | None:
    """The model's ISO date (YYYY-MM-DD), or None for placeholders and anything else."""
    if text is None or not _ISO_DATE.match(text.strip()):
        return None
    try:
        return dt.date.fromisoformat(text.strip())
    except ValueError:
        return None
