"""Render the reviewed message templates and filter model-generated notes (hard rule 3).

Flag messages never contain model free text. A model's rationale may be shown only as an
"unverified" note, after any term that characterises a person has been removed.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Iterable

from ratio.config import Messages

REMOVED = "[removed: characterisation of a person]"
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"))


def render(messages: Messages, key: str, **values: object) -> str:
    """Fill a template; raises KeyError for an unknown template or a missing placeholder value."""
    return messages.templates[key].format(**values)


def note(messages: Messages, key: str) -> str:
    return messages.notes[key]


@functools.lru_cache(maxsize=8)
def _block_pattern(block_list: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"(?<![\w-])(?:" + "|".join(f"(?:{term})" for term in block_list) + r")(?![\w-])", re.IGNORECASE)


def _normalise(text: str) -> str:
    """NFKC (folds look-alike letters) and drop invisible characters that could hide a word."""
    return unicodedata.normalize("NFKC", text).translate(_INVISIBLE)


def contains_blocked_term(text: str, block_list: Iterable[str]) -> bool:
    return bool(_block_pattern(tuple(block_list)).search(_normalise(text)))


def filter_model_note(text: str | None, block_list: Iterable[str]) -> str | None:
    """Return the note with blocked terms replaced, or None when there is nothing to show."""
    if text is None or not text.strip():
        return None
    return _block_pattern(tuple(block_list)).sub(REMOVED, _normalise(text).strip())
