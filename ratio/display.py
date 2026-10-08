"""Pure display helpers shared by the modules, the app and the tests (no Streamlit here).

Every piece of document text is HTML-escaped before it is wrapped in markup, so a document can
never inject HTML into the page; highlights only ever wrap exact source offsets.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from dataclasses import dataclass

from ratio.schema import SourceSpan

MARK_PRIORITY = {"span": 0, "verbatim": 1, "paraphrase": 2, "excluded": 3}
# Markdown, LaTeX ($) and Streamlit directives (:red[...], :shortcode:) all start with one of these.
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:=])")


@dataclass(frozen=True)
class Excerpt:
    """A span with the text around it, cut at word boundaries."""

    before: str
    match: str
    after: str
    line: int
    clipped_before: bool
    clipped_after: bool


@dataclass(frozen=True)
class Mark:
    """A range [start, end) in document offsets, rendered with the CSS class ``ratio-<kind>``."""

    start: int
    end: int
    kind: str
    label: str = ""
    title: str = ""


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def excerpt(text: str, span: SourceSpan, context_chars: int) -> Excerpt:
    start = max(0, span.start - context_chars)
    end = min(len(text), span.end + context_chars)
    if start > 0:
        cut = next((i for i in range(start, span.start) if text[i].isspace()), start)
        start = cut + 1 if cut < span.start else start
    if end < len(text):
        cut = next((i for i in range(end - 1, span.end - 1, -1) if text[i].isspace()), end)
        end = cut if cut >= span.end else end
    return Excerpt(
        before=text[start : span.start],
        match=text[span.start : span.end],
        after=text[span.end : end],
        line=line_number(text, span.start),
        clipped_before=start > 0,
        clipped_after=end < len(text),
    )


def excerpt_html(part: Excerpt) -> str:
    before = ("… " if part.clipped_before else "") + html.escape(part.before)
    after = html.escape(part.after) + (" …" if part.clipped_after else "")
    return f'<div class="ratio-doc">{before}<mark class="ratio-span">{html.escape(part.match)}</mark>{after}</div>'


def _merge(marks: Iterable[Mark]) -> list[Mark]:
    """Sort by start; an overlapping mark joins the previous one (labels combine, stronger kind wins)."""
    merged: list[Mark] = []
    for mark in sorted(marks, key=lambda m: (m.start, m.end)):
        if mark.end <= mark.start:
            continue
        if merged and mark.start < merged[-1].end:
            last = merged[-1]
            kind = min(last.kind, mark.kind, key=lambda k: MARK_PRIORITY.get(k, 99))
            labels = [label for label in dict.fromkeys(last.label.split(",") + mark.label.split(",")) if label]
            merged[-1] = Mark(last.start, max(last.end, mark.end), kind, ",".join(labels), last.title or mark.title)
        else:
            merged.append(mark)
    return merged


def _wrap(mark: Mark, inner: str) -> str:
    title = f' title="{html.escape(mark.title)}"' if mark.title else ""
    label = f'<sup class="ratio-label">{html.escape(mark.label)}</sup>' if mark.label else ""
    if mark.kind == "excluded":
        tag = f'<span class="ratio-tag">{html.escape(mark.label)}</span>' if mark.label else ""
        return f'<span class="ratio-excluded"{title}>{inner}</span>{tag}'
    return f'<mark class="ratio-{html.escape(mark.kind)}"{title}>{inner}</mark>{label}'


def marked_html(text: str, marks: Iterable[Mark], offset: int = 0) -> str:
    """Escape ``text`` and wrap the marked ranges. Marks use document offsets; ``offset`` is
    where ``text`` starts in its document (0 for a whole document)."""
    parts: list[str] = []
    cursor = 0
    for mark in _merge(marks):
        start = max(mark.start - offset, cursor)
        end = min(mark.end - offset, len(text))
        if end <= start:
            continue
        parts.append(html.escape(text[cursor:start]))
        parts.append(_wrap(mark, html.escape(text[start:end])))
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return f'<div class="ratio-doc">{"".join(parts)}</div>'


def duration_text(low: float, high: float) -> str:
    if low == high:
        return f"{low:g} hours"
    return f"between {low / 24:g} and {high / 24:g} days ({low:g} to {high:g} hours; dates are day-level)"


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def md_escape(text: str) -> str:
    """Text from a document or a model, made safe to pass to st.markdown: shown literally."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text).replace("\n", "  \n")
