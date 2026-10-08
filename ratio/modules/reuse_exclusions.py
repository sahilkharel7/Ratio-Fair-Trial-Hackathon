"""Module 3, step 1: which judgment passages are the court's own reasoning.

Legitimate quotation is excluded before reuse is scored (settings.reuse holds the patterns):
  header_or_signature    the caption block and the signature
  statute_quote          "Article 214(2) of the Penal Code provides: ..." (a marker plus a quotation or citation)
  charge_recital         "The accused is charged with ..."
  party_position         "The prosecution argues that ...", carried to the following sentences of the
                         same paragraph until the court speaks in its own voice ("The Court finds ...")
  non_reasoning_section  procedural history, submissions, disposition (unknown headings count as reasoning)
A party position the court endorses ("rightly", "the Court agrees") is the court's reasoning.
Headings are structure: neither reasoning nor excluded.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ratio.config import ReuseSettings
from ratio.results import ExclusionReason
from ratio.schema import Citation, Passage

_NUMBERING = re.compile(r"^\s*(?:[IVXLC]+|\d+|[A-Z])\.\s+")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_QUOTATION = re.compile(r"[\"“”«»]")


def _alternatives(patterns: Sequence[str]) -> str:
    return "|".join(f"(?:{pattern})" for pattern in patterns)


def _words(patterns: Sequence[str]) -> re.Pattern[str]:
    """Any of the patterns, not inside a longer word."""
    return re.compile(rf"(?<!\w)(?:{_alternatives(patterns)})(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class PassageRules:
    non_reasoning_headings: tuple[str, ...]
    statute: re.Pattern[str]
    recital: re.Pattern[str]
    attribution: re.Pattern[str]
    endorsement: re.Pattern[str]
    court_voice: re.Pattern[str]


def compile_rules(settings: ReuseSettings) -> PassageRules:
    party = _alternatives(settings.party_terms)
    verbs = _alternatives(settings.attribution_verbs)
    attribution = rf"(?<!\w)(?:{party})(?:\s+\w+){{0,2}}?\s+(?:{verbs})(?!\w)"
    if settings.attribution_prefixes:
        attribution += rf"|(?<!\w)(?:{_alternatives(settings.attribution_prefixes)})\s+(?:{party})(?!\w)"
    return PassageRules(
        non_reasoning_headings=tuple(heading.lower() for heading in settings.non_reasoning_headings),
        # Statute markers end in ":" or a word, so only their start is anchored.
        statute=re.compile(rf"(?<!\w)(?:{_alternatives(settings.statute_markers)})", re.IGNORECASE),
        recital=_words(settings.recital_markers),
        attribution=re.compile(attribution, re.IGNORECASE),
        endorsement=_words(settings.endorsement_words),
        court_voice=_words(settings.court_voice_markers),
    )


def heading_title(section: str | None) -> str:
    return _NUMBERING.sub("", section or "").strip().lower()


def is_statute_quote(passage: Passage, citations: Sequence[Citation], rules: PassageRules) -> bool:
    if not rules.statute.search(passage.span.text):
        return False
    return bool(_QUOTATION.search(passage.span.text)) or any(passage.span.contains(c.span) for c in citations)


def _paragraph_break(text: str, previous: Passage | None, passage: Passage) -> bool:
    return previous is None or bool(_PARAGRAPH_BREAK.search(text, previous.span.end, passage.span.start))


def classify(
    passages: Sequence[Passage], text: str, citations: Sequence[Citation], rules: PassageRules
) -> dict[str, ExclusionReason | None]:
    """Passage id -> exclusion reason, or None for the court's own reasoning. Passages must be one
    document's, in document order; ``text`` is that document's text. Headings are left out."""
    reasons: dict[str, ExclusionReason | None] = {}
    carrying = False
    previous: Passage | None = None
    for passage in passages:
        if passage.kind == "heading":
            carrying, previous = False, None
            continue
        if _paragraph_break(text, previous, passage):
            carrying = False
        previous = passage
        sentence = passage.span.text
        if passage.kind in ("header", "signature"):
            reasons[passage.id] = "header_or_signature"
            continue
        if is_statute_quote(passage, citations, rules):
            reasons[passage.id] = "statute_quote"
            continue
        if rules.recital.search(sentence):
            reasons[passage.id] = "charge_recital"
            continue
        if rules.court_voice.search(sentence) or rules.endorsement.search(sentence):
            carrying = False  # the court speaks, or adopts the position as its own
        elif rules.attribution.search(sentence):
            carrying = True
        if carrying:
            reasons[passage.id] = "party_position"
        elif any(heading in heading_title(passage.section) for heading in rules.non_reasoning_headings):
            reasons[passage.id] = "non_reasoning_section"
        else:
            reasons[passage.id] = None
    return reasons
