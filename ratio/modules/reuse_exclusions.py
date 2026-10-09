"""Module 3, step 1: which judgment passages are the court's own reasoning.

Legitimate quotation is excluded before reuse is scored (settings.reuse holds the patterns):
  header_or_signature    the caption and the signature
  statute_quote          "Article 214(2) of the Penal Code provides: ..." (a marker plus a quotation or
                         citation), and the quoted paragraph after "provides:" or "reads as follows:"
  party_position         a party's reported case: "The prosecution argues that ...", "According to the
                         defence, ...", "It was argued by the defence ...", "The prosecution's case is
                         ...". Sentences that go on reporting it ("It further submits ...") are carried;
                         any other sentence ends the carry. "The defence rightly argues ..." and
                         sentences in the court's voice are reasoning.
  charge_recital         "The accused is charged with ...", "According to the indictment, ..."
  non_reasoning_section  chapters such as procedural history, submissions or the disposition
Headings are structure: neither reasoning nor excluded. Patterns run on the sentence with its
whitespace collapsed, so a line break or a non-breaking space never hides a marker.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ratio.config import ReuseSettings
from ratio.results import ExclusionReason
from ratio.schema import PARAGRAPH_BREAK, Citation, Passage

_NUMBERING = re.compile(r"^\s*(?:[IVXLC]+|\d+(?:\.\d+)*|[A-Z]|\([a-z0-9]{1,4}\))[.)]?\s+")
_QUOTATION = re.compile(r"[\"“”«»„]|(?:^|[\s(:])[‘'](?=\w)")
_QUOTED_OPENING = re.compile(r"^(?:[\"“«„‘']|\([a-z0-9]{1,4}\)\s)", re.IGNORECASE)
_INTRODUCES_QUOTE = re.compile(r"(?::|\bas follows:?)\s*[\"“«„‘']?$", re.IGNORECASE)
_LINKS = r"(?:\s+(?:further|also|again|additionally|essentially|then|now|repeatedly|next|subsequently|has|had|have|did|does))*"
_ASIDE = r"(?:\s*,[^,]{1,80},)?"  # "The prosecution, in its closing speech, argued ..."


def _alternatives(patterns: Sequence[str]) -> str:
    return "|".join(f"(?:{pattern})" for pattern in patterns)


def normalise(text: str) -> str:
    """Whitespace collapsed and curly apostrophes straightened: what every pattern is matched on."""
    return " ".join(text.replace("’", "'").split())


@dataclass(frozen=True)
class PassageRules:
    non_reasoning: re.Pattern[str] | None
    reasoning_words: re.Pattern[str] | None
    statute: re.Pattern[str]
    recital: re.Pattern[str]
    attribution: re.Pattern[str]  # a party opens the sentence and reports its case
    endorsed: re.Pattern[str]  # "The defence rightly argues ...": the court adopts it
    reporting: re.Pattern[str]  # "The Court notes that the defence argues ...": still the party's case
    continuation: re.Pattern[str]  # "It further submits ...": the report goes on
    court_voice: re.Pattern[str]


def compile_rules(settings: ReuseSettings) -> PassageRules:
    flags = re.IGNORECASE
    party = _alternatives(settings.party_terms)
    verbs = _alternatives(settings.attribution_verbs)
    openers = rf"(?:(?:{_alternatives(settings.sentence_openers)})\s*,?\s*)?" if settings.sentence_openers else ""
    start = rf"^\W*(?:\(?\d+\)?[.)]?\s*)?{openers}"  # a paragraph number and a connective may come first
    not_party = rf"(?!\s+(?:{_alternatives(settings.not_a_party_after)})\b)" if settings.not_a_party_after else ""
    subject = rf"(?:{party}){not_party}{_ASIDE}{_LINKS}"
    forms = [rf"{start}{subject}\s+(?:{verbs})\b"]
    if settings.attribution_prefixes:
        forms.append(rf"{start}(?:{_alternatives(settings.attribution_prefixes)})\s+(?:{party})\b")
    forms.append(rf"\b(?:is|was|were|has been|had been)\s+(?:argued|submitted|contended|claimed|alleged|maintained|asserted|stated)\s+(?:by|on behalf of)\s+(?:{party})\b")
    if settings.attribution_possessives:
        possessive = _alternatives(settings.attribution_possessives)
        forms.append(rf"{start}(?:{party})'s\s+(?:{possessive})\b")  # "The prosecution's case is that ..."
        forms.append(rf"{start}in\s+(?:{party})'s\s+(?:{possessive}|opinion)\b")  # "In the prosecution's view, ..."
    headings = settings.non_reasoning_headings
    return PassageRules(
        non_reasoning=re.compile(rf"^(?:the\s+)?(?:{_alternatives(headings)})\b", flags) if headings else None,
        reasoning_words=re.compile(rf"\b(?:{_alternatives(settings.reasoning_heading_words)})\b", flags) if settings.reasoning_heading_words else None,
        statute=re.compile(rf"(?<!\w)(?:{_alternatives(settings.statute_markers)})", flags),
        recital=re.compile(rf"(?<!\w)(?:{_alternatives(settings.recital_markers)})(?!\w)", flags),
        attribution=re.compile("|".join(f"(?:{form})" for form in forms), flags),
        endorsed=re.compile(rf"{start}{subject}\s+(?:{_alternatives(settings.endorsement_words)})\s+(?:{verbs})\b", flags),
        reporting=re.compile(rf"{start}(?:the|this)\s+(?:court|chamber|tribunal)\s+(?:notes|observes|considers|has considered|recalls)\s+that\s+{subject}\s+(?:{verbs})\b", flags),
        continuation=re.compile(
            rf"{start}(?:it|they|he|she|counsel|{party}){_ASIDE}{_LINKS}\s+(?:{verbs})\b"
            r"|\bin\s+(?:its|their|his|her)\s+(?:view|submission|opinion)\b|^\W*according\s+to\s+(?:it|them|him|her)\b",
            flags,
        ),
        court_voice=re.compile(rf"(?<!\w)(?:{_alternatives(settings.court_voice_markers)})", flags),
    )


def heading_title(section: str | None) -> str:
    return normalise(_NUMBERING.sub("", section or "")).strip().lower()


def non_reasoning_chapter(title: str, rules: PassageRules) -> bool:
    """A chapter whose title starts like "Submissions ..." unless it goes on to name the court's reasons."""
    match = rules.non_reasoning.match(title) if rules.non_reasoning else None
    if match is None:
        return False
    return not (rules.reasoning_words and rules.reasoning_words.search(title[match.end() :]))


def is_statute_quote(passage: Passage, citations: Sequence[Citation], rules: PassageRules) -> bool:
    sentence = normalise(passage.span.text)
    if not rules.statute.search(sentence):
        return False
    return bool(_QUOTATION.search(sentence)) or any(passage.span.contains(c.span) for c in citations)


def _paragraph_break(text: str, previous: Passage | None, passage: Passage) -> bool:
    return previous is None or bool(PARAGRAPH_BREAK.search(text, previous.span.end, passage.span.start))


@dataclass
class _State:
    """What the previous sentences leave open: a reported party position, or a quoted provision."""

    carrying: bool = False
    quote_pending: bool = False  # the last sentence introduced a quotation ("... provides:")
    in_quote: bool = False  # inside the quoted provision's paragraph(s)


def _quoted_provision(passage: Passage, sentence: str, new_paragraph: bool, state: _State) -> bool:
    """Whether this passage is the text of a provision quoted as a block after its marker sentence."""
    if state.quote_pending:
        state.quote_pending, state.in_quote = False, True
        return True
    if state.in_quote and (not new_paragraph or _QUOTED_OPENING.match(sentence)):
        return True
    state.in_quote = False
    return False


def _reason(passage: Passage, sentence: str, citations: Sequence[Citation], rules: PassageRules, state: _State) -> ExclusionReason | None:
    if is_statute_quote(passage, citations, rules):
        state.carrying, state.quote_pending = False, bool(_INTRODUCES_QUOTE.search(sentence))
        return "statute_quote"
    if rules.reporting.search(sentence):
        state.carrying = True
        return "party_position"
    endorsed = bool(rules.endorsed.search(sentence))
    if not endorsed and rules.attribution.search(sentence):
        state.carrying = True
        return "party_position"
    court = endorsed or bool(rules.court_voice.search(sentence))
    if state.carrying and not court and rules.continuation.search(sentence):
        return "party_position"
    state.carrying = False
    if not court and rules.recital.search(sentence):
        return "charge_recital"
    if non_reasoning_chapter(heading_title(passage.chapter or passage.section), rules):
        return "non_reasoning_section"
    return None


def classify(
    passages: Sequence[Passage], text: str, citations: Sequence[Citation], rules: PassageRules
) -> dict[str, ExclusionReason | None]:
    """Passage id -> exclusion reason, or None for the court's own reasoning. Passages must be one
    document's, in document order; ``text`` is that document's text. Headings are left out."""
    reasons: dict[str, ExclusionReason | None] = {}
    state = _State()
    previous: Passage | None = None
    for passage in passages:
        if passage.kind == "heading":
            state, previous = _State(), None
            continue
        new_paragraph = _paragraph_break(text, previous, passage)
        previous = passage
        sentence = normalise(passage.span.text)
        if passage.kind in ("header", "signature"):
            reasons[passage.id] = "header_or_signature"
        elif _quoted_provision(passage, sentence, new_paragraph, state):
            reasons[passage.id] = "statute_quote"
        else:
            reasons[passage.id] = _reason(passage, sentence, citations, rules, state)
    return reasons
