"""Check every model quote against the precedent's stored text (hard rule 2).

A quote is kept only when it is in the text exactly, or after whitespace, quote-mark and Unicode
normalisation; never fuzzily, and never when it occurs more than once (the aligner refuses to
guess). It must be at least MIN_QUOTE_WORDS words long and hold each negation word as often as its
sentence does: "a violation of article 9, since he was not deprived of counsel" cut from "the
Committee does not find a violation of article 9, since he was not deprived of counsel" is dropped,
although it keeps one "not". A finding is stored as its whole sentence, so the shown finding always
carries its own negation; a sentence longer than FINDING_CHARS is cut to a window around the quote
that keeps every negation word before it, and the finding is dropped when they do not fit. Its kind
must fit the document (FINDING_KINDS) and its own words: a violation found is dropped when its sentence
holds a negation word, and no violation when it holds none. It becomes a span of the text, so what
the app shows is the document's own words. Every quote that fails is dropped and counted.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass

from pydantic import ValidationError

from corpus_builder.extract import (
    PROMPT_SHA,
    ExtractedFacet,
    Extraction,
    cache_key,
    merge_facets,
    schema_sha,
)
from corpus_builder.store import BuildStore
from ratio.config import FactPatternTaxonomy
from ratio.extraction.align import locate_quote, sentence_bounds
from ratio.precedent_schema import (
    ALLOWED_FINDINGS,
    FindingKind,
    PrecedentDoc,
    PrecedentFacet,
    PrecedentQuote,
)
from ratio.schema import Frozen, SourceSpan

log = logging.getLogger(__name__)

NEVER_FUZZY = 10**9  # locate_quote tries fuzzy matching only for quotes at least this long
ACCEPTED_MATCHES = frozenset({"exact", "normalized"})
MIN_QUOTE_WORDS = 8  # the prompt asks for 8 to 60 words; a shorter quote cannot stand on its own
FINDING_CHARS = 600  # a finding is shown as its sentence; a longer one is cut to this many characters around the quote
FINDING_KINDS = ALLOWED_FINDINGS  # one contract for builder and runtime
# The kind kept when the model's does not fit the document: valid for every kind, and never shown,
# because the app labels a finding only next to its quote, and such a finding has none.
UNSTATED: FindingKind = "not_examined"
# Words that reverse a clause; any word ending in "n't" counts too. Counted, not just found: a quote
# may keep one "not" of its sentence and cut the other. The verbs matter as much as "not": "the author
# was held for six days" cut from "the State party denies that the author was held for six days"
# states the opposite of what the document says.
NEGATIONS = frozenset({
    "no", "not", "never", "neither", "nor", "without", "cannot", "none", "nobody", "nothing", "absent", "unable",
    "deny", "denies", "denied", "denying",
    "refuse", "refuses", "refused", "refusing",
    "reject", "rejects", "rejected", "rejecting",
    "dismiss", "dismisses", "dismissed", "dismissing",
    "fail", "fails", "failed", "failing",
    "lack", "lacks", "lacked", "lacking",
    "contest", "contests", "contested", "contesting",
    "dispute", "disputes", "disputed", "disputing",
    "refute", "refutes", "refuted", "refuting",
    "decline", "declines", "declined", "declining",
    "untrue", "unsubstantiated", "unfounded",
})  # fmt: skip
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})
_WORD = re.compile(r"[A-Za-z']+")
_NUMBER_ABBREVIATION = re.compile(r"\bno\.(?=\s+\S)|\bno\s+(?=\d)", re.IGNORECASE)  # "Opinion No. 27/2017", "Law No 12"


class StaleVerification(RuntimeError):
    """A verified span no longer matches its document's text: run verify again."""


class VerifyReport(Frozen):
    documents: int  # documents verified
    facets: int  # facets kept
    quotes: int  # quotes kept (facts and findings)
    dropped: int  # quotes dropped
    wrong_kind: int = 0  # findings dropped because their kind does not fit the document (FINDING_KINDS)
    missing: tuple[str, ...]  # no extraction of their current text, prompt and schema; not re-verified


@dataclass(frozen=True)
class Verified:
    facets: tuple[PrecedentFacet, ...]
    dropped: int
    wrong_kind: int = 0


@dataclass(frozen=True)
class _Finding:
    kind: FindingKind
    quote: PrecedentQuote | None
    dropped: int  # the finding's quote, when it failed or its kind did not fit
    wrong_kind: bool


def check_span(doc: PrecedentDoc, span: SourceSpan) -> None:
    """Raise StaleVerification unless the span is this document's text at its offsets."""
    if span.doc_id != doc.doc_id or doc.text[span.start : span.end] != span.text:
        raise StaleVerification(
            f"{doc.id}: a verified quote no longer matches the text; run `python -m corpus_builder extract`, then `verify`"
        )


def _span(doc: PrecedentDoc, start: int, end: int) -> SourceSpan:
    return SourceSpan(doc_id=doc.doc_id, start=start, end=end, text=doc.text[start:end])


def _negation(word: str) -> str | None:
    """The negation a word counts as ("didn't" and "wasn't" both count as "n't"), or None."""
    word = word.strip("'").lower()
    if word.endswith("n't"):
        return "n't"
    return word if word in NEGATIONS else None


def negation_words(text: str) -> Iterator[tuple[int, str]]:
    """(offset, negation) for each negation word of the text, in order; the "No" of "Opinion No. 27"
    and "Law No 12" is a number, not a negation."""
    blank = _NUMBER_ABBREVIATION.sub(lambda match: " " * len(match[0]), text.translate(_APOSTROPHES))  # same offsets
    for match in _WORD.finditer(blank):
        negation = _negation(match[0])
        if negation is not None:
            yield match.start(), negation


def negation_counts(text: str) -> Counter[str]:
    """How often each negation word occurs in the text."""
    return Counter(negation for _, negation in negation_words(text))


def drops_negation(sentence: str, quote: str) -> bool:
    """True when the sentence holds some negation word more often than the quote does: the quote may
    have cut the clause that word negates, and so reverse what the sentence says."""
    return bool(negation_counts(sentence) - negation_counts(quote))


def locate(doc: PrecedentDoc, quote: str) -> PrecedentQuote | None:
    """The quote as a span of the document's text, or None: not found, ambiguous, only fuzzy, shorter
    than MIN_QUOTE_WORDS, or with fewer negations than its sentence (it may reverse the meaning)."""
    located = locate_quote(doc.text, quote, fuzzy_min_chars=NEVER_FUZZY)
    if located is None or located.match not in ACCEPTED_MATCHES:
        return None
    found = doc.text[located.start : located.end]
    if len(found.split()) < MIN_QUOTE_WORDS:
        return None
    low, high = sentence_bounds(doc.text, located.start, located.end)
    if drops_negation(doc.text[low:high], found):
        return None
    return PrecedentQuote(span=_span(doc, located.start, located.end), match=located.match)


def finding_bounds(text: str, start: int, end: int, cap: int = FINDING_CHARS) -> tuple[int, int] | None:
    """The sentence(s) around [start, end). One longer than `cap` is cut to whole words within `cap`
    characters around [start, end), keeping every negation word of the sentence that comes before it;
    None when [start, end) and those words do not fit in `cap` together. Never shorter than [start, end)."""
    low, high = sentence_bounds(text, start, end)
    width = max(cap, end - start)
    if high - low <= width:
        return low, high
    before = [low + offset for offset, _ in negation_words(text[low:high]) if low + offset < start]
    keep = before[0] if before else start  # the window starts here at the latest
    if end - keep > width:
        return None
    low = min(keep, max(low, min((start + end - width) // 2, high - width)))  # centred, when that keeps them
    high = max(end, min(high, low + width))
    while low < keep and not (text[low].isalnum() and (low == 0 or not text[low - 1].isalnum())):
        low += 1  # to the start of a whole word
    while high > end and not (text[high - 1].isalnum() and (high == len(text) or not text[high].isalnum())):
        high -= 1  # back to the end of a whole word
    return low, high


def locate_finding(doc: PrecedentDoc, quote: str) -> PrecedentQuote | None:
    """The finding as its whole sentence, or a window of it that keeps the negations before the quote
    (see finding_bounds), so the shown finding carries its own negation; None when it cannot."""
    located = locate(doc, quote)
    if located is None:
        return None
    bounds = finding_bounds(doc.text, located.span.start, located.span.end)
    if bounds is None:
        log.info("%s: finding %r: a negation of its sentence is too far before it to show both; dropped", doc.id, quote[:60])
        return None
    return PrecedentQuote(span=_span(doc, *bounds), match=located.match)


def _unique_by_span(quotes: Iterable[PrecedentQuote]) -> tuple[PrecedentQuote, ...]:
    first_by_span: dict[tuple[int, int], PrecedentQuote] = {}
    for quote in quotes:
        first_by_span.setdefault((quote.span.start, quote.span.end), quote)
    return tuple(first_by_span[span] for span in sorted(first_by_span))


# A negated outcome: the finding itself is denied ("does not find", "no violation", "not arbitrary"),
# not a negation elsewhere in the sentence ("a violation, since counsel was not present").
NEGATED_OUTCOME = re.compile(
    r"\bno\s+(?:separate\s+)?(?:violations?|breach(?:es)?)\b"
    r"|\bnot\s+(?:\w+\s+){0,3}?(?:find|finds|found|conclude|concludes|concluded|disclose|discloses|disclosed"
    r"|establish|establishes|established|reveal|reveals|revealed|show|shows|showed|shown"
    r"|demonstrate|demonstrates|demonstrated|substantiate|substantiates|substantiated)\b"
    r"|\b(?:not|unable)\s+(?:be\s+)?in\s+a\s+position\s+to\s+(?:find|conclude)\b"
    r"|\bcannot\s+(?:find|conclude)\b|\bnot\s+arbitrary\b",
    re.IGNORECASE,
)


def contradicts_kind(kind: FindingKind, finding: str) -> bool:
    """True when the finding's own words contradict its kind: a violation found where the outcome
    itself is negated, or no violation found where it is not."""
    negated = NEGATED_OUTCOME.search(finding) is not None
    return (kind == "violation_found" and negated) or (kind == "no_violation" and not negated)


def _verify_finding(doc: PrecedentDoc, facet: ExtractedFacet) -> _Finding:
    """The finding's kind and verified quote. A kind that does not fit the document drops the
    finding (the facts stay): a TrialWatch report never holds a violation found, and Views or an
    opinion never hold a monitor's assessment. So does a kind its own sentence contradicts
    (contradicts_kind); the facet then keeps UNSTATED, never the contradicted kind."""
    if facet.finding_kind not in FINDING_KINDS[doc.kind]:
        log.info("%s: %s finding %r does not fit a %s; dropped", doc.id, facet.facet_id, facet.finding_kind, doc.kind)
        return _Finding(UNSTATED, None, int(facet.finding is not None), wrong_kind=True)
    if facet.finding is None:
        return _Finding(facet.finding_kind, None, 0, wrong_kind=False)
    quote = locate_finding(doc, facet.finding.quote)
    if quote is not None and contradicts_kind(facet.finding_kind, quote.span.text):
        log.info("%s: %s finding %r contradicts its sentence; dropped", doc.id, facet.facet_id, facet.finding_kind)
        return _Finding(UNSTATED, None, 1, wrong_kind=False)
    return _Finding(facet.finding_kind, quote, int(quote is None), wrong_kind=False)


def _verify_facet(doc: PrecedentDoc, facet: ExtractedFacet) -> tuple[PrecedentFacet | None, int, bool]:
    """The facet with its verified quotes (None when no fact survives), how many quotes were dropped,
    and whether its finding's kind did not fit the document."""
    located = [locate(doc, fact.quote) for fact in facet.facts]
    facts = _unique_by_span(quote for quote in located if quote is not None)
    finding = _verify_finding(doc, facet)
    dropped = sum(quote is None for quote in located) + finding.dropped
    if not facts:
        return None, dropped + int(finding.quote is not None), finding.wrong_kind  # the finding goes with its facet
    kept = PrecedentFacet(precedent_id=doc.id, facet_id=facet.facet_id, facts=facts, finding_kind=finding.kind, finding=finding.quote)
    return kept, dropped, finding.wrong_kind


def verify_extraction(doc: PrecedentDoc, extraction: Extraction, taxonomy: FactPatternTaxonomy) -> Verified:
    """Keep the facets of the taxonomy with at least one verified fact, in the taxonomy's order."""
    order = {pattern.id: index for index, pattern in enumerate(taxonomy.facets)}
    kept: list[PrecedentFacet] = []
    dropped = wrong_kind = 0
    for facet in merge_facets(extraction.facets):
        if facet.facet_id not in order:
            dropped += _quote_count(facet)
            continue
        verified, lost, misfit = _verify_facet(doc, facet)
        dropped += lost
        wrong_kind += int(misfit)
        if verified is not None:
            kept.append(verified)
    return Verified(tuple(sorted(kept, key=lambda facet: order[facet.facet_id])), dropped, wrong_kind)


def read_only(store: BuildStore) -> sqlite3.Connection:
    """A read-only connection to the build store, for queries BuildStore has no method for."""
    return sqlite3.connect(f"{store.path.resolve().as_uri()}?mode=ro", uri=True)


def _extraction_rows(store: BuildStore) -> dict[str, list[tuple[str, str, str]]]:
    """(cache key, model, answer) per precedent, newest first."""
    with closing(read_only(store)) as db:
        rows = db.execute(
            "SELECT precedent_id, cache_key, model, answer_json FROM extractions ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for precedent_id, key, model, answer in rows:
        grouped.setdefault(precedent_id, []).append((key, model, answer))
    return grouped


def current_extraction(doc: PrecedentDoc, rows: Sequence[tuple[str, str, str]], digest: str) -> tuple[str, Extraction] | None:
    """The newest extraction made from this exact text with this prompt and schema."""
    for key, model, answer in rows:
        if key != cache_key(text_sha256=doc.text_sha256, prompt_sha=PROMPT_SHA, schema_sha=digest, model=model):
            continue
        try:
            return key, Extraction.model_validate_json(answer)
        except ValidationError as exc:
            log.warning("%s: stored extraction %s is unreadable (%s errors)", doc.id, key[:12], exc.error_count())
    return None


def verify_all(store: BuildStore, taxonomy: FactPatternTaxonomy) -> VerifyReport:
    """Verify each document's current extraction and store the facets that survive."""
    digest = schema_sha(taxonomy)
    rows = _extraction_rows(store)
    results: list[Verified] = []
    missing: list[str] = []
    for doc in store.documents():
        found = current_extraction(doc, rows.get(doc.id, ()), digest)
        if found is None:
            missing.append(doc.id)
            continue
        key, extraction = found
        result = verify_extraction(doc, extraction, taxonomy)
        store.put_verified(doc.id, result.facets, result.dropped, key)
        results.append(result)
    return VerifyReport(
        documents=len(results),
        facets=sum(len(result.facets) for result in results),
        quotes=sum(_quote_count(facet) for result in results for facet in result.facets),
        dropped=sum(result.dropped for result in results),
        wrong_kind=sum(result.wrong_kind for result in results),
        missing=tuple(missing),
    )


def _quote_count(facet: ExtractedFacet | PrecedentFacet) -> int:
    return len(facet.facts) + int(facet.finding is not None)

