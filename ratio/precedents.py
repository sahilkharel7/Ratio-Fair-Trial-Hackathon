"""Similar cases: the fact patterns of a case (its profile), and the public past cases that share them.

A reading aid, never a finding and never a prediction (hard rule 3). Pure functions, under the same
import rules as the modules: no store, no model, no files, no network.

- profile(): a fact pattern comes only from a stored finding whose module, standard and status match a
  signal in fact_patterns.yaml (and that the reviewing lawyer has not rejected), or from an exact
  keyword match that describes the accused or names the charge: a media word said of the accused
  ("Daro Venn, ..., journalist,"), never of an object or a victim ("stole a camera from a journalist");
  an offence word in the indictment's wording of a charge the accused faces ("charged with ..."), never
  an aside ("called the complaint fake news") or the object of another noun ("theft of a phone used to
  spread false news"). The charge rule is conservative, as a false charge misstates the case: a
  negation, a withdrawal or a past marker in the offence word's clause or later in its sentence vetoes
  it ("is not charged with", "but the charge was later dropped", "previously convicted of"). A "no
  evidence" follow-up or a benchmark that needs legal review is never a flag, so it never makes one
  (hard rule 5).
- link(): a precedent needs at least ``min_shared`` shared fact patterns, one of them about procedure.
  Rarer shared patterns weigh more (IDF over the corpus). The embedder only picks the precedent's
  passage closest to this case's evidence and breaks ties. Every precedent span is checked against its
  stored text, and one that fails is dropped (hard rule 2).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ratio.config import FactPattern, FactPatternTaxonomy, PrecedentSettings, keyword_pattern
from ratio.context import Embedder
from ratio.precedent_schema import (
    CaseFacet,
    CaseProfile,
    PrecedentDoc,
    PrecedentFacet,
    PrecedentIndex,
    PrecedentLink,
    PrecedentQuote,
    ScoreBreakdown,
    SharedFacet,
)
from ratio.provenance import DocResolver, span_is_valid
from ratio.results import CaseAnalysis
from ratio.schema import PARAGRAPH_BREAK, CaseRecord, Document, Evidence, Flag, Frozen, SourceSpan

# The two backends compute a cosine in float32 and differ by about 1e-7. Kept to 4 decimals, they almost
# always give the same value, so float noise does not reorder links and an exact tie falls to the
# precedent id. It narrows the difference without ruling it out: a cosine within 1e-7 of a rounding
# boundary can still round two ways.
COSINE_DECIMALS = 4
HEADING_CHARS = 80  # an all-capitals line up to this long is a heading ("VI. CHARGE")
_FIRST_SEARCHED = {"indictment": 0, "judgment": 1}  # then every other document, in record order
_SENTENCE_END = re.compile(r"[.!?][\"'”’»)\]]*\s+(?=[\"'“‘«(\[]?[A-Z0-9])")
_ACCUSED_WORDS = r"\bthe\s+accused\b|\bdefendants?\b|\bthe\s+author\b"  # "the author": a Committee's word for them
_PARTY_END = re.compile(r",|\(|\s+(?:and|&|et\s+al)\b", re.IGNORECASE)  # after the first party in "X v. Y and others"
_WORD = re.compile(r"[^\W\d_][\w'’-]*")

# A media word describes the accused when it follows their mention as an apposition ("Daro Venn, born
# ..., journalist,"; "the accused, a reporter,"; "the accused journalist") or as what they are ("the
# accused is a reporter", "works as an editor"). Read on the text between the mention and the word.
_NOT_A_MODIFIER = (
    r"(?:a|an|the|not|no|never|nor|with|without|of|for|to|by|from|against|and|or|but|as|at|in|on|into|than|like|about"
    r"|who|whom|whose|which|that|his|her|their|its|is|was|are|were)"
)
_MODIFIER = rf"(?!{_NOT_A_MODIFIER}\b)[^\W\d_][\w'’-]*\s+"  # "freelance ", "chief "; never "of ", "his "
_DESCRIPTION = rf"(?:an?|the)\s+(?:{_MODIFIER}){{0,3}}"  # "a ", "an investigative "
_BARE = rf"(?:(?![\w'’-]*(?:ed|ing)\b){_MODIFIER}){{0,2}}"  # "freelance " with no article; never a verb like "attacked "
_ASIDE = r"\s*,(?:(?!\b(?:who|whom|whose|which|that|with|and|or|his|her|their|its)\b)[^,;:])*"  # ", born in 1988"
_BRACKETS = r"(?:\s*\([^()]*\))?"  # "Daro Venn (born 1987), journalist"
_BE = r"(?:is|was|are|were|has\s+been|have\s+been|had\s+been|became|becomes|remains|remained)(?:\s+(?:also|still|now|then|currently|formerly|previously))?"
_WORKS_AS = r"(?:(?:is|was)\s+)?(?:works?|worked|working|serves?|served|serving|employed|active)\s+as"
_APPOSITION = re.compile(rf"\s+|{_BRACKETS}(?:{_ASIDE}){{0,3}}\s*,\s*(?:{_DESCRIPTION}|{_BARE})", re.IGNORECASE)
_PREDICATE = re.compile(rf"{_BRACKETS}(?:(?:{_ASIDE}){{1,3}}\s*,)?\s+(?:{_BE}|{_WORKS_AS})\s+(?:{_DESCRIPTION})?", re.IGNORECASE)
_POSSESSIVE = re.compile(r"['’](?:s\b|(?=\s))")  # "a publisher's warehouse", "the reporters' union"

# An offence word is the charge only in the indictment's charge wording: a "charged with ..." clause (to
# the end of its sentence, or to the first word that starts another clause or another charge wording),
# or a sentence of a CHARGE section that cites the provision, when that section has no such clause.
_CHARGED = re.compile(
    r"\b(?:charged|indicted|prosecuted)\s+(?:[\w()]+\s+){0,4}?(?:with|for)\b"  # "charged under Article 301 with"
    r"|\b(?:is|was|are|were|been|being|stands?|stood)\s+accused\s+of\b"  # never "a man accused of"
    r"|(?<!in )\b(?:offen[cs]es?|crimes?|charges?|counts?)\s+of\b",
    re.IGNORECASE,
)
# A charge the accused may not face is not one, and the rule is conservative: a missed charge facet costs
# little, a false one misstates the case. Any of these cues in the offence word's clause, or anywhere after
# it in its sentence, vetoes it: a negation ("No. 3713" is a number, not one), a withdrawal or an end, a
# past marker.
_NOT_FACED_CUE = re.compile(
    r"\b(?:not|never|no\b(?!\.?\s*\d)|cannot|neither|nor)\b|n['’]t\b"
    r"|\b(?:withdrawn|withdrew|dropped|dismissed|discontinued|quashed|struck\s+out|abandoned|acquitted|cleared)\b"
    r"|\b(?:previously|formerly|earlier|prior|past)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\bin\s+20\d\d\b", re.IGNORECASE)  # a past charge before the wording ("In 2019 ... was charged with")
# A past conviction leading into the wording from the clause before it, which the cues above do not read:
# "convicted of the offence of", "sentenced for the crime of", "previously accused of the offence of".
_CONVICTED_BEFORE = re.compile(
    r"\b(?:convicted|sentenced|(?:previously|formerly|earlier)\s+(?:been\s+)?accused)\s+(?:of|with|for)\s+(?:(?:the|an?)\s+)?$",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(  # a semicolon, a relative or subordinate clause, or "a man accused of ..."
    r";|\b(?:who|whom|whose|which|that|after|because|when|whenever|while|where|whereas|although|though|since|before|until"
    r"|following)\b|(?<!the )\b(?:accused|suspected|convicted|acquitted)\b",
    re.IGNORECASE,
)
# After a noun, these words make what follows the object of that noun, not the offence: "theft of a phone
# used to spread false news", "a camera belonging to a journalist", "the murder of a separatist leader".
# Not after a word for a charge ("offences relating to terrorism", "offences, including separatism"), and
# "of" not after a word for the conduct charged ("membership of a terrorist organisation", "publication
# of a defamatory article").
_OBJECT_OF_A_NOUN = re.compile(
    r"\b(?P<head>[^\W\d_][\w'’-]*)\s*,?\s+(?P<word>used|belonging|including|concerning|relating|containing|about|against|of\s+(?:an?|the))\b",
    re.IGNORECASE,
)
_CHARGE_NOUN = re.compile(r"offen[cs]es?|crimes?|charges?|counts?|acts?", re.IGNORECASE)
_CONDUCT_NOUN = re.compile(
    r"membership|participation|leadership|creation|founding|formation|establishment|organi[sz]ation|financing|funding|support"
    r"|publication|publishing|dissemination|distribution|spreading|production|possession|display|glorification|justification"
    r"|incitement",
    re.IGNORECASE,
)
_PROVISION = re.compile(r"(?:\bArticle|\bArt\.|\bSection|\bSec\.|§)\s*\d|\b(?:Penal|Criminal)\s+Code\b", re.IGNORECASE)
_HEADING_LINE = re.compile(r"^[ \t]*(?:(?:[IVXLC]+|\d+)[.)][ \t]*)?(?P<title>\S[^\n]*?)[ \t]*:?[ \t]*$", re.MULTILINE)
_CHARGE_TITLE = re.compile(r"(?:the\s+)?(?:charges?|counts?)(?:\s+sheet|\s+\d+|\s+[IVX]+)?", re.IGNORECASE)

Nearest = Mapping[str, tuple[PrecedentQuote, float]]
Candidate = tuple[PrecedentDoc, Sequence[PrecedentFacet]]


# --- profile ----------------------------------------------------------------------------------


def profile(
    record: CaseRecord, analysis: CaseAnalysis, taxonomy: FactPatternTaxonomy, rejected_flag_ids: Collection[str] = ()
) -> CaseProfile:
    """The case's fact patterns, in taxonomy order, each with the exact passages it rests on.

    ``rejected_flag_ids``: the findings the reviewing lawyer rejected; none of them makes a facet."""
    rejected = frozenset(rejected_flag_ids)
    flags = tuple(flag for flag in analysis.all_flags() if flag.id not in rejected)
    accused = _accused(record)
    facets = (_from_findings(pattern, flags) if pattern.signals else _from_keywords(pattern, record, accused) for pattern in taxonomy.facets)
    return CaseProfile(case_id=record.case_id, facets=tuple(facet for facet in facets if facet is not None))


def _from_findings(pattern: FactPattern, flags: Sequence[Flag]) -> CaseFacet | None:
    signals = {(signal.module, signal.standard, signal.status) for signal in pattern.signals}
    flag = next((flag for flag in flags if (flag.module, flag.standard_id, flag.status) in signals), None)
    if flag is None:
        return None
    return CaseFacet(facet_id=pattern.id, origin="finding", evidence=flag.evidence, flag_id=flag.id)


@dataclass(frozen=True)
class _Accused:
    """How the record names the accused: "the accused", "defendant", "the author", and their surname."""

    pattern: re.Pattern[str]
    common_surname: str | None = None  # a surname the record also writes as an ordinary word ("long")

    def mentions(self, text: str, start: int, end: int) -> Iterator[re.Match[str]]:
        """The mentions in text[start:end], where ``start`` opens a sentence. A surname that is also a common
        word does not count there, where its capital says nothing: "Long queues formed" is not Ms Long."""
        for mention in self.pattern.finditer(text, start, end):
            if not (mention.start() == start and mention.group() == self.common_surname):
                yield mention


def _from_keywords(pattern: FactPattern, record: CaseRecord, accused: _Accused) -> CaseFacet | None:
    """The first whole-word match that describes the accused or names the charge; its sentence is the evidence.

    A profile facet (who was prosecuted) needs the media word to describe the accused, so a reporter in
    the courtroom, press coverage of the arrest or a journalist robbed by the accused is not one. A
    charge facet comes only from the charge wording of an indictment, never from a note, an order or a
    judgment: the record marks no charge recital in a judgment, and a claim, a witness's words or a
    recital of the law there is not the charge."""
    matcher = keyword_pattern(pattern.keywords)
    for doc in _searched(pattern, record):
        for match in matcher.finditer(doc.text):
            sentence = _sentence_of(record, doc, match.start(), match.end())
            if not _keyword_counts(pattern, doc.text, sentence, match, accused):
                continue
            start, end = sentence
            span = SourceSpan(doc_id=doc.id, start=start, end=end, text=doc.text[start:end])
            return CaseFacet(facet_id=pattern.id, origin="keyword", evidence=(Evidence(role="mention", span=span),))
    return None


def _keyword_counts(pattern: FactPattern, text: str, sentence: tuple[int, int], match: re.Match[str], accused: _Accused) -> bool:
    if pattern.group == "charge":
        return _names_the_charge(text, sentence, match)
    return _describes_the_accused(text, sentence, match, accused)


def _describes_the_accused(text: str, sentence: tuple[int, int], match: re.Match[str], accused: _Accused) -> bool:
    """Whether the media word is said of the accused, in apposition or as what they are, rather than of
    someone or something else in the sentence ("stole a laptop belonging to a reporter", "a publisher's
    warehouse", "the accused's brother is a journalist")."""
    if _POSSESSIVE.match(text, match.end()):
        return False
    for mention in accused.mentions(text, sentence[0], match.start()):
        between = text[mention.end() : match.start()]
        if _APPOSITION.fullmatch(between) or _PREDICATE.fullmatch(between):
            return True
    return False


def _names_the_charge(text: str, sentence: tuple[int, int], match: re.Match[str]) -> bool:
    """Whether the offence word is in the wording of a charge the accused faces: a "charged with ..." clause,
    else a sentence of a CHARGE section that cites the provision when the section has no such clause."""
    wording = _charge_wording(text, sentence, match)
    return wording is not None and _faced(text, sentence, match, wording)


def _charge_wording(text: str, sentence: tuple[int, int], match: re.Match[str]) -> int | None:
    """Where the charge wording holding the offence word starts: its "charged with ..." clause, else the
    offence word itself in a CHARGE section sentence citing the provision; None when it is in neither."""
    start, end = sentence
    for clause in _CHARGED.finditer(text, start, end):
        if clause.start() <= match.start() and _in_clause(text, clause.end(), match):
            return clause.start()
    section = _charge_section(text, match.start())
    if section is None or _CHARGED.search(text, *section) is not None:
        return None
    return match.start() if _PROVISION.search(text, start, end) and _in_clause(text, start, match) else None


def _faced(text: str, sentence: tuple[int, int], match: re.Match[str], wording: int) -> bool:
    """Whether the charge whose wording starts at ``wording`` is one the accused faces. Never when a cue sits
    in the offence word's clause (from the last clause break before it) or after it in its sentence ("is not
    charged with", "the charge was later dropped"), when "in 2019" comes before the wording in that clause,
    or when a past conviction leads into the wording ("previously convicted of the offence of")."""
    start, end = sentence
    clause = max((found.start() for found in _CLAUSE_BREAK.finditer(text, start, match.start())), default=start)
    if _NOT_FACED_CUE.search(text, clause, end) or _YEAR.search(text, clause, max(clause, wording)):
        return False
    return _CONVICTED_BEFORE.search(text, start, wording) is None


def _in_clause(text: str, start: int, match: re.Match[str]) -> bool:
    """Whether the offence word is in the clause from ``start``: nothing in between ends that clause."""
    return _clause_end(text, start, match.start()) == match.start()


def _clause_end(text: str, start: int, end: int) -> int:
    """Where the clause from ``start`` ends: at a semicolon, a word that opens another clause, another
    charge wording, or a word that makes what follows the object of a noun; else at ``end``."""
    stops = [found.start() for found in (_CLAUSE_BREAK.search(text, start, end), _CHARGED.search(text, start, end)) if found]
    stops += [found.start("word") for found in _OBJECT_OF_A_NOUN.finditer(text, start, end) if _ends_the_charge(found)][:1]
    return min(stops, default=end)


def _ends_the_charge(found: re.Match[str]) -> bool:
    """Whether "<head> used/of a/..." makes what follows the object of the head rather than part of the charge."""
    head = found["head"]
    if _CHARGE_NOUN.fullmatch(head):
        return False
    return not (found["word"].lower().startswith("of") and _CONDUCT_NOUN.fullmatch(head))


def _charge_section(text: str, pos: int) -> tuple[int, int] | None:
    """The CHARGE section holding ``pos``: from its heading to the next heading, if ``pos`` is in one."""
    headings = _headings(text)
    current = next((heading for heading in reversed(headings) if heading[1] <= pos), None)
    if current is None or not current[2]:
        return None
    following = next((heading[0] for heading in headings if heading[0] > pos), len(text))
    return current[1], following


def _headings(text: str) -> tuple[tuple[int, int, bool], ...]:
    """(start, end, is a CHARGE heading) of each heading line: "VI. CHARGE", "Charges", "STATEMENT OF FACTS"."""
    lines = ((line, _CHARGE_TITLE.fullmatch(line["title"]) is not None) for line in _HEADING_LINE.finditer(text))
    return tuple(
        (line.start(), line.end(), is_charge)
        for line, is_charge in lines
        if is_charge or (line["title"].isupper() and len(line["title"]) <= HEADING_CHARS)
    )


def _searched(pattern: FactPattern, record: CaseRecord) -> Sequence[Document]:
    """Where a keyword facet is looked for: a charge in the indictment only; a profile facet in every
    document, the indictment and judgment first."""
    if pattern.group == "charge":
        return record.documents_of_type("indictment")
    return sorted(record.documents, key=lambda doc: _FIRST_SEARCHED.get(doc.type, len(_FIRST_SEARCHED)))


def _accused(record: CaseRecord) -> _Accused:
    """Words that name the accused: "the accused", "defendant", "the author", and their surname from the title.

    The surname matches case-sensitively ("Long" is a name, "long" is not). One the record also writes in
    lowercase, as an ordinary word, is a common word, and is not taken for the accused at a sentence start."""
    surname = _defendant_surname(record.meta.title)
    named = rf"|\b{re.escape(surname)}\b" if surname else ""
    pattern = re.compile(rf"(?i:{_ACCUSED_WORDS}){named}")
    if surname is None:
        return _Accused(pattern)
    as_a_word = re.compile(rf"\b{re.escape(surname.lower())}\b")
    common = any(as_a_word.search(doc.text) for doc in record.documents)
    return _Accused(pattern, surname if common else None)


def _defendant_surname(title: str) -> str | None:
    """The last word of the first party after " v. ", as in "Republic of Calderra v. Daro Venn"."""
    _, versus, party = title.partition(" v. ")
    words = _WORD.findall(_PARTY_END.split(party, maxsplit=1)[0]) if versus else []
    return words[-1] if words and len(words[-1]) > 1 else None


def _sentence_of(record: CaseRecord, doc: Document, start: int, end: int) -> tuple[int, int]:
    """The record's own sentence (a passage or an observation) holding the match, else one found here."""
    segments = [p.span for p in record.passages if p.doc_id == doc.id] + [o.span for o in record.observations if o.span.doc_id == doc.id]
    for span in segments:
        if span.start <= start and end <= span.end:
            return span.start, span.end
    return _sentence_bounds(doc.text, start, end)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """[start, end) widened to its sentence, within its paragraph, without surrounding whitespace."""
    paragraph_start = max((m.end() for m in PARAGRAPH_BREAK.finditer(text, 0, start)), default=0)
    following = PARAGRAPH_BREAK.search(text, end)
    paragraph_end = following.start() if following else len(text)
    low = max((m.end() for m in _SENTENCE_END.finditer(text, paragraph_start, start)), default=paragraph_start)
    stop = _SENTENCE_END.search(text, end, paragraph_end)
    high = stop.end() if stop else paragraph_end
    while low < start and text[low].isspace():
        low += 1
    while high > end and text[high - 1].isspace():
        high -= 1
    return low, high


# --- link -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Linking:
    """What every precedent is linked against: this case's facets and the index's answers."""

    case_facets: Mapping[str, CaseFacet]
    nearest: Mapping[str, Nearest]  # facet id -> precedent id -> closest fact and its cosine
    resolve: DocResolver
    weight: Callable[[str], float]  # facet id -> IDF weight
    settings: PrecedentSettings
    taxonomy: FactPatternTaxonomy


def link(
    profile: CaseProfile, index: PrecedentIndex, embedder: Embedder, settings: PrecedentSettings, taxonomy: FactPatternTaxonomy
) -> tuple[PrecedentLink, ...]:
    """The precedents sharing enough of the case's fact patterns, best first (at most ``top_k``)."""
    case_facets = {facet.facet_id: facet for facet in profile.facets}
    if not case_facets:
        return ()
    index = _for_one_call(index)
    candidates = [
        (doc, facets)
        for doc, facets in index.candidates(tuple(case_facets))
        if _eligible([facet.facet_id for facet in facets if facet.facet_id in case_facets], settings, taxonomy)
    ]
    linking = _Linking(case_facets, _nearest(case_facets, candidates, index, embedder), index.text, _idf_weights(index), settings, taxonomy)
    links = (_link_one(doc, facets, linking) for doc, facets in candidates)
    ranked = sorted((found for found in links if found is not None), key=lambda found: found.rank_key, reverse=True)
    return tuple(ranked[: settings.top_k])


def _for_one_call(index: PrecedentIndex) -> PrecedentIndex:
    """The index as one link() call reads it. A backend that can fail mid-call (OpenSearch) offers
    ``per_call()``: a view that, after its first failed search, answers the rest of the call from the file."""
    per_call = getattr(index, "per_call", None)
    return per_call() if callable(per_call) else index


def _eligible(shared_ids: Sequence[str], settings: PrecedentSettings, taxonomy: FactPatternTaxonomy) -> bool:
    return len(shared_ids) >= settings.min_shared and any(taxonomy.facet(fid).group == "procedure" for fid in shared_ids)


def _idf_weights(index: PrecedentIndex) -> Callable[[str], float]:
    documents, freq = index.meta().documents, index.facet_doc_freq()
    return lambda facet_id: math.log((documents + 1) / (freq.get(facet_id, 0) + 1)) + 1


def _nearest(case_facets: Mapping[str, CaseFacet], candidates: Sequence[Candidate], index: PrecedentIndex, embedder: Embedder) -> dict[str, Nearest]:
    """facet id -> precedent id -> the precedent's fact closest to this case's evidence, and the cosine."""
    wanted = {fid: sorted(doc.id for doc, facets in candidates if any(f.facet_id == fid for f in facets)) for fid in case_facets}
    wanted = {fid: ids for fid, ids in wanted.items() if ids}
    queries = _queries(embedder, {fid: case_facets[fid].evidence for fid in wanted})
    return {fid: index.nearest_facts(fid, queries[fid], ids) for fid, ids in wanted.items() if queries.get(fid) is not None}


def _queries(embedder: Embedder, evidence: Mapping[str, Sequence[Evidence]]) -> dict[str, np.ndarray | None]:
    """One query per facet: the mean of its evidence vectors, renormalised (None if it has no direction)."""
    texts = [item.span.text for items in evidence.values() for item in items]
    if not texts:
        return {}
    vectors = _unit_rows(np.asarray(embedder.encode(texts), dtype=np.float32))
    queries: dict[str, np.ndarray | None] = {}
    row = 0
    for facet_id, items in evidence.items():
        mean = vectors[row : row + len(items)].mean(axis=0)
        row += len(items)
        norm = float(np.linalg.norm(mean))
        queries[facet_id] = mean / norm if norm > 0 else None
    return queries


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _link_one(doc: PrecedentDoc, facets: Sequence[PrecedentFacet], linking: _Linking) -> PrecedentLink | None:
    """The link to one precedent, from the shared facets whose quotes pass the provenance check."""
    picks = ((facet, linking.nearest.get(facet.facet_id, {}).get(doc.id)) for facet in facets if facet.facet_id in linking.case_facets)
    shared = [found for facet, pick in picks if (found := _shared_facet(doc, facet, pick, linking)) is not None]
    if not _eligible([s.facet_id for s in shared], linking.settings, linking.taxonomy):
        return None
    cosines = [s.pair_cosine for s in shared if s.pair_cosine is not None]
    score = ScoreBreakdown(
        idf_sum=sum(linking.weight(s.facet_id) for s in shared),
        shared=len(shared),
        mean_pair_cosine=round(sum(cosines) / len(cosines), COSINE_DECIMALS) if cosines else None,
    )
    return PrecedentLink(precedent=doc, shared=tuple(shared), score=score)


def _shared_facet(doc: PrecedentDoc, facet: PrecedentFacet, pick: tuple[PrecedentQuote, float] | None, linking: _Linking) -> SharedFacet | None:
    """The closest fact (else the first), checked against the stored text; None when it fails."""
    fact, cosine = pick if pick is not None else (facet.facts[0], None)
    if fact.span.doc_id != doc.doc_id or not span_is_valid(fact.span, linking.resolve):
        return None
    finding = facet.finding if facet.finding is not None and span_is_valid(facet.finding.span, linking.resolve) else None
    return SharedFacet(
        facet_id=facet.facet_id,
        label=linking.taxonomy.facet(facet.facet_id).label,
        case_evidence=linking.case_facets[facet.facet_id].evidence,
        precedent_fact=fact,
        finding_kind=facet.finding_kind,
        precedent_finding=finding,
        pair_cosine=None if cosine is None else round(float(cosine), COSINE_DECIMALS),
    )



# --- similar wording: the case's own paragraphs searched against the library's ------------------

WORDING_K = 3  # library paragraphs fetched per paragraph of the case
WORDING_MIN_COSINE = 0.6  # a weaker match is not shown: similar wording, not the same topic
CASE_PASSAGE_CHARS = 800
MIN_CASE_PASSAGE_CHARS = 80
MAX_CASE_PASSAGES = 160
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+")


class WordingMatch(Frozen):
    """A precedent whose wording is close to this case's: the best pair of paragraphs, and how many of
    the case's paragraphs found it. Similar wording only: no fact pattern is claimed."""

    precedent: PrecedentDoc
    case_passage: SourceSpan
    precedent_passage: PrecedentQuote
    cosine: float
    support: int


def _pieces(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    """[start, end) cut at sentence ends into pieces of at most CASE_PASSAGE_CHARS (a longer sentence stays whole)."""
    piece_start = start
    for match in _SENTENCE_BREAK.finditer(text, start, end):
        if match.start() - piece_start > CASE_PASSAGE_CHARS:
            yield piece_start, match.start()
            piece_start = match.end()
    yield piece_start, end


def case_passages(record: CaseRecord) -> tuple[SourceSpan, ...]:
    """The case's documents as paragraphs, each an exact, trimmed span, in document order (capped)."""
    spans: list[SourceSpan] = []
    for doc in record.documents:
        text, start = doc.text, 0
        for brk in [*re.finditer(r"\n\s*\n", text), None]:
            end = brk.start() if brk else len(text)
            for low, high in _pieces(text, start, end):
                while low < high and text[low].isspace():
                    low += 1
                while high > low and text[high - 1].isspace():
                    high -= 1
                if high - low >= MIN_CASE_PASSAGE_CHARS:
                    spans.append(SourceSpan(doc_id=doc.id, start=low, end=high, text=text[low:high]))
            start = brk.end() if brk else len(text)
    return tuple(spans[:MAX_CASE_PASSAGES])


def similar_wording(record: CaseRecord, index: PrecedentIndex, embedder: Embedder, *, limit: int = 5) -> tuple[WordingMatch, ...]:
    """The precedents whose paragraphs are closest to the case's own, best first. Every precedent
    paragraph shown is re-checked against its text; no model runs."""
    spans = case_passages(record)
    if not spans:
        return ()
    index = _for_one_call(index)  # a stalled OpenSearch costs one timeout, not one per paragraph
    vectors = embedder.encode([span.text for span in spans])
    best: dict[str, tuple[float, SourceSpan, PrecedentQuote]] = {}
    support: dict[str, int] = {}
    for span, vector in zip(spans, vectors, strict=True):
        found: set[str] = set()
        for hit in index.search_passages(vector, k=WORDING_K):
            if hit.cosine < WORDING_MIN_COSINE or not span_is_valid(hit.quote.span, index.text):
                continue
            found.add(hit.precedent_id)
            if hit.precedent_id not in best or hit.cosine > best[hit.precedent_id][0]:
                best[hit.precedent_id] = (hit.cosine, span, hit.quote)
        for pid in found:  # each case paragraph counts once per precedent, however many of its paragraphs it found
            support[pid] = support.get(pid, 0) + 1
    docs = {doc.id: doc for doc in index.documents()}
    matches = [
        WordingMatch(precedent=docs[pid], case_passage=span, precedent_passage=quote, cosine=round(cosine, 4), support=support[pid])
        for pid, (cosine, span, quote) in best.items() if pid in docs
    ]  # fmt: skip
    return tuple(sorted(matches, key=lambda m: (-(m.cosine + 0.01 * min(m.support, 10)), m.precedent.id))[:limit])
