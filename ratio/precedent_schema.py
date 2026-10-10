"""The precedent corpus contract, shared by the online builder (corpus_builder/) and the offline
runtime (ratio/precedents.py and its two index backends).

A precedent is a public past case: a UN Human Rights Committee decision, a UN Working Group on
Arbitrary Detention opinion, or a TrialWatch report. It is reference material, like the
jurisprudence corpus: it never enters the case store, never becomes a finding, and is never counted.
Every quote shown from it is a span of its stored text, checked character for character (hard rule 2).

The SQLite file below is the source of truth; OpenSearch, when used, is an index built from it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Literal, Protocol

import numpy as np
from pydantic import Field, model_validator

from ratio.schema import Evidence, Frozen, SourceSpan

SCHEMA_VERSION = 2  # 2: uploaded collections (private flag) and paragraph passages for search
EMBED_MODEL_TAG = "all-MiniLM-L6-v2/384"
PRECEDENT_PREFIX = "precedent-"  # SourceSpan.case_id of every precedent span: never a real case id

PrecedentKind = Literal["ccpr_views", "wgad_opinion", "trialwatch_report", "collection_document"]
COLLECTION_PREFIX = "collection:"  # the address of a document uploaded into a collection: collection:<slug>/<file>
FindingKind = Literal["violation_found", "no_violation", "not_examined", "monitor_assessment"]
QuoteMatch = Literal["exact", "normalized"]  # never fuzzy: a quote either is in the text or is dropped

# The kinds of finding each kind of document states: a Committee or the Working Group rules on the facts,
# a TrialWatch monitor assesses them. One table for the builder, the store and the app.
ALLOWED_FINDINGS: dict[PrecedentKind, frozenset[FindingKind]] = {
    "ccpr_views": frozenset({"violation_found", "no_violation", "not_examined"}),
    "wgad_opinion": frozenset({"violation_found", "no_violation", "not_examined"}),
    "trialwatch_report": frozenset({"monitor_assessment", "not_examined"}),
    "collection_document": frozenset({"monitor_assessment", "not_examined"}),  # an uploaded document assesses; it rules on no Covenant violation
}


def precedent_doc_id(precedent_id: str) -> str:
    """The SourceSpan.doc_id of a precedent's text; its case_id is "precedent-<id>"."""
    return f"{PRECEDENT_PREFIX}{precedent_id}/text"


def is_precedent_span(span: SourceSpan) -> bool:
    return span.case_id.startswith(PRECEDENT_PREFIX)


class PrecedentDoc(Frozen):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,80}$")  # e.g. "wgad-2023-1", "ccpr-1787-2008", "tw-abzas-media"
    kind: PrecedentKind
    symbol: str | None = None  # A/HRC/WGAD/2023/1, CCPR/C/107/D/1787/2008, or None for a report
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)  # who decided or assessed: "UN Human Rights Committee", ...
    state: str | None = None
    year: int | None = Field(default=None, ge=1950, le=2100)
    url: str = Field(pattern=r"^(?:https://|collection:)")  # a public address, or a document uploaded into a collection
    retrieved_at: str
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    language: str = "en"
    attribution: str = Field(min_length=1)
    text: str = Field(min_length=1)
    private: bool = False  # from a collection marked private: read only by the local model, never sent anywhere

    @property
    def doc_id(self) -> str:
        return precedent_doc_id(self.id)


class PrecedentQuote(Frozen):
    span: SourceSpan
    match: QuoteMatch


class PrecedentFacet(Frozen):
    """One fact pattern a precedent shows: the facts, quoted, and what the deciding body found on them."""

    precedent_id: str
    facet_id: str
    facts: tuple[PrecedentQuote, ...] = Field(min_length=1)
    finding_kind: FindingKind
    finding: PrecedentQuote | None = None

    @model_validator(mode="after")
    def _spans_are_this_precedent(self) -> PrecedentFacet:
        doc_id = precedent_doc_id(self.precedent_id)
        quotes = [*self.facts, *([self.finding] if self.finding else [])]
        if any(quote.span.doc_id != doc_id for quote in quotes):
            raise ValueError(f"{self.facet_id}: a quote is not from precedent {self.precedent_id}")
        return self


class CorpusSource(Frozen):
    kind: PrecedentKind
    name: str
    terms_url: str
    terms_checked: str
    attribution: str
    documents: int = Field(ge=0)


class CorpusMeta(Frozen):
    schema_version: int
    taxonomy_sha: str
    embed_model: str
    extraction_model: str
    prompt_sha: str
    built_at: str
    documents: int = Field(ge=0)
    facets: int = Field(ge=0)
    dropped_quotes: int = Field(ge=0)
    sources: tuple[CorpusSource, ...] = ()


class PrecedentIndex(Protocol):
    """Read access to a built corpus. Implemented by SqlitePrecedentIndex (default, no setup) and
    OpenSearchPrecedentIndex (a vector index on 127.0.0.1); both must return the same links."""

    def meta(self) -> CorpusMeta: ...

    def facet_doc_freq(self) -> Mapping[str, int]: ...  # facet id -> number of precedents showing it

    def candidates(self, facet_ids: Collection[str]) -> Sequence[tuple[PrecedentDoc, tuple[PrecedentFacet, ...]]]:
        """Every precedent with at least one of these facets, with all its facets."""
        ...

    def nearest_facts(self, facet_id: str, query: np.ndarray, precedent_ids: Collection[str]) -> Mapping[str, tuple[PrecedentQuote, float]]:
        """For each precedent, its fact quote on this facet closest to the query vector, with the cosine."""
        ...

    def text(self, doc_id: str) -> str | None: ...  # resolver for provenance.span_is_valid

    def documents(self) -> Sequence[PrecedentDoc]: ...

    def search_passages(
        self, query: np.ndarray, *, words: str = "", k: int = 10, precedent_ids: Collection[str] | None = None
    ) -> Sequence[PassageHit]:
        """The paragraphs closest to the query vector (and sharing its words), best first."""
        ...


class PassageHit(Frozen):
    """A paragraph of a precedent that a search found: a span of its stored text, and how close it is."""

    precedent_id: str
    quote: PrecedentQuote
    cosine: float
    words: float = 0.0  # the share of the query's words the paragraph contains
    score: float


# --- runtime results (ratio/precedents.py) -------------------------------------------------


class CaseFacet(Frozen):
    """A fact pattern of the case under review, with the exact passages it rests on."""

    facet_id: str
    origin: Literal["finding", "keyword"]
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    flag_id: str | None = None  # the finding it comes from, when origin == "finding"


class CaseProfile(Frozen):
    case_id: str
    facets: tuple[CaseFacet, ...] = ()


class SharedFacet(Frozen):
    facet_id: str
    label: str
    case_evidence: tuple[Evidence, ...] = Field(min_length=1)
    precedent_fact: PrecedentQuote
    finding_kind: FindingKind
    precedent_finding: PrecedentQuote | None = None
    pair_cosine: float | None = None


class ScoreBreakdown(Frozen):
    idf_sum: float
    shared: int
    mean_pair_cosine: float | None


class PrecedentLink(Frozen):
    precedent: PrecedentDoc
    shared: tuple[SharedFacet, ...] = Field(min_length=1)
    score: ScoreBreakdown

    @property
    def rank_key(self) -> tuple[float, int, float, str]:
        """Higher is better; the id breaks the last tie, so the order never depends on the backend."""
        return (round(self.score.idf_sum, 9), self.score.shared, round(self.score.mean_pair_cosine or 0.0, 9), self.precedent.id)


# --- SQLite layout of data/corpus/precedents.db --------------------------------------------

SQLITE_DDL: Sequence[str] = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, symbol TEXT, title TEXT NOT NULL, body TEXT NOT NULL,
        state TEXT, year INTEGER, url TEXT NOT NULL, retrieved_at TEXT NOT NULL, raw_sha256 TEXT NOT NULL,
        text_sha256 TEXT NOT NULL, language TEXT NOT NULL, attribution TEXT NOT NULL, text TEXT NOT NULL,
        private INTEGER NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS facets (
        precedent_id TEXT NOT NULL REFERENCES documents(id), facet_id TEXT NOT NULL, finding_kind TEXT NOT NULL,
        PRIMARY KEY (precedent_id, facet_id))""",
    """CREATE TABLE IF NOT EXISTS quotes (
        precedent_id TEXT NOT NULL, facet_id TEXT NOT NULL, role TEXT NOT NULL CHECK (role IN ('fact', 'finding')),
        start INTEGER NOT NULL, end INTEGER NOT NULL, match TEXT NOT NULL CHECK (match IN ('exact', 'normalized')),
        vec BLOB,  -- float32[384] of the sentence around the quote, L2-normalised (facts only)
        PRIMARY KEY (precedent_id, facet_id, role, start, end),
        FOREIGN KEY (precedent_id, facet_id) REFERENCES facets(precedent_id, facet_id))""",
    """CREATE TABLE IF NOT EXISTS passages (
        precedent_id TEXT NOT NULL REFERENCES documents(id), start INTEGER NOT NULL, end INTEGER NOT NULL,
        vec BLOB NOT NULL,  -- float32[384] of the paragraph, L2-normalised
        PRIMARY KEY (precedent_id, start, end))""",
    """CREATE TABLE IF NOT EXISTS sources (
        kind TEXT PRIMARY KEY, name TEXT NOT NULL, terms_url TEXT NOT NULL, terms_checked TEXT NOT NULL,
        attribution TEXT NOT NULL)""",
)
META_KEYS = ("schema_version", "taxonomy_sha", "embed_model", "extraction_model", "prompt_sha", "built_at", "dropped_quotes")
