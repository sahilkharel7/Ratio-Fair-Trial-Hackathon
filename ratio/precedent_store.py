"""The precedent corpus file, data/corpus/precedents.db, read-only: the default backend of Similar cases
and the source of truth for every precedent text and quote (OpenSearch, when used, only indexes it).

The file is opened read-only and read once: the corpus is small, and an in-memory copy can be shared by
every Streamlit session without a connection per thread. A corpus built for another schema, set of fact
patterns or embedding model is refused with every build step that rebuilds it (BUILD_CHAIN). A quote that falls outside its
text is skipped, and a text edited after its quotes were verified (its sha256 no longer matches)
resolves to nothing, so the provenance check drops every quote of it. A finding of a kind its document
never states (ALLOWED_FINDINGS: a TrialWatch report finds no violation) is dropped, and its facts kept.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from ratio.config import FactPatternTaxonomy, default_config
from ratio.embeddings import EMBEDDING_DIM
from ratio.paths import corpus_db_path
from ratio.precedent_schema import (
    ALLOWED_FINDINGS,
    EMBED_MODEL_TAG,
    PRECEDENT_PREFIX,
    SCHEMA_VERSION,
    CorpusMeta,
    CorpusSource,
    FindingKind,
    PassageHit,
    PrecedentDoc,
    PrecedentFacet,
    PrecedentKind,
    PrecedentQuote,
    precedent_doc_id,
)
from ratio.schema import SourceSpan

# Every step, so the fix is right even on a computer without the builder's work (data/corpus/build.db).
BUILD_CHAIN = "python -m corpus_builder fetch, then normalize, extract, verify, embed and build-db"
_FIX = f"while online: {BUILD_CHAIN} (see corpus_builder/)."
_TEXT_SUFFIX = "/text"
WORD_WEIGHT = 0.25  # in a search, how much the shared words count beside the meaning (the cosine)
COSINE_DECIMALS = 4  # as in ratio/precedents.py: cosines are compared at this precision, where both backends agree
CANDIDATES = 200  # passages re-scored for shared words, taken by cosine first
_WORD = re.compile(r"[a-z][a-z'-]{2,}")
_STOP = frozenset("the and for with that this from was were has have had are not but his her their its into who which been also".split())
UNSTATED: FindingKind = "not_examined"  # valid for every kind of document, and never shown without a finding quote


class CorpusUnavailable(RuntimeError):
    """No usable corpus: the file is missing, unreadable, or built for another schema, taxonomy or model."""


@dataclass(frozen=True)
class _Rows:
    meta: dict[str, str]
    documents: list[sqlite3.Row]
    facets: list[sqlite3.Row]
    quotes: list[sqlite3.Row]
    sources: list[sqlite3.Row]
    passages: list[sqlite3.Row]


def _refused(path: Path, problem: str) -> CorpusUnavailable:
    return CorpusUnavailable(f"The precedent corpus at {path} {problem}. Rebuild it {_FIX}")


def _read(path: Path) -> _Rows:
    if not path.is_file():
        raise CorpusUnavailable(f"No precedent corpus at {path}. Build it {_FIX}")
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            return _Rows(
                meta={row["key"]: row["value"] for row in db.execute("SELECT key, value FROM meta")},
                documents=db.execute("SELECT * FROM documents ORDER BY id").fetchall(),
                facets=db.execute("SELECT * FROM facets ORDER BY precedent_id, facet_id").fetchall(),
                quotes=db.execute("SELECT * FROM quotes ORDER BY precedent_id, facet_id, role, start, end").fetchall(),
                sources=db.execute("SELECT * FROM sources ORDER BY kind").fetchall(),
                passages=db.execute("SELECT precedent_id, start, end, vec FROM passages ORDER BY precedent_id, start").fetchall(),
            )
    except sqlite3.Error as exc:
        raise _refused(path, f"cannot be read ({exc})") from exc


def _meta_value(meta: Mapping[str, str], key: str, path: Path) -> str:
    if key not in meta:
        raise _refused(path, f"has no {key} in its meta")
    return meta[key]


def _int_meta(meta: Mapping[str, str], key: str, path: Path) -> int:
    try:
        return int(_meta_value(meta, key, path))
    except ValueError as exc:
        raise _refused(path, f"has an invalid {key}") from exc


def _check_meta(meta: Mapping[str, str], taxonomy: FactPatternTaxonomy, path: Path) -> None:
    """Refuse a corpus this Ratio cannot read correctly, saying what to run."""
    version = _meta_value(meta, "schema_version", path)
    if version != str(SCHEMA_VERSION):
        raise _refused(path, f"has schema version {version}; this Ratio reads version {SCHEMA_VERSION}")
    if _meta_value(meta, "taxonomy_sha", path) != taxonomy.sha:
        raise _refused(path, "was built for other fact patterns (fact_patterns.yaml changed)")
    model = _meta_value(meta, "embed_model", path)
    if model != EMBED_MODEL_TAG:
        raise _refused(path, f"has vectors from the embedding model {model}; this Ratio uses {EMBED_MODEL_TAG}")


def _quote(doc: PrecedentDoc, row: sqlite3.Row) -> PrecedentQuote | None:
    """The quote as a span of the stored text; None when it falls outside the text or is blank."""
    start, end, text = row["start"], row["end"], doc.text
    if not (0 <= start < end <= len(text)) or not text[start:end].strip() or row["match"] not in ("exact", "normalized"):
        return None
    span = SourceSpan(doc_id=precedent_doc_id(doc.id), start=start, end=end, text=text[start:end])
    return PrecedentQuote(span=span, match=row["match"])


def _vector(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    vec = np.frombuffer(blob, dtype=np.float32)
    norm = float(np.linalg.norm(vec)) if vec.size == EMBEDDING_DIM else 0.0
    return vec / norm if norm > 0 else None


def _text_changed(doc: PrecedentDoc) -> bool:
    """Whether the stored text is no longer the text its quotes were verified against."""
    return hashlib.sha256(doc.text.encode("utf-8")).hexdigest() != doc.text_sha256


def _fitting(facet: PrecedentFacet, kind: PrecedentKind) -> PrecedentFacet:
    """The facet without a finding its kind of document never states (a TrialWatch report finds no
    violation); its facts stay."""
    if facet.finding_kind in ALLOWED_FINDINGS[kind]:
        return facet
    return facet.model_copy(update={"finding_kind": UNSTATED, "finding": None})


@dataclass(frozen=True)
class _Fact:
    quote: PrecedentQuote
    vec: np.ndarray | None


def _facts_and_findings(
    docs: Mapping[str, PrecedentDoc], rows: Sequence[sqlite3.Row]
) -> tuple[dict[tuple[str, str], list[_Fact]], dict[tuple[str, str], PrecedentQuote]]:
    """(precedent, facet) -> its facts in text order with their vectors, and its first finding."""
    facts: dict[tuple[str, str], list[_Fact]] = defaultdict(list)
    findings: dict[tuple[str, str], PrecedentQuote] = {}
    for row in rows:
        doc = docs.get(row["precedent_id"])
        quote = _quote(doc, row) if doc is not None else None
        if quote is None:
            continue
        key = (row["precedent_id"], row["facet_id"])
        if row["role"] == "fact":
            facts[key].append(_Fact(quote, _vector(row["vec"])))
        else:
            findings.setdefault(key, quote)
    return facts, findings


class SqlitePrecedentIndex:
    """PrecedentIndex over precedents.db. ``path`` defaults to corpus_db_path() (RATIO_CORPUS_DB)."""

    def __init__(self, path: Path | None = None, taxonomy: FactPatternTaxonomy | None = None) -> None:
        self.path = Path(path) if path is not None else corpus_db_path()
        taxonomy = taxonomy if taxonomy is not None else default_config().fact_patterns
        rows = _read(self.path)
        _check_meta(rows.meta, taxonomy, self.path)
        try:
            self._load(rows, taxonomy)
        except ValidationError as exc:
            raise _refused(self.path, f"has an invalid row ({exc.errors()[0]['msg']})") from exc

    def _load(self, rows: _Rows, taxonomy: FactPatternTaxonomy) -> None:
        self._docs = {row["id"]: PrecedentDoc.model_validate(dict(row)) for row in rows.documents}
        self._altered = frozenset(pid for pid, doc in self._docs.items() if _text_changed(doc))
        facts, findings = _facts_and_findings(self._docs, rows.quotes)
        self._facts = {key: tuple(found) for key, found in facts.items()}
        self._facets = self._build_facets(rows.facets, findings, taxonomy)
        self._meta = self._build_meta(rows, taxonomy)
        self._load_passages(rows.passages)

    def _load_passages(self, rows: Sequence[sqlite3.Row]) -> None:
        """The paragraphs a search reads: spans of texts that still match their checksum, with vectors."""
        spans, vectors = [], []
        for row in rows:
            doc, vec = self._docs.get(row["precedent_id"]), _vector(row["vec"])
            if doc is None or vec is None or doc.id in self._altered or not (0 <= row["start"] < row["end"] <= len(doc.text)):
                continue
            spans.append((doc.id, row["start"], row["end"]))
            vectors.append(vec)
        self._passage_spans = tuple(spans)
        self._passage_vecs = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    def _build_facets(
        self, rows: Sequence[sqlite3.Row], findings: Mapping[tuple[str, str], PrecedentQuote], taxonomy: FactPatternTaxonomy
    ) -> dict[str, tuple[PrecedentFacet, ...]]:
        """Each precedent's facets in taxonomy order; a facet left with no fact quote is skipped, and a
        finding of a kind its document never states is dropped."""
        order = {facet.id: number for number, facet in enumerate(taxonomy.facets)}
        by_precedent: dict[str, list[PrecedentFacet]] = defaultdict(list)
        for row in rows:
            key = (row["precedent_id"], row["facet_id"])
            if row["facet_id"] not in order or not self._facts.get(key):
                continue
            facts = tuple(fact.quote for fact in self._facts[key])
            facet = PrecedentFacet(precedent_id=key[0], facet_id=key[1], facts=facts, finding_kind=row["finding_kind"], finding=findings.get(key))
            by_precedent[key[0]].append(_fitting(facet, self._docs[key[0]].kind))
        return {pid: tuple(sorted(found, key=lambda f: order[f.facet_id])) for pid, found in sorted(by_precedent.items())}

    def _build_meta(self, rows: _Rows, taxonomy: FactPatternTaxonomy) -> CorpusMeta:
        counts = Counter(doc.kind for doc in self._docs.values())
        sources = tuple(CorpusSource.model_validate({**dict(row), "documents": counts.get(row["kind"], 0)}) for row in rows.sources)
        return CorpusMeta(
            schema_version=SCHEMA_VERSION,
            taxonomy_sha=taxonomy.sha,
            embed_model=EMBED_MODEL_TAG,
            extraction_model=rows.meta.get("extraction_model", ""),
            prompt_sha=rows.meta.get("prompt_sha", ""),
            built_at=_meta_value(rows.meta, "built_at", self.path),
            documents=len(self._docs),
            facets=len(rows.facets),
            dropped_quotes=_int_meta(rows.meta, "dropped_quotes", self.path),
            sources=sources,
        )

    # --- PrecedentIndex ----------------------------------------------------------------------

    def meta(self) -> CorpusMeta:
        return self._meta

    def facet_doc_freq(self) -> Mapping[str, int]:
        return Counter(facet.facet_id for facets in self._facets.values() for facet in facets)

    def candidates(self, facet_ids: Collection[str]) -> Sequence[tuple[PrecedentDoc, tuple[PrecedentFacet, ...]]]:
        wanted = set(facet_ids)
        return tuple((self._docs[pid], facets) for pid, facets in self._facets.items() if any(f.facet_id in wanted for f in facets))

    def nearest_facts(self, facet_id: str, query: np.ndarray, precedent_ids: Collection[str]) -> Mapping[str, tuple[PrecedentQuote, float]]:
        """The cosine of the query with each stored fact vector; on a tie, the fact earliest in the text."""
        query = np.asarray(query, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return {}
        query = query / norm
        found: dict[str, tuple[PrecedentQuote, float]] = {}
        for pid in sorted(set(precedent_ids)):
            facts = [fact for fact in self._facts.get((pid, facet_id), ()) if fact.vec is not None]
            if not facts:
                continue
            if query.shape[0] != facts[0].vec.shape[0]:
                raise ValueError(f"query has {query.shape[0]} dimensions; the corpus vectors have {facts[0].vec.shape[0]}")
            cosines = np.clip(np.stack([fact.vec for fact in facts]) @ query, -1.0, 1.0)
            best = int(np.argmax(cosines))
            found[pid] = (facts[best].quote, float(cosines[best]))
        return found

    def text(self, doc_id: str) -> str | None:
        if not (doc_id.startswith(PRECEDENT_PREFIX) and doc_id.endswith(_TEXT_SUFFIX)):
            return None
        pid = doc_id[len(PRECEDENT_PREFIX) : -len(_TEXT_SUFFIX)]
        doc = self._docs.get(pid)
        return None if doc is None or pid in self._altered else doc.text

    def documents(self) -> Sequence[PrecedentDoc]:
        return tuple(self._docs.values())

    def passage_count(self) -> int:
        return len(self._passage_spans)

    def passage(self, precedent_id: str, start: int, end: int) -> PrecedentQuote | None:
        """The stored paragraph at this span as a quote of its text, or None (a span the file does not hold)."""
        doc = self._docs.get(precedent_id)
        if doc is None or precedent_id in self._altered or not (0 <= start < end <= len(doc.text)):
            return None
        return PrecedentQuote(span=SourceSpan(doc_id=precedent_doc_id(precedent_id), start=start, end=end, text=doc.text[start:end]), match="exact")

    def search_passages(
        self, query: np.ndarray, *, words: str = "", k: int = 10, precedent_ids: Collection[str] | None = None
    ) -> Sequence[PassageHit]:
        """The paragraphs closest in meaning to the query vector, re-ranked by the share of the query's
        words they contain; best first."""
        if not self._passage_spans:
            return ()
        query = np.asarray(query, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return ()
        cosines = np.clip(self._passage_vecs @ (query / norm), -1.0, 1.0)
        if precedent_ids is not None:
            allowed = set(precedent_ids)
            cosines = np.where([pid in allowed for pid, _, _ in self._passage_spans], cosines, -2.0)
        order = np.argsort(-cosines)[: max(k, CANDIDATES)]
        return rank_passages([(self._passage_spans[i], float(cosines[i])) for i in order if cosines[i] > -2.0], self.passage, words, k)

    # --- for the OpenSearch backend and its builder --------------------------------------------

    def fact(self, precedent_id: str, facet_id: str, start: int, end: int) -> PrecedentQuote | None:
        """The stored fact quote at this span, or None (an index hit the file does not hold)."""
        for found in self._facts.get((precedent_id, facet_id), ()):
            if (found.quote.span.start, found.quote.span.end) == (start, end):
                return found.quote
        return None

    def passage_vectors(self) -> Iterator[tuple[PrecedentDoc, int, int, np.ndarray]]:
        """Every searchable paragraph with its precedent, span and vector (for the OpenSearch index)."""
        for (pid, start, end), vec in zip(self._passage_spans, self._passage_vecs, strict=True):
            yield self._docs[pid], start, end, vec

    def fact_vectors(self) -> Iterator[tuple[PrecedentDoc, str, PrecedentQuote, np.ndarray]]:
        """Every fact quote of a facet in the corpus that has a vector, with its precedent and facet id."""
        for pid, facets in self._facets.items():
            for facet in facets:
                for found in self._facts[(pid, facet.facet_id)]:
                    if found.vec is not None:
                        yield self._docs[pid], facet.facet_id, found.quote, found.vec


def query_words(words: str) -> frozenset[str]:
    return frozenset(_WORD.findall(words.lower())) - _STOP


def rank_passages(
    found: Sequence[tuple[tuple[str, int, int], float]], quote: Callable[[str, int, int], PrecedentQuote | None], words: str, k: int
) -> tuple[PassageHit, ...]:
    """Score each (passage span, cosine): the cosine, plus WORD_WEIGHT times the share of the query's
    words the passage holds; the best k, on a tie the earliest passage."""
    wanted = query_words(words)
    hits = []
    for (pid, start, end), raw_cosine in found:
        cosine = round(raw_cosine, COSINE_DECIMALS)  # where the two backends agree (float32 noise is about 1e-7)
        found_quote = quote(pid, start, end)
        if found_quote is None:
            continue
        share = len(wanted & set(_WORD.findall(found_quote.span.text.lower()))) / len(wanted) if wanted else 0.0
        score = (1 - WORD_WEIGHT) * cosine + WORD_WEIGHT * share if wanted else cosine
        hits.append(PassageHit(precedent_id=pid, quote=found_quote, cosine=round(cosine, 6), words=round(share, 6), score=round(score, 6)))
    return tuple(sorted(hits, key=lambda h: (-h.score, h.precedent_id, h.quote.span.start))[:k])
