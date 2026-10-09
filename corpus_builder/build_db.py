"""Write data/corpus/precedents.db, the file the app reads, from the build store.

The corpus is written whole into a temporary file next to the target and moved into place with
os.replace, so the app never sees a half-written corpus and a failed build leaves the old one.
It holds the documents with at least one verified facet, their facets and quotes (fact quotes with
their vectors), the sources they came from, and the meta the runtime checks before using it.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from corpus_builder.store import BuildStore
from corpus_builder.verify import check_span, read_only
from ratio.config import FactPatternTaxonomy
from ratio.embeddings import EMBEDDING_DIM
from ratio.paths import corpus_db_path
from ratio.precedent_schema import (
    EMBED_MODEL_TAG,
    META_KEYS,
    SCHEMA_VERSION,
    SQLITE_DDL,
    CorpusMeta,
    CorpusSource,
    PrecedentDoc,
    PrecedentFacet,
)

log = logging.getLogger(__name__)

Corpus = tuple[tuple[PrecedentDoc, tuple[PrecedentFacet, ...]], ...]
Vectors = Mapping[tuple[str, str, int, int], np.ndarray]
Verified = Mapping[str, tuple[tuple[PrecedentFacet, ...], int]]  # BuildStore.verified()
_DOCUMENT_COLUMNS = tuple(PrecedentDoc.model_fields)  # the documents table has one column per field


class BuildError(RuntimeError):
    """The build store is not ready to become a corpus; the message says which step to run."""


class SourceInfo(Protocol):
    """The fields of corpus_builder.sources.Source read here."""

    kind: str
    name: str
    terms_url: str
    terms_checked: object  # an ISO date or its string
    attribution: str
    allowed_prefixes: Sequence[str]


# --- what goes in ------------------------------------------------------------------------------


def _check_facets(doc: PrecedentDoc, facets: tuple[PrecedentFacet, ...], known: Collection[str]) -> None:
    for facet in facets:
        if facet.facet_id not in known:
            raise BuildError(f"{doc.id}: facet {facet.facet_id!r} is not in the taxonomy; run `python -m corpus_builder verify`")
        for quote in (*facet.facts, *([facet.finding] if facet.finding else [])):
            check_span(doc, quote.span)


def _corpus(store: BuildStore, verified: Verified, taxonomy: FactPatternTaxonomy) -> Corpus:
    """The documents with at least one verified facet, each span re-checked against its text."""
    known = {pattern.id for pattern in taxonomy.facets}
    corpus = []
    for precedent_id, (facets, _) in sorted(verified.items()):
        if not facets:
            continue
        doc = store.document(precedent_id)
        if doc is None:
            log.warning("%s: verified but no longer in the build store; left out", precedent_id)
            continue
        _check_facets(doc, facets, known)
        corpus.append((doc, facets))
    if not corpus:
        raise BuildError("no document has a verified facet; run `python -m corpus_builder extract`, then `verify`")
    return tuple(corpus)


def _require_vectors(corpus: Corpus, vectors: Vectors) -> None:
    for doc, facets in corpus:
        for facet in facets:
            for quote in facet.facts:
                vector = vectors.get((doc.id, facet.facet_id, quote.span.start, quote.span.end))
                if vector is None or vector.shape != (EMBEDDING_DIM,):
                    raise BuildError(f"{doc.id}/{facet.facet_id}: a fact quote has no vector; run `python -m corpus_builder embed`")


def _source_of(doc: PrecedentDoc, sources: Sequence[SourceInfo]) -> SourceInfo:
    for source in sources:
        same_origin = doc.attribution == source.attribution or doc.url.startswith(tuple(source.allowed_prefixes))
        if source.kind == doc.kind and same_origin:
            return source
    raise BuildError(f"{doc.id}: no source in sources.yaml matches this document (kind {doc.kind}, {doc.url})")


def _distinct(values: Iterable[str]) -> str:
    return "; ".join(dict.fromkeys(values))


def _merged(kind: str, group: Sequence[SourceInfo], documents: int) -> CorpusSource:
    return CorpusSource(
        kind=kind,
        name=_distinct(s.name for s in group),
        terms_url=_distinct(s.terms_url for s in group),
        terms_checked=min(str(s.terms_checked) for s in group),  # the oldest check is the one to trust
        attribution=_distinct(s.attribution for s in group),
        documents=documents,
    )


def corpus_sources(sources: Sequence[SourceInfo], docs: Sequence[PrecedentDoc]) -> tuple[CorpusSource, ...]:
    """One row per kind (the sources table's key), in sources.yaml order: the sources of that kind
    that gave documents, merged, with the number of documents of that kind."""
    origins = [_source_of(doc, sources) for doc in docs]
    used = [source for source in sources if source in origins]
    counts = Counter(doc.kind for doc in docs)
    kinds = dict.fromkeys(source.kind for source in used)
    return tuple(_merged(kind, [s for s in used if s.kind == kind], counts[kind]) for kind in kinds)


# --- writing -----------------------------------------------------------------------------------


def _meta_rows(meta: CorpusMeta) -> list[tuple[str, str]]:
    values = {key: str(getattr(meta, key)) for key in META_KEYS}
    return sorted(values.items())


def _quote_rows(corpus: Corpus, vectors: Vectors) -> Iterator[tuple]:
    for doc, facets in corpus:
        for facet in facets:
            for quote in facet.facts:
                vector = vectors[(doc.id, facet.facet_id, quote.span.start, quote.span.end)]
                blob = np.asarray(vector, dtype=np.float32).tobytes()
                yield (doc.id, facet.facet_id, "fact", quote.span.start, quote.span.end, quote.match, blob)
            if facet.finding is not None:
                span = facet.finding.span
                yield (doc.id, facet.facet_id, "finding", span.start, span.end, facet.finding.match, None)


def _fill(db: sqlite3.Connection, corpus: Corpus, vectors: Vectors, meta: CorpusMeta) -> None:
    for statement in SQLITE_DDL:
        db.execute(statement)
    columns, marks = ", ".join(_DOCUMENT_COLUMNS), ", ".join("?" * len(_DOCUMENT_COLUMNS))
    documents = [tuple(getattr(doc, column) for column in _DOCUMENT_COLUMNS) for doc, _ in corpus]
    facets = [(doc.id, facet.facet_id, facet.finding_kind) for doc, doc_facets in corpus for facet in doc_facets]
    sources = [(s.kind, s.name, s.terms_url, s.terms_checked, s.attribution) for s in meta.sources]
    db.executemany(f"INSERT INTO documents ({columns}) VALUES ({marks})", documents)  # noqa: S608 - the model's field names
    db.executemany("INSERT INTO facets VALUES (?, ?, ?)", facets)
    db.executemany("INSERT INTO quotes VALUES (?, ?, ?, ?, ?, ?, ?)", _quote_rows(corpus, vectors))
    db.executemany("INSERT INTO sources VALUES (?, ?, ?, ?, ?)", sources)
    db.executemany("INSERT INTO meta VALUES (?, ?)", _meta_rows(meta))


def _write_atomically(out: Path, corpus: Corpus, vectors: Vectors, meta: CorpusMeta) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        with closing(sqlite3.connect(temporary)) as db:
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                _fill(db, corpus, vectors, meta)
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)


# --- the step ----------------------------------------------------------------------------------


def build_db(
    store: BuildStore,
    taxonomy: FactPatternTaxonomy,
    sources: Sequence[SourceInfo],
    out_path: Path | None = None,
    *,
    extraction_model: str,
    prompt_sha: str,
) -> CorpusMeta:
    """Write a fresh corpus file (default: corpus_db_path(), which RATIO_CORPUS_DB overrides)."""
    verified = store.verified()
    corpus = _corpus(store, verified, taxonomy)
    vectors = store.vectors()
    _require_vectors(corpus, vectors)
    meta = CorpusMeta(
        schema_version=SCHEMA_VERSION,
        taxonomy_sha=taxonomy.sha,
        embed_model=EMBED_MODEL_TAG,
        extraction_model=extraction_model,
        prompt_sha=prompt_sha,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        documents=len(corpus),
        facets=sum(len(facets) for _, facets in corpus),
        dropped_quotes=sum(dropped for _, dropped in verified.values()),
        sources=corpus_sources(sources, [doc for doc, _ in corpus]),
    )
    out = Path(out_path) if out_path is not None else corpus_db_path()
    _write_atomically(out, corpus, vectors, meta)
    log.info("wrote %s: %d documents, %d facets", out, meta.documents, meta.facets)
    return meta


def extraction_models(store: BuildStore) -> str:
    """The model(s) whose answers the verified facets come from, for the corpus meta."""
    with closing(read_only(store)) as db:
        rows = db.execute(
            "SELECT DISTINCT e.model FROM verified v JOIN extractions e ON e.cache_key = v.cache_key ORDER BY e.model"
        ).fetchall()
    return ", ".join(row[0] for row in rows) or "none"
