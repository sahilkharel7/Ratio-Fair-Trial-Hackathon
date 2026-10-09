"""Which precedent index Similar cases reads: the corpus file, or OpenSearch on this computer when the
settings ask for it and it holds the same corpus build. It never fails: with no corpus it says what to
run, and with OpenSearch down it falls back to the file and says so in the status line."""

from __future__ import annotations

from ratio.config import FactPatternTaxonomy, PrecedentSettings
from ratio.paths import corpus_db_path
from ratio.precedent_opensearch import OpenSearchPrecedentIndex
from ratio.precedent_schema import PrecedentIndex
from ratio.precedent_store import CorpusUnavailable, SqlitePrecedentIndex

FILE_STATUS = "Corpus file"


def open_index(settings: PrecedentSettings, taxonomy: FactPatternTaxonomy) -> tuple[PrecedentIndex | None, str]:
    """The index to use and a one-line status, or (None, what to run) when there is no usable corpus."""
    try:
        corpus = SqlitePrecedentIndex(corpus_db_path(), taxonomy)
    except CorpusUnavailable as exc:
        return None, str(exc)
    if settings.backend != "opensearch":
        return corpus, FILE_STATUS
    search = OpenSearchPrecedentIndex(settings.opensearch_url, settings.opensearch_index, corpus)
    try:
        search.check()
    except CorpusUnavailable as exc:
        return corpus, f"{FILE_STATUS} ({exc})"
    return search, f"OpenSearch on {search.address}"
