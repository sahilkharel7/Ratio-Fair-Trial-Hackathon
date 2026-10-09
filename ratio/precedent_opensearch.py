"""Similar cases on OpenSearch running on this computer (docker compose, 127.0.0.1:9200).

The vector search runs in OpenSearch: one document per verified fact quote, with the vector of the
sentence around it. Every text, document, facet and quote still comes from precedents.db, the source of
truth, so both backends return the same links and the provenance check runs against the same text.
OpenSearch must be on this computer (hard rule 1). This is the only module that imports opensearchpy,
and it does so inside functions, so Ratio runs without it.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from ratio.embeddings import EMBEDDING_DIM
from ratio.netguard import is_loopback_host
from ratio.precedent_schema import CorpusMeta, PrecedentDoc, PrecedentFacet, PrecedentQuote
from ratio.precedent_store import CorpusUnavailable, SqlitePrecedentIndex

_log = logging.getLogger(__name__)

HEALTH_TIMEOUT_S = 1.0
QUERY_TIMEOUT_S = 2.0  # a search that takes longer is answered from the corpus file instead
MIN_K = 10  # candidate hits per query, at least
K_PER_PRECEDENT = 4  # ... and this many per precedent searched
INDEX_COMMAND = "python -m corpus_builder index-opensearch"
NOT_INSTALLED = "opensearch-py is not installed"

INDEX_SETTINGS: dict[str, Any] = {"index": {"knn": True}}
MAPPING: dict[str, Any] = {
    "properties": {
        "precedent_id": {"type": "keyword"},
        "facet_id": {"type": "keyword"},
        "kind": {"type": "keyword"},
        "start": {"type": "integer"},
        "end": {"type": "integer"},
        "text": {"type": "text"},
        "vec": {
            "type": "knn_vector",
            "dimension": EMBEDDING_DIM,
            "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
        },
    }
}


def index_body(meta: CorpusMeta) -> dict[str, Any]:
    """The body that creates an index of this corpus build; its _meta tells a stale index apart."""
    built = {"built_at": meta.built_at, "taxonomy_sha": meta.taxonomy_sha, "embed_model": meta.embed_model}
    return {"settings": INDEX_SETTINGS, "mappings": {"_meta": built, **MAPPING}}


def checked_url(url: str) -> str:
    """The URL, if it names OpenSearch on this computer; otherwise ValueError."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not is_loopback_host(parts.hostname or ""):
        raise ValueError(f"OpenSearch must run on this computer (127.0.0.1), not {url!r}")
    return url


def connect(url: str, timeout: float = QUERY_TIMEOUT_S) -> Any:
    """An opensearch-py client for OpenSearch on this computer, with no retries.

    CorpusUnavailable when opensearch-py is not installed: Ratio runs without it, on the corpus file."""
    url = checked_url(url)
    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:
        raise CorpusUnavailable(NOT_INSTALLED) from exc
    return OpenSearch(hosts=[url], timeout=timeout, max_retries=0, retry_on_timeout=False)


def request_errors() -> tuple[type[Exception], ...]:
    """What a failed request to OpenSearch raises; CorpusUnavailable when there is no client to make one."""
    try:
        from opensearchpy.exceptions import OpenSearchException
    except ImportError:
        return (CorpusUnavailable, OSError)
    return (CorpusUnavailable, OpenSearchException, OSError)


def available(url: str, timeout: float = HEALTH_TIMEOUT_S) -> bool:
    """Whether OpenSearch answers at this local URL within ``timeout`` seconds (False without opensearch-py)."""
    try:
        client = connect(url, timeout)
    except (ValueError, CorpusUnavailable):
        return False
    try:
        return bool(client.ping())
    except request_errors() as exc:
        _log.info("OpenSearch at %s did not answer: %s", url, exc)
        return False


def cosine_from_score(score: float) -> float:
    """Lucene's cosinesimil score is (1 + cosine) / 2."""
    return float(np.clip(2.0 * float(score) - 1.0, -1.0, 1.0))


class OpenSearchPrecedentIndex:
    """PrecedentIndex whose nearest_facts runs in OpenSearch; everything else reads ``sqlite``."""

    def __init__(self, url: str, index: str, sqlite: SqlitePrecedentIndex, *, client: Any = None) -> None:
        self.url = checked_url(url)
        self.index = index
        self._sqlite = sqlite
        self._client = client

    @property
    def address(self) -> str:
        return urlsplit(self.url).netloc

    def client(self) -> Any:
        if self._client is None:
            self._client = connect(self.url)
        return self._client

    def check(self) -> None:
        """Raise CorpusUnavailable, with a one-line reason, unless the index holds this corpus build."""
        self.client()  # CorpusUnavailable when opensearch-py is not installed
        if not available(self.url):
            raise CorpusUnavailable("OpenSearch not running")
        try:
            built = self._built_at()
        except request_errors() as exc:
            _log.warning("OpenSearch at %s could not describe index %s: %s", self.url, self.index, exc)
            raise CorpusUnavailable("OpenSearch not answering") from exc
        if not built:
            raise CorpusUnavailable(f"OpenSearch index not built; run `{INDEX_COMMAND}`")
        if built != {self._sqlite.meta().built_at}:
            raise CorpusUnavailable(f"OpenSearch index out of date; run `{INDEX_COMMAND}`")

    def _built_at(self) -> set[str | None]:
        """The corpus builds the index (or the indices behind the alias) hold; empty when there is none."""
        indices = self.client().indices
        if not indices.exists(index=self.index):
            return set()
        return {found["mappings"].get("_meta", {}).get("built_at") for found in indices.get_mapping(index=self.index).values()}

    # --- PrecedentIndex: the file answers everything but the vector search ---------------------

    def meta(self) -> CorpusMeta:
        return self._sqlite.meta()

    def facet_doc_freq(self) -> Mapping[str, int]:
        return self._sqlite.facet_doc_freq()

    def candidates(self, facet_ids: Collection[str]) -> Sequence[tuple[PrecedentDoc, tuple[PrecedentFacet, ...]]]:
        return self._sqlite.candidates(facet_ids)

    def text(self, doc_id: str) -> str | None:
        return self._sqlite.text(doc_id)

    def nearest_facts(self, facet_id: str, query: np.ndarray, precedent_ids: Collection[str]) -> Mapping[str, tuple[PrecedentQuote, float]]:
        """For each precedent, its fact on this facet closest to the query; the file answers if the search fails."""
        return self.per_call().nearest_facts(facet_id, query, precedent_ids)

    def per_call(self) -> OneCall:
        """A view of this index for one link() call (see OneCall)."""
        return OneCall(self, self._sqlite)

    def search_nearest(self, facet_id: str, query: np.ndarray, ids: list[str]) -> dict[str, tuple[PrecedentQuote, float]]:
        """The kNN answer; a precedent it missed (kNN is approximate) gets the file's, so both backends agree.
        Raises what request_errors() names when the search fails."""
        found = self._best_per_precedent(facet_id, self._search(facet_id, query, ids))
        missed = [pid for pid in ids if pid not in found]
        return {**found, **self._sqlite.nearest_facts(facet_id, query, missed)} if missed else found

    def _search(self, facet_id: str, query: np.ndarray, ids: list[str]) -> list[dict[str, Any]]:
        k = max(MIN_K, len(ids) * K_PER_PRECEDENT)
        only = {"bool": {"filter": [{"term": {"facet_id": facet_id}}, {"terms": {"precedent_id": ids}}]}}
        vector = np.asarray(query, dtype=np.float32).ravel().tolist()
        body = {
            "size": k,
            "_source": ["precedent_id", "start", "end"],
            "query": {"knn": {"vec": {"vector": vector, "k": k, "filter": only}}},
        }
        return self.client().search(index=self.index, body=body)["hits"]["hits"]

    def _best_per_precedent(self, facet_id: str, hits: list[dict[str, Any]]) -> dict[str, tuple[PrecedentQuote, float]]:
        """The best hit per precedent that the file holds; on a tie, the fact earliest in the text."""
        found: dict[str, tuple[PrecedentQuote, float]] = {}
        for hit in hits:
            source = hit["_source"]
            quote = self._sqlite.fact(source["precedent_id"], facet_id, source["start"], source["end"])
            if quote is None:
                continue
            cosine = cosine_from_score(hit["_score"])
            best = found.get(source["precedent_id"])
            if best is None or (cosine, -quote.span.start) > (best[1], -best[0].span.start):
                found[source["precedent_id"]] = (quote, cosine)
        return found


class OneCall:
    """OpenSearchPrecedentIndex as one link() call reads it. After the first failed search, the corpus
    file answers the rest of the call, so an OpenSearch that stalls costs one timeout per call, not one
    per fact pattern. link() makes a fresh one for every call, so the next call tries OpenSearch again;
    nothing is shared between the Streamlit sessions that share the index."""

    def __init__(self, search: OpenSearchPrecedentIndex, sqlite: SqlitePrecedentIndex) -> None:
        self._search = search
        self._sqlite = sqlite
        self.failed = False

    def meta(self) -> CorpusMeta:
        return self._sqlite.meta()

    def facet_doc_freq(self) -> Mapping[str, int]:
        return self._sqlite.facet_doc_freq()

    def candidates(self, facet_ids: Collection[str]) -> Sequence[tuple[PrecedentDoc, tuple[PrecedentFacet, ...]]]:
        return self._sqlite.candidates(facet_ids)

    def text(self, doc_id: str) -> str | None:
        return self._sqlite.text(doc_id)

    def nearest_facts(self, facet_id: str, query: np.ndarray, precedent_ids: Collection[str]) -> Mapping[str, tuple[PrecedentQuote, float]]:
        """OpenSearch's answer until a search fails; from then on, the file's."""
        ids = sorted(set(precedent_ids))
        if not ids:
            return {}
        if not self.failed:
            try:
                return self._search.search_nearest(facet_id, query, ids)
            except request_errors() as exc:
                _log.warning("OpenSearch search failed (%s); answering the rest of this request from the corpus file", exc)
                self.failed = True
        return self._sqlite.nearest_facts(facet_id, query, ids)
