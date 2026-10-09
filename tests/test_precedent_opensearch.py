"""The OpenSearch backend of Similar cases (ratio/precedent_opensearch.py) and its builder
(corpus_builder/opensearch.py). The unit tests use a fake client and never open a socket. The tests
marked ``opensearch`` index the SYNTHETIC fixture corpus into a temporary alias of the OpenSearch on
127.0.0.1:9200 (``docker compose up -d opensearch``), check that both backends return the same links,
and clean up after themselves; they are skipped when it is not running."""

import sys
import types
import uuid

import numpy as np
import pytest

from corpus_builder import opensearch as builder
from ratio import precedent_opensearch
from ratio.config import PrecedentSettings, load_config
from ratio.precedent_access import open_index
from ratio.precedent_opensearch import MAPPING, OpenSearchPrecedentIndex, available, cosine_from_score, index_body
from ratio.precedent_store import CorpusUnavailable, SqlitePrecedentIndex
from ratio.precedents import link
from tests.precedent_fixtures import BUILT_AT, DEMO_LIKE, LINKED, FakeEmbedder, build_fixture_corpus

URL = "http://127.0.0.1:9200"


@pytest.fixture(scope="module")
def taxonomy():
    return load_config().fact_patterns


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, taxonomy):
    return build_fixture_corpus(tmp_path_factory.mktemp("corpus") / "precedents.db", FakeEmbedder(), taxonomy)


@pytest.fixture(scope="module")
def sqlite_index(corpus, taxonomy):
    return SqlitePrecedentIndex(corpus, taxonomy)


class FakeIndices:
    def __init__(self, built_at: str | None = BUILT_AT, exists: bool = True):
        self.built_at, self._exists = built_at, exists

    def exists(self, index):
        return self._exists

    def get_mapping(self, index):
        return {f"{index}-v1": {"mappings": {"_meta": {"built_at": self.built_at}, **MAPPING}}}


class FakeClient:
    """Answers every search with ``hits``, or raises ``error``."""

    def __init__(self, hits=(), error: Exception | None = None, indices: FakeIndices | None = None):
        self.hits, self.error, self.indices, self.bodies = list(hits), error, indices or FakeIndices(), []

    def search(self, index, body, **kwargs):
        self.bodies.append(body)
        if self.error:
            raise self.error
        return {"hits": {"hits": self.hits}}

    def ping(self):
        return True


def facet_of(index, precedent_id: str, facet_id: str):
    facets = next(facets for doc, facets in index.candidates([facet_id]) if doc.id == precedent_id)
    return next(facet for facet in facets if facet.facet_id == facet_id)


def hit(pid: str, start: int, end: int, cosine: float) -> dict:
    return {"_score": (1 + cosine) / 2, "_source": {"precedent_id": pid, "facet_id": "gc35_48h", "start": start, "end": end}}


# --- unit: no socket is opened ----------------------------------------------------------------


def test_the_mapping_is_one_document_per_fact_with_a_cosine_hnsw_vector():
    vec = MAPPING["properties"]["vec"]
    assert vec["dimension"] == 384
    assert vec["method"] == {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}
    assert {name: MAPPING["properties"][name]["type"] for name in ("precedent_id", "facet_id", "kind", "start", "end", "text")} == {
        "precedent_id": "keyword", "facet_id": "keyword", "kind": "keyword", "start": "integer", "end": "integer", "text": "text",
    }  # fmt: skip


def test_the_index_records_which_corpus_build_it_holds(sqlite_index):
    body = index_body(sqlite_index.meta())
    assert body["settings"] == {"index": {"knn": True}}
    assert body["mappings"]["_meta"]["built_at"] == BUILT_AT


@pytest.mark.parametrize("url", ["http://10.0.0.5:9200", "https://search.example.org", "ftp://127.0.0.1:9200"])
def test_only_opensearch_on_this_computer_is_accepted(sqlite_index, url):
    with pytest.raises(ValueError, match="this computer"):
        OpenSearchPrecedentIndex(url, "precedent_passages", sqlite_index)
    assert available(url) is False


def test_lucene_scores_are_converted_back_to_cosines():
    assert [cosine_from_score(score) for score in (1.0, 0.5, 0.0, 0.8)] == pytest.approx([1.0, 0.0, -1.0, 0.6])


def test_nearest_keeps_the_best_hit_per_precedent_and_only_quotes_the_file_holds(sqlite_index):
    a_late = facet_of(sqlite_index, "tw-synthetic-a", "gc35_48h").facts[1].span
    c_late = facet_of(sqlite_index, "wgad-synthetic-c", "gc35_48h").facts[0].span
    hits = [hit("tw-synthetic-a", a_late.start, a_late.end, 0.4), hit("tw-synthetic-a", 1, 9, 0.99), hit("wgad-synthetic-c", c_late.start, c_late.end, 0.2)]
    client = FakeClient(hits)
    index = OpenSearchPrecedentIndex(URL, "precedent_passages", sqlite_index, client=client)
    found = index.nearest_facts("gc35_48h", np.ones(384, dtype=np.float32) / np.sqrt(384), ["tw-synthetic-a", "wgad-synthetic-c"])
    assert found["tw-synthetic-a"][0].span == a_late and found["tw-synthetic-a"][1] == pytest.approx(0.4)  # 1-9 is not a stored fact
    assert found["wgad-synthetic-c"][1] == pytest.approx(0.2)
    knn = client.bodies[0]["query"]["knn"]["vec"]
    assert knn["k"] == 10
    assert knn["filter"]["bool"]["filter"] == [{"term": {"facet_id": "gc35_48h"}}, {"terms": {"precedent_id": ["tw-synthetic-a", "wgad-synthetic-c"]}}]


def test_a_failed_search_falls_back_to_the_file(sqlite_index):
    from opensearchpy.exceptions import ConnectionError as SearchConnectionError

    client = FakeClient(error=SearchConnectionError("N/A", "refused", None))
    index = OpenSearchPrecedentIndex(URL, "precedent_passages", sqlite_index, client=client)
    query = FakeEmbedder().encode(["arrested and brought before a judge five days later"])[0]
    assert index.nearest_facts("gc35_48h", query, ["tw-synthetic-a"]) == sqlite_index.nearest_facts("gc35_48h", query, ["tw-synthetic-a"])


def test_precedents_the_knn_search_missed_are_answered_by_the_file(sqlite_index):
    """kNN is approximate: a precedent with facts on the facet but no hit gets the file's answer, as with SQLite."""
    a_late = facet_of(sqlite_index, "tw-synthetic-a", "gc35_48h").facts[1].span
    query = FakeEmbedder().encode(["arrested and brought before a judge five days later"])[0]
    index = OpenSearchPrecedentIndex(URL, "precedent_passages", sqlite_index, client=FakeClient([hit("tw-synthetic-a", a_late.start, a_late.end, 0.4)]))
    found = index.nearest_facts("gc35_48h", query, ["tw-synthetic-a", "wgad-synthetic-c"])
    assert found["tw-synthetic-a"][0].span == a_late
    assert found["wgad-synthetic-c"] == sqlite_index.nearest_facts("gc35_48h", query, ["wgad-synthetic-c"])["wgad-synthetic-c"]


def test_the_first_failed_search_sends_the_rest_of_one_link_to_the_file(sqlite_index, taxonomy):
    from opensearchpy.exceptions import ConnectionTimeout

    client = FakeClient(error=ConnectionTimeout("TIMEOUT", "timed out", None))
    index = OpenSearchPrecedentIndex(URL, "precedent_passages", sqlite_index, client=client)
    links = link(DEMO_LIKE, index, FakeEmbedder(), PrecedentSettings(), taxonomy)
    assert len(client.bodies) == 1  # one timeout, not one per shared fact pattern
    assert links == link(DEMO_LIKE, sqlite_index, FakeEmbedder(), PrecedentSettings(), taxonomy)
    link(DEMO_LIKE, index, FakeEmbedder(), PrecedentSettings(), taxonomy)
    assert len(client.bodies) == 2  # the next link() call tries OpenSearch again


@pytest.fixture
def without_opensearchpy(monkeypatch):
    """Ratio installed without opensearch-py: importing it fails."""
    for name in [name for name in sys.modules if name == "opensearchpy" or name.startswith("opensearchpy.")] + ["opensearchpy"]:
        monkeypatch.setitem(sys.modules, name, None)


def test_without_opensearch_py_the_file_is_used(without_opensearchpy, corpus, sqlite_index, taxonomy, monkeypatch):
    assert available(URL) is False
    with pytest.raises(CorpusUnavailable, match="opensearch-py is not installed"):
        precedent_opensearch.connect(URL)
    index = OpenSearchPrecedentIndex(URL, "precedent_passages", sqlite_index)
    with pytest.raises(CorpusUnavailable, match="opensearch-py is not installed"):
        index.check()
    query = FakeEmbedder().encode(["arrested and brought before a judge five days later"])[0]
    assert index.nearest_facts("gc35_48h", query, ["tw-synthetic-a"]) == sqlite_index.nearest_facts("gc35_48h", query, ["tw-synthetic-a"])
    monkeypatch.setenv("RATIO_CORPUS_DB", str(corpus))
    found, status = open_index(PrecedentSettings(backend="opensearch"), taxonomy)
    assert isinstance(found, SqlitePrecedentIndex) and status == "Corpus file (opensearch-py is not installed)"


def test_searches_time_out_after_two_seconds(monkeypatch):
    made = []
    monkeypatch.setitem(sys.modules, "opensearchpy", types.SimpleNamespace(OpenSearch=lambda **kwargs: made.append(kwargs)))
    precedent_opensearch.connect(URL)
    assert precedent_opensearch.QUERY_TIMEOUT_S == 2.0 and made[0]["timeout"] == 2.0
    assert (made[0]["max_retries"], made[0]["retry_on_timeout"]) == (0, False)


@pytest.mark.parametrize(
    ("indices", "says"),
    [(FakeIndices(exists=False), "index not built"), (FakeIndices(built_at="2020-01-01T00:00:00+00:00"), "out of date")],
)
def test_check_refuses_a_missing_or_stale_index(sqlite_index, monkeypatch, indices, says):
    monkeypatch.setattr(precedent_opensearch, "available", lambda url, timeout=1.0: True)
    index = OpenSearchPrecedentIndex(URL, "precedent_passages", sqlite_index, client=FakeClient(indices=indices))
    with pytest.raises(CorpusUnavailable, match=says):
        index.check()


def test_the_other_reads_come_from_the_file(sqlite_index):
    index = OpenSearchPrecedentIndex(URL, "precedent_passages", sqlite_index, client=FakeClient())
    assert index.meta() == sqlite_index.meta()
    assert index.facet_doc_freq() == sqlite_index.facet_doc_freq()
    assert index.candidates(["gc35_48h"]) == sqlite_index.candidates(["gc35_48h"])
    assert index.text("precedent-tw-synthetic-a/text") == sqlite_index.text("precedent-tw-synthetic-a/text")


# --- integration: OpenSearch on 127.0.0.1:9200 ---------------------------------------------------

running = pytest.mark.skipif(not available(URL), reason="OpenSearch is not running on 127.0.0.1:9200 (docker compose up -d opensearch)")


@pytest.fixture
def alias():
    name = f"ratio-test-{uuid.uuid4().hex[:12]}"
    yield name
    client = precedent_opensearch.connect(URL)
    client.indices.delete(index=f"{name}-v*", expand_wildcards="all", ignore_unavailable=True)


def comparable(links):
    """What a reader sees of each link; its cosines are kept to COSINE_DECIMALS, where the backends agree."""
    return [
        (found.precedent.id, round(found.score.idf_sum, 9), found.score.shared, found.score.mean_pair_cosine,
         [(s.facet_id, s.precedent_fact, s.precedent_finding, s.pair_cosine) for s in found.shared])
        for found in links
    ]  # fmt: skip


@pytest.mark.opensearch
@running
def test_both_backends_return_the_same_links(corpus, sqlite_index, taxonomy, alias):
    indexed = builder.index_corpus(corpus, URL, alias, taxonomy=taxonomy)
    assert indexed == sum(1 for _ in sqlite_index.fact_vectors())
    search = OpenSearchPrecedentIndex(URL, alias, sqlite_index)
    search.check()
    for settings in (PrecedentSettings(), PrecedentSettings(min_shared=1), PrecedentSettings(top_k=2)):
        expected = link(DEMO_LIKE, sqlite_index, FakeEmbedder(), settings, taxonomy)
        assert comparable(link(DEMO_LIKE, search, FakeEmbedder(), settings, taxonomy)) == comparable(expected)
    assert [found.precedent.id for found in link(DEMO_LIKE, search, FakeEmbedder(), PrecedentSettings(), taxonomy)] == list(LINKED)


@pytest.mark.opensearch
@running
def test_reindexing_moves_the_alias_and_removes_the_old_version(corpus, taxonomy, alias):
    builder.index_corpus(corpus, URL, alias, taxonomy=taxonomy)
    builder.index_corpus(corpus, URL, alias, taxonomy=taxonomy)
    client = precedent_opensearch.connect(URL)
    assert sorted(client.indices.get(index=f"{alias}-v*")) == [f"{alias}-v2"]
    assert list(client.indices.get_alias(name=alias)) == [f"{alias}-v2"]


@pytest.mark.opensearch
@running
def test_open_index_uses_opensearch_and_falls_back_when_the_index_is_gone(corpus, taxonomy, alias, monkeypatch):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(corpus))
    settings = PrecedentSettings(backend="opensearch", opensearch_index=alias)
    index, status = open_index(settings, taxonomy)
    assert isinstance(index, SqlitePrecedentIndex) and status.startswith("Corpus file (OpenSearch index not built")
    builder.index_corpus(corpus, URL, alias, taxonomy=taxonomy)
    index, status = open_index(settings, taxonomy)
    assert isinstance(index, OpenSearchPrecedentIndex) and status == "OpenSearch on 127.0.0.1:9200"


@pytest.mark.opensearch
@running
def test_a_snapshot_restores_the_index(corpus, sqlite_index, taxonomy, alias):
    builder.index_corpus(corpus, URL, alias, taxonomy=taxonomy)
    name = f"{alias}-snapshot"
    try:
        builder.snapshot(URL, name, alias=alias)
        precedent_opensearch.connect(URL).indices.delete(index=f"{alias}-v*")
        assert builder.restore(URL, name, alias=alias) == f"{alias}-v1"
        search = OpenSearchPrecedentIndex(URL, alias, sqlite_index)
        search.check()
        assert [found.precedent.id for found in link(DEMO_LIKE, search, FakeEmbedder(), PrecedentSettings(), taxonomy)] == list(LINKED)
    finally:
        precedent_opensearch.connect(URL).snapshot.delete(repository=builder.SNAPSHOT_REPOSITORY, snapshot=name, ignore=[404])
