"""The precedent corpus file, read-only (ratio/precedent_store.py), and the backend choice
(ratio/precedent_access.py): a corpus built for another schema, taxonomy or embedding model is refused
with the command to run, and the app falls back to the file when OpenSearch is not there."""

import hashlib
import sqlite3
from contextlib import closing
from typing import get_args

import numpy as np
import pytest

from ratio import precedent_opensearch, precedent_store
from ratio.config import PrecedentSettings, load_config
from ratio.precedent_access import open_index
from ratio.precedent_schema import ALLOWED_FINDINGS, EMBED_MODEL_TAG, SCHEMA_VERSION, FindingKind, PrecedentKind, precedent_doc_id
from ratio.precedent_store import UNSTATED, CorpusUnavailable, SqlitePrecedentIndex
from tests.precedent_fixtures import BUILT_AT, PRECEDENTS, TAMPERED, FakeEmbedder, build_fixture_corpus, set_meta


@pytest.fixture(scope="module")
def taxonomy():
    return load_config().fact_patterns


@pytest.fixture
def corpus(tmp_path, taxonomy):
    return build_fixture_corpus(tmp_path / "precedents.db", FakeEmbedder(), taxonomy)


@pytest.fixture
def index(corpus, taxonomy):
    return SqlitePrecedentIndex(corpus, taxonomy)


def file_sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_corpus_is_opened_read_only(corpus, taxonomy):
    before = file_sha(corpus)
    index = SqlitePrecedentIndex(corpus, taxonomy)
    index.candidates(["gc35_48h"])
    index.nearest_facts("gc35_48h", FakeEmbedder().encode(["brought before a judge"])[0], ["tw-synthetic-a"])
    assert file_sha(corpus) == before
    assert sorted(p.name for p in corpus.parent.iterdir()) == ["precedents.db"]  # no journal, no WAL


def test_a_missing_corpus_is_refused_and_never_created(tmp_path, taxonomy):
    path = tmp_path / "corpus" / "precedents.db"
    with pytest.raises(CorpusUnavailable, match="python -m corpus_builder"):
        SqlitePrecedentIndex(path, taxonomy)
    assert not path.exists()


def test_a_file_that_is_not_a_corpus_is_refused(tmp_path, taxonomy):
    path = tmp_path / "precedents.db"
    path.write_text("SYNTHETIC: not a database", encoding="utf-8")
    with pytest.raises(CorpusUnavailable):
        SqlitePrecedentIndex(path, taxonomy)


@pytest.mark.parametrize(
    ("key", "value", "says"),
    [
        ("schema_version", str(SCHEMA_VERSION + 1), "schema version"),
        ("taxonomy_sha", "0" * 64, "fact patterns"),
        ("embed_model", "another-model/768", "embedding model"),
        ("embed_model", None, "embed_model"),
    ],
)
def test_a_corpus_built_for_something_else_is_refused_with_the_command(corpus, taxonomy, key, value, says):
    set_meta(corpus, key, value)
    with pytest.raises(CorpusUnavailable, match=says) as caught:
        SqlitePrecedentIndex(corpus, taxonomy)
    assert "python -m corpus_builder" in str(caught.value)


BUILD_STEPS = ("fetch", "normalize", "extract", "verify", "embed", "build-db")
DAMAGE = {
    "missing": lambda path: path.unlink(),
    "not-a-database": lambda path: path.write_text("SYNTHETIC: not a database", encoding="utf-8"),
    "schema-version": lambda path: set_meta(path, "schema_version", str(SCHEMA_VERSION + 1)),
    "no-built-at": lambda path: set_meta(path, "built_at", None),
    "other-fact-patterns": lambda path: set_meta(path, "taxonomy_sha", "0" * 64),
    "other-embedding-model": lambda path: set_meta(path, "embed_model", "another-model/768"),
}


def names_the_build_chain(message: str) -> bool:
    """Every step of the build, in order, after the command: never build-db alone."""
    steps = message.partition("python -m corpus_builder")[2]
    positions = [steps.find(step) for step in BUILD_STEPS]
    return -1 not in positions and positions == sorted(positions)


@pytest.mark.parametrize("damage", DAMAGE.values(), ids=DAMAGE.keys())
def test_a_corpus_that_cannot_be_used_names_the_whole_build_chain(corpus, taxonomy, damage):
    damage(corpus)
    with pytest.raises(CorpusUnavailable) as caught:
        SqlitePrecedentIndex(corpus, taxonomy)
    assert names_the_build_chain(str(caught.value)), str(caught.value)


def test_a_changed_taxonomy_is_refused(corpus, taxonomy):
    first = taxonomy.facets[0]
    changed = taxonomy.model_copy(update={"facets": (first.model_copy(update={"label": "Changed"}), *taxonomy.facets[1:])})
    with pytest.raises(CorpusUnavailable, match="fact patterns"):
        SqlitePrecedentIndex(corpus, changed)


def test_meta_describes_the_build(index, taxonomy):
    meta = index.meta()
    assert (meta.schema_version, meta.taxonomy_sha, meta.embed_model, meta.built_at) == (SCHEMA_VERSION, taxonomy.sha, EMBED_MODEL_TAG, BUILT_AT)
    assert meta.documents == len(PRECEDENTS)
    assert meta.facets == sum(len(p.facets) for p in PRECEDENTS)
    assert meta.dropped_quotes == 2
    counts = {source.kind: source.documents for source in meta.sources}
    assert counts == {"ccpr_views": 2, "wgad_opinion": 1, "trialwatch_report": 3}


def test_the_default_path_follows_the_environment(corpus, taxonomy, monkeypatch):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(corpus))
    assert SqlitePrecedentIndex(taxonomy=taxonomy).meta().documents == len(PRECEDENTS)


def test_candidates_are_the_precedents_with_any_of_the_facets(index, taxonomy):
    found = index.candidates(["renewal_review", "iccpr_14_3_f"])
    assert [doc.id for doc, _ in found] == ["ccpr-synthetic-f", "wgad-synthetic-c"]
    order = [facet.id for facet in taxonomy.facets]
    for _, facets in found:
        ids = [facet.facet_id for facet in facets]
        assert ids == sorted(ids, key=order.index)  # all its facets, in taxonomy order
    wgad = next(facets for doc, facets in found if doc.id == "wgad-synthetic-c")
    assert [facet.facet_id for facet in wgad] == ["gc35_48h", "renewal_review", "media_defendant"]


def test_quotes_are_spans_of_the_stored_text(index):
    for doc, facets in index.candidates(["gc35_48h", "media_defendant", "iccpr_14_3_e"]):
        for facet in facets:
            for quote in (*facet.facts, *([facet.finding] if facet.finding else [])):
                assert quote.span.doc_id == precedent_doc_id(doc.id)
                assert doc.text[quote.span.start : quote.span.end] == quote.span.text


def test_facet_doc_freq_counts_precedents(index):
    freq = index.facet_doc_freq()
    assert (freq["gc35_48h"], freq["media_defendant"], freq["renewal_review"], freq["iccpr_14_3_e"]) == (3, 4, 1, 2)
    assert "charge_extremism" not in freq


def test_nearest_facts_picks_the_closest_fact_per_precedent(index):
    embedder = FakeEmbedder()
    query = embedder.encode(["The accused was arrested at his home and was brought before a judge five days later."])[0]
    found = index.nearest_facts("gc35_48h", query, ["tw-synthetic-a", "wgad-synthetic-c", "ccpr-synthetic-b"])
    assert set(found) == {"tw-synthetic-a", "wgad-synthetic-c"}  # ccpr-synthetic-b has no gc35_48h fact
    quote, cosine = found["tw-synthetic-a"]
    assert quote.span.text == "She was arrested at her home and was brought before a judge six days later."
    expected = float(embedder.encode([quote.span.text])[0] @ query)
    assert cosine == pytest.approx(expected, abs=1e-6)
    assert index.nearest_facts("gc35_48h", query, []) == {}


def test_text_resolves_precedent_doc_ids_only(index):
    doc = PRECEDENTS[0]
    assert index.text(precedent_doc_id(doc.id)).startswith("SYNTHETIC:")
    assert index.text(precedent_doc_id("unknown")) is None
    assert index.text("venn-2025/judgment.txt") is None


def test_a_text_edited_after_verification_resolves_to_nothing(index):
    """Its sha256 no longer matches, so no quote of it can pass the provenance check."""
    assert index.text(precedent_doc_id(TAMPERED)) is None
    assert any(doc.id == TAMPERED for doc, _ in index.candidates(["gc35_48h"]))


def test_an_invalid_row_is_refused_not_raised(corpus, taxonomy):
    with closing(sqlite3.connect(corpus)) as db, db:
        db.execute("UPDATE facets SET finding_kind = 'guilty' WHERE precedent_id = 'wgad-synthetic-c' AND facet_id = 'gc35_48h'")
    with pytest.raises(CorpusUnavailable, match="invalid row"):
        SqlitePrecedentIndex(corpus, taxonomy)


@pytest.mark.parametrize(
    ("precedent_id", "wrong_kind"),
    [("tw-synthetic-a", "violation_found"), ("tw-synthetic-a", "no_violation"), ("wgad-synthetic-c", "monitor_assessment")],
)
def test_a_finding_its_kind_of_document_never_states_is_dropped_and_the_facts_kept(corpus, taxonomy, precedent_id, wrong_kind):
    """A TrialWatch report assesses, it finds no violation; Views or an opinion hold no monitor's assessment."""
    with closing(sqlite3.connect(corpus)) as db, db:
        db.execute("UPDATE facets SET finding_kind = ? WHERE precedent_id = ? AND facet_id = 'gc35_48h'", (wrong_kind, precedent_id))
    facets = {doc.id: {f.facet_id: f for f in found} for doc, found in SqlitePrecedentIndex(corpus, taxonomy).candidates(["gc35_48h"])}
    gc35 = facets[precedent_id]["gc35_48h"]
    assert (gc35.finding_kind, gc35.finding) == ("not_examined", None)  # the kind valid for every document, never shown
    fixture = next(f for p in PRECEDENTS if p.id == precedent_id for f in p.facets if f.facet_id == "gc35_48h")
    assert tuple(q.span.text for q in gc35.facts) == fixture.facts
    assert facets["tw-synthetic-a"]["iccpr_14_3_b"].finding is not None  # a finding that fits is kept


def test_the_kinds_of_finding_each_document_states_are_one_table_in_the_schema():
    assert set(ALLOWED_FINDINGS) == set(get_args(PrecedentKind))
    assert all(kinds <= set(get_args(FindingKind)) and UNSTATED in kinds for kinds in ALLOWED_FINDINGS.values())
    assert precedent_store.ALLOWED_FINDINGS is ALLOWED_FINDINGS  # the store reads the schema's table
    assert not hasattr(precedent_store, "FINDING_KINDS")  # and keeps no copy of its own


def test_the_builder_checks_findings_against_the_same_table():
    from corpus_builder import verify

    built = getattr(verify, "FINDING_KINDS", None)  # the builder's own copy, until it imports the schema's
    assert dict(built if built is not None else verify.ALLOWED_FINDINGS) == dict(ALLOWED_FINDINGS)


def test_a_quote_outside_its_text_is_skipped(corpus, taxonomy):
    with closing(sqlite3.connect(corpus)) as db, db:
        db.execute("DELETE FROM quotes WHERE precedent_id = 'wgad-synthetic-c' AND facet_id = 'renewal_review' AND role = 'fact'")
        vec = np.zeros(384, dtype=np.float32).tobytes()
        db.execute("INSERT INTO quotes VALUES ('wgad-synthetic-c', 'renewal_review', 'fact', 5000, 5100, 'exact', ?)", (vec,))
    index = SqlitePrecedentIndex(corpus, taxonomy)
    [(_, facets)] = [entry for entry in index.candidates(["renewal_review", "gc35_48h"]) if entry[0].id == "wgad-synthetic-c"]
    assert [f.facet_id for f in facets] == ["gc35_48h", "media_defendant"]  # the facet lost its only fact


# --- the backend choice ---------------------------------------------------------------------


def test_no_corpus_gives_no_index_and_the_command(tmp_path, monkeypatch, taxonomy):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(tmp_path / "missing.db"))
    index, status = open_index(PrecedentSettings(backend="sqlite"), taxonomy)
    assert index is None
    assert "python -m corpus_builder" in status


def test_the_sqlite_backend_reads_the_file(corpus, monkeypatch, taxonomy):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(corpus))
    index, status = open_index(PrecedentSettings(backend="sqlite"), taxonomy)
    assert isinstance(index, SqlitePrecedentIndex)
    assert status == "Corpus file"


def test_opensearch_not_running_falls_back_to_the_file(corpus, monkeypatch, taxonomy):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(corpus))
    monkeypatch.setattr(precedent_opensearch, "available", lambda url, timeout=1.0: False)
    index, status = open_index(PrecedentSettings(backend="opensearch"), taxonomy)
    assert isinstance(index, SqlitePrecedentIndex)
    assert status == "Corpus file (OpenSearch not running)"
