"""Corpus builder, embed and build-db: one vector per verified fact (keyed by the quote span), and a
fresh precedents.db written through SQLITE_DDL with the meta the runtime checks."""

import datetime as dt
import sqlite3

import numpy as np
import pytest

from corpus_builder.build_db import BuildError, build_db, extraction_models
from corpus_builder.embed import CONTEXT_CHARS, context_bounds, embed_all
from corpus_builder.extract import PROMPT_SHA, extract_all
from corpus_builder.store import BuildStore
from corpus_builder.verify import StaleVerification, verify_all
from fixtures.corpus_b.synthetic import REPORT, VIEWS, FakeEmbedder, FakeLLM, FakeSource, facet, fixture_text, make_doc
from ratio.config import default_config
from ratio.embeddings import EMBEDDING_DIM
from ratio.precedent_schema import EMBED_MODEL_TAG, META_KEYS, SCHEMA_VERSION, PrecedentDoc
from ratio.provenance import span_is_valid

TAXONOMY = default_config().fact_patterns

# verify keeps a quote only with every negation of its sentence ("without"), so it quotes the whole clause
EXACT = "She was held in police custody for five days before she was first brought before a judge, who ordered her pretrial detention without hearing her lawyer"
PARAPHRASE = "The author was held for five days without ever seeing a judge"
CURLY = 'the court refused to hear the two defence witnesses named by the author, stating that their evidence was "not relevant to the charge"'
FINDING = "The Committee finds that the refusal to hear the defence witnesses violated article 14(3)(e) of the Covenant"
MEDIA = "The author is a journalist and the editor of an independent news portal in the capital"
KEPT_4_DAYS = "The monitor observed that the defendant was kept in a police station for four days before any judge saw him"
ASSESSMENT = "In the monitor's assessment, the delay before the first appearance breached the requirement of prompt judicial control"
BLOGGER = "Dario Melk, a blogger who wrote about local elections, was charged with spreading false news through his website"
B_LINE = "An extra invented line for the second synthetic source."

SOURCES = (
    FakeSource("views", "ccpr_views", "Synthetic Committee Views", "Synthetic source, for tests only"),
    FakeSource("reports_a", "trialwatch_report", "Synthetic reports A", "Synthetic reports A", terms_checked="2099-03-01"),
    FakeSource(
        "reports_b", "trialwatch_report", "Synthetic reports B", "Synthetic reports B",
        terms_url="https://b.example.invalid/terms", terms_checked="2099-02-01", allowed_prefixes=("https://b.example.invalid/",),
    ),
    FakeSource("wgad", "wgad_opinion", "Synthetic opinions", "Synthetic opinions", allowed_prefixes=("https://w.example.invalid/",)),
)  # fmt: skip


def answer(user):
    if B_LINE in user:
        return {"facets": [facet("media_defendant", [BLOGGER])]}
    if "Dario Melk" in user:
        return {"facets": [facet("gc35_48h", [KEPT_4_DAYS], "monitor_assessment", ASSESSMENT)]}
    if "Orla Venk" in user:
        return {
            "facets": [
                facet("gc35_48h", [EXACT, PARAPHRASE]),
                facet("iccpr_14_3_e", [CURLY], finding=FINDING),
                facet("media_defendant", [MEDIA], "not_examined"),
            ]
        }
    return {"facets": [facet("gc35_48h", [PARAPHRASE])]}


def documents():
    report = fixture_text(REPORT)
    return (
        make_doc("ccpr-9999-2099", fixture_text(VIEWS)),
        make_doc("tw-synthetic-melk", report, kind="trialwatch_report", attribution="Synthetic reports A"),
        make_doc(
            "tw-synthetic-b", f"{report}\n{B_LINE}\n", kind="trialwatch_report",
            url="https://b.example.invalid/melk", attribution="Synthetic reports B",
        ),
        make_doc("tw-synthetic-empty", "SYNTHETIC: an invented document.\n\nNothing happened here, by design.", kind="trialwatch_report"),
    )  # fmt: skip


@pytest.fixture
def store(tmp_path):
    build = BuildStore(tmp_path / "build.db")
    for doc in documents():
        build.put_document(doc, doc.url)
    extract_all(build, TAXONOMY, FakeLLM(answer=answer))
    verify_all(build, TAXONOMY)
    return build


@pytest.fixture
def embedded(store):
    embed_all(store, FakeEmbedder())
    return store


def build(store, out, taxonomy=TAXONOMY, sources=SOURCES):
    return build_db(store, taxonomy, sources, out, extraction_model="fake-model", prompt_sha=PROMPT_SHA)


def fact_keys(store):
    return {
        (pid, item.facet_id, quote.span.start, quote.span.end)
        for pid, (facets, _) in store.verified().items()
        for item in facets
        for quote in item.facts
    }


def rows(path, query):
    with sqlite3.connect(path) as db:
        return db.execute(query).fetchall()


# --- embed -----------------------------------------------------------------------------------


def test_embed_stores_one_unit_vector_per_fact_keyed_by_the_quote_span(store):
    count = embed_all(store, FakeEmbedder())
    vectors = store.vectors()
    assert count == len(fact_keys(store)) == 5
    assert set(vectors) == fact_keys(store)
    for vec in vectors.values():
        assert vec.dtype == np.float32 and vec.shape == (EMBEDDING_DIM,)
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


def test_embed_reads_the_sentence_around_each_quote(store):
    embedder = FakeEmbedder()
    embed_all(store, embedder)
    views = fixture_text(VIEWS)
    (around,) = [text for text in embedder.texts if EXACT in text]
    assert around != EXACT and around in views
    assert around.rstrip().endswith("without hearing her lawyer.")


def test_context_is_capped_and_centred_on_the_quote():
    text = "SYNTHETIC: one very long sentence.\n\n" + "word " * 400 + "the quoted passage here " + "word " * 400
    start = text.index("the quoted passage")
    end = start + len("the quoted passage here")
    low, high = context_bounds(text, start, end)
    assert high - low == CONTEXT_CHARS
    assert low <= start and end <= high
    assert abs((low + high) / 2 - (start + end) / 2) <= 1


def test_embed_refuses_vectors_of_the_wrong_shape(store):
    class Wrong:
        def encode(self, texts):
            return np.ones((len(texts), 8), dtype=np.float32)

    with pytest.raises(ValueError, match="384"):
        embed_all(store, Wrong())
    assert store.vectors() == {}


# --- build-db --------------------------------------------------------------------------------


def test_build_db_round_trips_through_the_ddl(embedded, tmp_path):
    out = tmp_path / "corpus" / "precedents.db"
    build(embedded, out)
    names = {row[0] for row in rows(out, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert names == {"meta", "documents", "facets", "quotes", "sources"}
    assert rows(out, "PRAGMA foreign_key_check") == []

    columns = [row[1] for row in rows(out, "PRAGMA table_info(documents)")]
    stored = {r[0]: PrecedentDoc.model_validate(dict(zip(columns, r, strict=True))) for r in rows(out, "SELECT * FROM documents")}
    assert sorted(stored) == ["ccpr-9999-2099", "tw-synthetic-b", "tw-synthetic-melk"]  # the empty one has no facet
    assert all(stored[pid] == embedded.document(pid) for pid in stored)

    verified = embedded.verified()
    facets = set(rows(out, "SELECT precedent_id, facet_id, finding_kind FROM facets"))
    assert facets == {(pid, f.facet_id, f.finding_kind) for pid in stored for f in verified[pid][0]}

    vectors = embedded.vectors()
    quotes = rows(out, "SELECT precedent_id, facet_id, role, start, end, match, vec FROM quotes")
    for pid, facet_id, role, start, end, _match, vec in quotes:
        text = stored[pid].text[start:end]
        assert text.strip()
        if role == "fact":
            assert np.array_equal(np.frombuffer(vec, dtype=np.float32), vectors[(pid, facet_id, start, end)])
        else:
            assert vec is None
    assert {(q[0], q[1], q[3], q[4]) for q in quotes if q[2] == "fact"} == fact_keys(embedded)
    findings = {(q[0], q[1], q[3], q[4], q[5]) for q in quotes if q[2] == "finding"}
    expected = {
        (pid, f.facet_id, f.finding.span.start, f.finding.span.end, f.finding.match)
        for pid in stored
        for f in verified[pid][0]
        if f.finding
    }
    assert findings == expected and len(findings) == 2


def test_every_stored_span_is_valid_against_the_stored_text(embedded, tmp_path):
    out = tmp_path / "precedents.db"
    build(embedded, out)
    texts = dict(rows(out, "SELECT id, text FROM documents"))
    resolve = lambda doc_id: texts.get(doc_id.removeprefix("precedent-").removesuffix("/text"))  # noqa: E731
    for pid, (facets, _) in embedded.verified().items():
        for item in facets:
            for quote in (*item.facts, *([item.finding] if item.finding else [])):
                assert pid not in texts or span_is_valid(quote.span, resolve)


def test_meta_holds_what_the_runtime_checks(embedded, tmp_path):
    out = tmp_path / "precedents.db"
    meta = build(embedded, out)
    stored = dict(rows(out, "SELECT key, value FROM meta"))
    assert set(stored) == set(META_KEYS)
    assert stored["schema_version"] == str(SCHEMA_VERSION)
    assert stored["taxonomy_sha"] == TAXONOMY.sha
    assert stored["embed_model"] == EMBED_MODEL_TAG
    assert stored["extraction_model"] == "fake-model"
    assert stored["prompt_sha"] == PROMPT_SHA
    assert stored["dropped_quotes"] == "2"  # the paraphrase in the views and in the empty document
    built_at = dt.datetime.fromisoformat(stored["built_at"])
    assert built_at.utcoffset() == dt.timedelta(0)
    assert stored["built_at"] == meta.built_at


def test_build_db_returns_the_corpus_meta_with_sources_merged_by_kind(embedded, tmp_path):
    meta = build(embedded, tmp_path / "precedents.db")
    assert (meta.documents, meta.facets, meta.dropped_quotes) == (3, 5, 2)
    views, reports = meta.sources
    assert (views.kind, views.name, views.documents) == ("ccpr_views", "Synthetic Committee Views", 1)
    assert (reports.kind, reports.documents) == ("trialwatch_report", 2)
    assert reports.name == "Synthetic reports A; Synthetic reports B"
    assert reports.terms_url == "https://example.invalid/terms; https://b.example.invalid/terms"
    assert reports.terms_checked == "2099-02-01"  # the oldest check
    stored = rows(tmp_path / "precedents.db", "SELECT kind, name, terms_url, terms_checked, attribution FROM sources ORDER BY kind")
    assert stored == [(s.kind, s.name, s.terms_url, s.terms_checked, s.attribution) for s in meta.sources]


def test_a_fact_without_a_vector_is_an_error_and_leaves_the_old_corpus(store, tmp_path):
    out = tmp_path / "out" / "precedents.db"
    out.parent.mkdir()
    out.write_bytes(b"the previous corpus")
    with pytest.raises(BuildError, match="embed"):
        build(store, out)
    assert out.read_bytes() == b"the previous corpus"
    assert [p.name for p in out.parent.iterdir()] == ["precedents.db"]


def test_a_store_with_no_verified_facet_never_replaces_the_corpus(tmp_path):
    out = tmp_path / "precedents.db"
    with pytest.raises(BuildError, match="extract"):
        build(BuildStore(tmp_path / "empty-build.db"), out)
    assert not out.exists()


def test_an_existing_corpus_is_replaced_whole(embedded, tmp_path):
    out = tmp_path / "precedents.db"
    out.write_bytes(b"not a database")
    build(embedded, out)
    assert len(rows(out, "SELECT id FROM documents")) == 3
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_the_default_out_path_follows_ratio_corpus_db(embedded, tmp_path, monkeypatch):
    target = tmp_path / "from-env.db"
    monkeypatch.setenv("RATIO_CORPUS_DB", str(target))
    build_db(embedded, TAXONOMY, SOURCES, extraction_model="fake-model", prompt_sha=PROMPT_SHA)
    assert target.exists()


def test_a_span_that_no_longer_matches_its_text_is_refused(embedded, tmp_path):
    changed = make_doc("ccpr-9999-2099", "SYNTHETIC: edited.\n" + fixture_text(VIEWS))
    embedded.put_document(changed, changed.url)
    with pytest.raises(StaleVerification, match="verify"):
        build(embedded, tmp_path / "precedents.db")


def test_a_document_from_no_known_source_is_an_error(embedded, tmp_path):
    with pytest.raises(BuildError, match="ccpr-9999-2099"):
        build(embedded, tmp_path / "precedents.db", sources=SOURCES[1:])


def test_a_facet_outside_the_taxonomy_is_an_error(embedded, tmp_path):
    smaller = TAXONOMY.model_copy(update={"facets": tuple(f for f in TAXONOMY.facets if f.id != "media_defendant")})
    with pytest.raises(BuildError, match="media_defendant"):
        build(embedded, tmp_path / "precedents.db", taxonomy=smaller)


def test_extraction_models_names_the_models_behind_the_verified_facets(embedded):
    assert extraction_models(embedded) == "fake-model"
