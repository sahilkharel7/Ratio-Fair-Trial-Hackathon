"""Search over every paragraph of the library, Similar wording on a case, and the Precedent library page,
on the SYNTHETIC fixture corpus (tests/precedent_fixtures.py), whose paragraphs are its search passages.

Search finds a paragraph by its meaning and the words it shares with the query; a paragraph of a
document edited after it was checked is never shown; OpenSearch and the corpus file give the same
answer; and the page searches, opens each paragraph's source, and adds uploaded documents to a
collection without sending them anywhere."""

import re
import uuid

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import corpus_builder.collections as col
from corpus_builder import opensearch as builder
from ratio import precedent_opensearch
from ratio.config import PrecedentSettings, load_config
from ratio.embeddings import EmbeddingModelMissing
from ratio.paths import REPO_ROOT
from ratio.precedent_opensearch import OpenSearchPrecedentIndex, available
from ratio.precedent_schema import PrecedentQuote
from ratio.precedent_store import SqlitePrecedentIndex, rank_passages
from ratio.precedents import similar_wording
from ratio.schema import SourceSpan
from ratio.testing import make_record
from tests.precedent_fixtures import PRECEDENTS, TAMPERED, FakeEmbedder, build_fixture_corpus

CONFIG = load_config()
TAXONOMY = CONFIG.fact_patterns
EMBEDDER = FakeEmbedder()
URL = "http://127.0.0.1:9200"
APP_DIR = REPO_ROOT / "app"
LIBRARY = "views/library.py"
LAWYER = "Her lawyer was not allowed to attend the remand hearing and received the case file only the day before trial."
EDITED = "Her lawyer was not present at the remand hearing."  # a paragraph of the tampered precedent
UNRELATED = "The committee planted four hundred oak trees along the river bank during the spring festival week."


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return build_fixture_corpus(tmp_path_factory.mktemp("library") / "precedents.db", EMBEDDER, TAXONOMY)


@pytest.fixture(scope="module")
def index(corpus):
    return SqlitePrecedentIndex(corpus, TAXONOMY)


def search(index, text: str, **options):
    return index.search_passages(EMBEDDER.encode([text])[0], words=text, **options)


def found(hits) -> list[tuple[str, int, int]]:
    return [(hit.precedent_id, hit.quote.span.start, hit.quote.span.end) for hit in hits]


# --- search ------------------------------------------------------------------------------------


def test_every_paragraph_of_the_fixture_is_searchable_except_those_of_the_edited_document(index):
    assert index.passage_count() == sum(len(p.paragraphs) for p in PRECEDENTS if p.id != TAMPERED)


def test_search_finds_the_paragraph_closest_to_the_query_first(index):
    hits = search(index, LAWYER)
    assert hits[0].precedent_id == "tw-synthetic-a" and hits[0].quote.span.text == LAWYER
    assert hits[0].quote.span.doc_id == "precedent-tw-synthetic-a/text" and hits[0].quote.match == "exact"
    assert hits[0].cosine == pytest.approx(1.0) and hits[0].words == 1.0
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_search_returns_at_most_k_paragraphs(index):
    assert len(search(index, LAWYER, k=2)) == 2


def test_search_can_be_limited_to_some_documents(index):
    hits = search(index, LAWYER, precedent_ids=["ccpr-synthetic-b"])
    assert hits and {hit.precedent_id for hit in hits} == {"ccpr-synthetic-b"}
    assert search(index, LAWYER, precedent_ids=[]) == ()


def test_a_paragraph_of_a_document_edited_after_it_was_checked_is_never_shown(index):
    assert TAMPERED not in {hit.precedent_id for hit in search(index, EDITED, k=50)}


def test_an_empty_query_vector_finds_nothing(index):
    assert index.search_passages(np.zeros(384, dtype=np.float32)) == ()


def test_the_query_words_decide_between_equally_close_paragraphs():
    texts = {("a", 0, 31): "the court refused bail outright", ("b", 0, 31): "witnesses were heard in camera."}

    def quote(pid, start, end):
        return PrecedentQuote(span=SourceSpan(doc_id=f"precedent-{pid}/text", start=start, end=end, text=texts[(pid, start, end)]), match="exact")

    ranked = rank_passages([(("a", 0, 31), 0.5), (("b", 0, 31), 0.5)], quote, "witnesses heard", 2)
    assert [hit.precedent_id for hit in ranked] == ["b", "a"]
    assert ranked[0].score == pytest.approx(0.75 * 0.5 + 0.25 * 1.0)
    without_words = rank_passages([(("a", 0, 31), 0.5), (("b", 0, 31), 0.5)], quote, "", 2)
    assert [hit.score for hit in without_words] == [0.5, 0.5]  # a tie: the first precedent id first


# --- Similar wording -------------------------------------------------------------------------


def case_with(*paragraphs: str):
    return make_record([("notes/hearing_1.txt", "monitoring_note", "\n\n".join(paragraphs))])


def test_similar_wording_pairs_a_case_paragraph_with_the_closest_precedent_paragraph(index):
    matches = similar_wording(case_with(LAWYER, UNRELATED), index, EMBEDDER)
    assert matches[0].precedent.id == "tw-synthetic-a"
    assert matches[0].case_passage.text == LAWYER and matches[0].precedent_passage.span.text == LAWYER
    assert matches[0].cosine == pytest.approx(1.0) and matches[0].support >= 1
    assert all(match.cosine >= 0.6 for match in matches)


def test_similar_wording_finds_nothing_when_no_paragraph_is_close(index):
    assert similar_wording(case_with(UNRELATED), index, EMBEDDER) == ()


def test_similar_wording_never_shows_an_edited_document(index):
    padded = EDITED + " The family said so in a statement to the monitors after the hearing."
    assert TAMPERED not in {match.precedent.id for match in similar_wording(case_with(padded), index, EMBEDDER)}


# --- OpenSearch: the same answer as the corpus file ------------------------------------------


class FailingClient:
    def search(self, index, body, **kwargs):
        raise precedent_opensearch.request_errors()[0]("the index broke")


def test_a_failed_opensearch_search_is_answered_from_the_corpus_file(index):
    search_index = OpenSearchPrecedentIndex(URL, "unused", index, client=FailingClient())
    assert found(search(search_index, LAWYER)) == found(search(index, LAWYER))


running = pytest.mark.skipif(not available(URL), reason="OpenSearch is not running on 127.0.0.1:9200 (docker compose up -d opensearch)")


@pytest.fixture
def alias():
    name = f"ratio-test-{uuid.uuid4().hex[:12]}"
    yield name
    precedent_opensearch.connect(URL).indices.delete(index=f"{name}-v*", expand_wildcards="all", ignore_unavailable=True)


@pytest.mark.opensearch
@running
def test_both_backends_return_the_same_paragraphs(corpus, index, alias):
    builder.index_corpus(corpus, URL, alias, taxonomy=TAXONOMY)
    search_index = OpenSearchPrecedentIndex(URL, alias, index)
    search_index.check()
    for query in (LAWYER, "journalist charged with false information", EDITED):
        assert found(search(search_index, query)) == found(search(index, query))
    only = ["wgad-synthetic-c"]
    assert found(search(search_index, LAWYER, precedent_ids=only)) == found(search(index, LAWYER, precedent_ids=only))


# --- the Precedent library page --------------------------------------------------------------


@pytest.fixture
def library(tmp_path, corpus, monkeypatch):
    """The page on the fixture corpus, with its own collections folder and work database."""
    monkeypatch.setenv("RATIO_DB", str(tmp_path / "ratio.db"))
    monkeypatch.setenv("RATIO_CORPUS_DB", str(corpus))
    monkeypatch.setenv("RATIO_COLLECTIONS_DIR", str(tmp_path / "collections"))
    monkeypatch.setenv("RATIO_BUILD_DB", str(tmp_path / "build.db"))
    monkeypatch.syspath_prepend(str(APP_DIR))
    from ratio_ui import session

    monkeypatch.setattr(session, "precedent_settings", lambda: PrecedentSettings(backend="sqlite"))
    monkeypatch.setattr(session, "embedder", lambda: EMBEDDER)
    session.clear_precedent_index()
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=60)
    at.switch_page(LIBRARY).run()
    assert not at.exception
    yield at
    session.clear_precedent_index()


def texts(at: AppTest) -> list[str]:
    """Each text element as the reader sees it: no backslashes before punctuation."""
    elements = (*at.markdown, *at.caption, *at.info, *at.error, *at.success, *at.title, *at.header)
    return [re.sub(r"\\(.)", r"\1", element.value) for element in elements]


def add_button(at: AppTest):
    return next(button for button in at.button if button.label == "Add to the library")


def test_the_page_says_what_it_reads_from(library):
    assert [title.value for title in library.title] == ["Precedent library"]
    assert any("Corpus file" in text and f"{len(PRECEDENTS)} documents" in text for text in texts(library))


def test_a_search_shows_the_closest_paragraph_with_its_source(library):
    library.text_input(key="library_query").input(LAWYER).run()
    assert not library.exception
    assert any(LAWYER in element.proto.body for element in library.get("html"))
    assert any("Fairness report: the trial of Ilsa Moravec (synthetic)" in text for text in texts(library))
    library.button(key="library-source-0").click().run()
    assert not library.exception and not library.error
    assert any("checked against the source: exact match" in text for text in texts(library))


def test_a_search_can_be_limited_to_one_source(library):
    library.text_input(key="library_query").input(LAWYER).run()
    sources = library.multiselect(key="library_sources").options
    other = next(source for source in sources if "TrialWatch" not in source)
    library.multiselect(key="library_sources").select(other).run()
    titles = [text for text in texts(library) if text.startswith("**")]
    assert titles and not any("Moravec" in title for title in titles)


def test_without_a_corpus_the_page_says_how_to_add_documents(library, tmp_path, monkeypatch):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(tmp_path / "missing.db"))
    library.run()
    assert CONFIG.messages.notes["library_empty"] in texts(library)
    assert library.file_uploader  # documents can still be added


def test_uploaded_documents_go_into_a_private_collection_on_this_computer(library):
    library.text_input[1].input("Indonesia")
    library.file_uploader[0].set_value([("arden.txt", b"SYNTHETIC: a test document about a hearing.", "text/plain")])
    add_button(library).click().run()
    assert not library.exception and not library.error
    (found_collection,) = col.list_collections()
    assert (found_collection.name, found_collection.private) == ("Indonesia", True)
    assert [path.name for path in col.files_in(found_collection)] == ["arden.txt"]
    assert any("Added 1 documents to Indonesia" in text for text in texts(library))
    assert library.dataframe and library.button(key="library_build")


def test_a_file_of_another_type_is_refused_with_the_reason(library):
    library.text_input[1].input("Indonesia")
    library.file_uploader[0].set_value([("fake.pdf", b"not a pdf", "application/pdf")])
    add_button(library).click().run()
    assert any("not a PDF file" in error.value for error in library.error)


def test_adding_without_a_file_asks_for_one(library):
    library.text_input[1].input("Indonesia")
    add_button(library).click().run()
    assert any("Choose at least one document" in error.value for error in library.error)
    assert col.list_collections() == ()


def test_reading_the_collections_uses_only_the_local_model(library, monkeypatch):
    col.create_collection("Indonesia")
    calls = []

    def fake_build(store, taxonomy, llm, embedder, *, collections, sources, index, progress):
        calls.append((llm, [c.slug for c in collections]))
        progress("Reading the uploaded documents")
        return col.BuildReport(documents=("col-indonesia-arden",), skipped=(), extracted=1, failed=(), passages=3, meta=None, indexed=None)

    monkeypatch.setattr(col, "build", fake_build)
    monkeypatch.setattr(col, "local_model", lambda: "the local model")
    monkeypatch.setattr(col, "crawled_sources", lambda: ())
    library.run()
    library.button(key="library_build").click().run()
    assert not library.exception and not library.error
    assert calls == [("the local model", ["indonesia"])]


# --- regressions -------------------------------------------------------------------------------


class AllParagraphsOf:
    """The fixture index, but every search answers with all paragraphs of one precedent."""

    def __init__(self, index, pid: str):
        self._index, self.pid = index, pid

    def text(self, doc_id):
        return self._index.text(doc_id)

    def documents(self):
        return self._index.documents()

    def search_passages(self, query, *, words="", k=10, precedent_ids=None):
        hits = [hit for hit in search(self._index, LAWYER, k=50) if hit.precedent_id == self.pid]
        return [hit.model_copy(update={"cosine": 0.9}) for hit in hits]


def test_similar_wording_counts_each_case_paragraph_once_per_precedent(index):
    fake = AllParagraphsOf(index, "tw-synthetic-a")
    assert len(fake.search_passages(None)) > 1  # one case paragraph finds several paragraphs of the precedent
    (match,) = similar_wording(case_with(LAWYER), fake, EMBEDDER)
    assert match.support == 1


class CountingFailingClient(FailingClient):
    def __init__(self):
        self.calls = 0

    def search(self, index, body, **kwargs):
        self.calls += 1
        return super().search(index, body, **kwargs)


def test_similar_wording_tries_a_failing_opensearch_once_then_reads_the_corpus_file(index):
    client = CountingFailingClient()
    search_index = OpenSearchPrecedentIndex(URL, "unused", index, client=client)
    record = case_with(LAWYER, UNRELATED, "She was charged with spreading false information after publishing reports on a flood relief fund.")
    assert similar_wording(record, search_index, EMBEDDER) == similar_wording(record, index, EMBEDDER)
    assert client.calls == 1


def test_a_search_without_the_embedding_model_says_how_to_get_it(library, monkeypatch):
    from ratio_ui import session

    def missing():
        raise EmbeddingModelMissing("Embedding model not found. Run `python scripts/fetch_models.py` once while online.")

    monkeypatch.setattr(session, "embedder", missing)
    library.text_input(key="library_query").input(LAWYER).run()
    assert not library.exception
    assert library.error and any("fetch_models.py" in text for text in texts(library))


def test_an_unusable_corpus_says_why(library, tmp_path, monkeypatch):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(tmp_path / "missing.db"))
    library.run()
    captions = [text for text in texts(library) if "corpus_builder" in text or "corpus" in text.lower()]
    assert captions, texts(library)
