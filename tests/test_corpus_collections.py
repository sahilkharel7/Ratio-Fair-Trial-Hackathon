"""Collections (corpus_builder/collections.py): documents a user uploads, read into the precedent corpus
with no crawling. Every collection and document below is SYNTHETIC; each test works in its own
temporary folder and work database. The local model is OllamaLLM with a fake transport, so no test
opens a socket."""

import hashlib
import json
import shutil
import sqlite3
from contextlib import closing

import pytest

import corpus_builder.__main__ as cli
import corpus_builder.collections as col
from corpus_builder import extract
from corpus_builder.embed import embed_passages, note_share, passage_spans
from corpus_builder.extract import CloudModelRefused, OllamaLLM, extract_all
from corpus_builder.normalize import normalize_all
from corpus_builder.store import BuildStore
from fixtures.corpus_b.synthetic import FakeEmbedder, FakeLLM, facet
from ratio.config import default_config
from ratio.precedent_schema import COLLECTION_PREFIX
from ratio.precedent_store import SqlitePrecedentIndex

TAXONOMY = default_config().fact_patterns
FACT = "Mr. Arden was held in police custody for six days before he was first brought before a judge."
INVENTED = "The court ordered his immediate release and apologised for the delay."  # in no document
DOCUMENT = f"""SYNTHETIC: test document; every name, place and court in it is invented.

Monitoring note on the trial of Mr. Arden before the Velmora District Court.

{FACT} His lawyer was not allowed to visit him during those six days, according to the family.

The prosecution relied on a single statement that the defence was not allowed to question in court.
The judge refused to call the two witnesses the defence named, saying that their evidence would not
change the outcome, and extended the detention for a further two months without giving reasons.
"""


def tag(slug: str, name: str, data: bytes = DOCUMENT.encode()) -> str:
    """The 8 hex characters an uploaded file's id ends with: the sha256 of its place and contents."""
    return hashlib.sha256(f"{slug}/{name}\n".encode() + data).hexdigest()[:8]


ARDEN = f"col-indonesia-arden-{tag('indonesia', 'arden.txt')}"  # col-<collection>-<file name>-<tag>


@pytest.fixture
def folder(tmp_path, monkeypatch):
    root = tmp_path / "collections"
    monkeypatch.setenv("RATIO_COLLECTIONS_DIR", str(root))
    return root


@pytest.fixture
def store(tmp_path):
    return BuildStore(tmp_path / "build.db")


class FakePost:
    """Ollama on 127.0.0.1 with a scripted answer (and a local model's description); records every request."""

    def __init__(self, answer: dict):
        self.answer, self.requests = answer, []

    def __call__(self, url, body, timeout):
        self.requests.append((url, body))
        return {"message": {"content": json.dumps(self.answer)}, "done_reason": "stop"}


def local_model(answer: dict) -> tuple[OllamaLLM, FakePost]:
    post = FakePost(answer)
    return OllamaLLM(post=post), post


def answer(*facts: str) -> dict:
    return {"facets": [facet("gc35_48h", list(facts), finding_kind="not_examined")]}


# --- creating a collection --------------------------------------------------------------------


def test_a_new_collection_is_private_by_default_and_kept_in_its_folder(folder):
    found = col.create_collection("Indonesia (East)", region="South-East Asia")
    assert (found.slug, found.name, found.region, found.private) == ("indonesia-east", "Indonesia (East)", "South-East Asia", True)
    assert (folder / "indonesia-east" / "collection.yaml").is_file() and (folder / "indonesia-east" / "files").is_dir()
    assert col.list_collections() == (found,)
    assert col.load_collection("indonesia-east") == found


def test_creating_an_existing_collection_returns_it(folder):
    first = col.create_collection("Indonesia", private=False)
    assert col.create_collection("indonesia", private=False) == first
    assert len(col.list_collections()) == 1


def test_a_collection_never_changes_from_private_to_public(folder):
    col.create_collection("Indonesia")
    with pytest.raises(col.CollectionError, match="already exists as a private collection"):
        col.create_collection("Indonesia", private=False)
    assert col.load_collection("indonesia").private is True


@pytest.mark.parametrize("name", ["", "   ", "!!!"])
def test_a_collection_needs_a_name_with_letters_or_digits(folder, name):
    with pytest.raises(col.CollectionError, match="needs a name"):
        col.create_collection(name)


def test_no_collections_folder_means_no_collections(folder):
    assert col.list_collections() == ()


# --- adding files -----------------------------------------------------------------------------


def test_an_added_file_is_saved_and_recorded_for_the_build(folder, store):
    found = col.create_collection("Indonesia")
    item = col.add_file(found, "Arden v State_2024.txt", DOCUMENT.encode(), store)
    assert item.url == f"{COLLECTION_PREFIX}indonesia/arden-v-state-2024.txt"
    assert (item.source_id, item.title, item.symbol) == ("collection-indonesia", "Arden v State 2024", f"indonesia-arden-v-state-2024-{tag('indonesia', 'arden-v-state-2024.txt')}")
    assert item.path.read_text(encoding="utf-8") == DOCUMENT
    assert col.files_in(found) == (item.path,)
    assert [raw.url for raw in store.raw_items()] == [item.url]


@pytest.mark.parametrize(
    ("filename", "data", "reason"),
    [
        ("notes.docx", b"PK\x03\x04", "only PDF, text"),
        ("run.sh", b"#!/bin/sh", "only PDF, text"),
        ("fake.pdf", b"<html>not a pdf</html>", "not a PDF file"),
    ],
)
def test_a_file_of_another_type_is_refused(folder, store, filename, data, reason):
    found = col.create_collection("Indonesia")
    with pytest.raises(col.CollectionError, match=reason):
        col.add_file(found, filename, data, store)
    assert col.files_in(found) == ()


def test_a_file_over_the_size_limit_is_refused(folder, store, monkeypatch):
    monkeypatch.setattr(col, "MAX_BYTES", 10)
    with pytest.raises(col.CollectionError, match="larger than"):
        col.add_file(col.create_collection("Indonesia"), "big.txt", b"x" * 11, store)


def test_two_different_files_with_the_same_name_are_both_kept(folder, store):
    found = col.create_collection("Indonesia")
    first = col.add_file(found, "note.txt", DOCUMENT.encode(), store)
    second = col.add_file(found, "note.txt", (DOCUMENT + "\nA later hearing.\n").encode(), store)
    again = col.add_file(found, "note.txt", DOCUMENT.encode(), store)
    assert first.path != second.path and again.path == first.path
    assert len(col.files_in(found)) == 2


# --- building ---------------------------------------------------------------------------------


def build(store, tmp_path, llm, *collections, index=None):
    return col.build(store, TAXONOMY, llm, FakeEmbedder(), collections=collections, sources=(),
                     out_path=tmp_path / "precedents.db", index=index)  # fmt: skip


def test_a_private_collection_is_never_sent_to_a_cloud_model(folder, store, tmp_path):
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    cloud = FakeLLM(answer=answer(FACT))
    with pytest.raises(col.CollectionError, match="read only by the local model"):
        build(store, tmp_path, cloud, found)
    assert cloud.calls == [] and not (tmp_path / "precedents.db").exists()


def test_a_public_collection_may_be_read_by_another_model(folder, store, tmp_path):
    found = col.create_collection("Public reports", private=False)
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    report = build(store, tmp_path, FakeLLM(answer=answer(FACT)), found)
    assert report.documents == (f"col-public-reports-arden-{tag('public-reports', 'arden.txt')}",)


def test_a_private_collection_is_read_by_the_local_model_into_the_corpus(folder, store, tmp_path):
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    llm, post = local_model(answer(FACT, INVENTED))
    indexed = []
    report = build(store, tmp_path, llm, found, index=lambda path: indexed.append(path) or 7)
    assert report.documents == (ARDEN,) and report.extracted == 1
    assert report.skipped == () and report.failed == () and report.passages >= 1 and report.indexed == 7
    assert indexed == [tmp_path / "precedents.db"]
    assert [url for url, _ in post.requests] == ["http://127.0.0.1:11434/api/show", "http://127.0.0.1:11434/api/chat"]
    with closing(sqlite3.connect(tmp_path / "precedents.db")) as db:
        kind, private, url, text = db.execute("SELECT kind, private, url, text FROM documents WHERE id = ?", (ARDEN,)).fetchone()
        quotes = [text[start:end] for start, end in db.execute("SELECT start, end FROM quotes WHERE precedent_id = ?", (ARDEN,))]
        passages = db.execute("SELECT COUNT(*) FROM passages WHERE precedent_id = ?", (ARDEN,)).fetchone()[0]
    assert (kind, private, url) == ("collection_document", 1, f"{COLLECTION_PREFIX}indonesia/arden.txt")
    assert quotes == [FACT.rstrip(".")]  # the invented quote is not in the document, so it was dropped
    assert passages == report.passages


def test_a_building_reports_files_it_could_not_read(folder, store, tmp_path):
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    col.add_file(found, "broken.pdf", b"%PDF-1.4 not really a pdf", store)
    llm, _ = local_model(answer(FACT))
    report = build(store, tmp_path, llm, found)
    assert report.documents == (ARDEN,)
    assert len(report.skipped) == 1 and report.skipped[0].startswith(f"{COLLECTION_PREFIX}indonesia/broken.pdf: ")


def test_a_document_without_a_fact_pattern_is_still_found_by_its_wording(folder, store, tmp_path):
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    report = build(store, tmp_path, local_model({"facets": []})[0], found)
    assert report.documents == (ARDEN,) and report.passages >= 1
    index = SqlitePrecedentIndex(tmp_path / "precedents.db", TAXONOMY)
    assert [doc.id for doc in index.documents()] == [ARDEN]
    hits = index.search_passages(FakeEmbedder().encode(["witnesses the defence named"])[0], words="witnesses the defence named")
    assert hits and hits[0].precedent_id == ARDEN and "witnesses the defence named" in hits[0].quote.span.text


def test_a_document_the_local_model_could_not_read_is_reported_and_still_searchable(folder, store, tmp_path):
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    report = build(store, tmp_path, OllamaLLM(post=FakeDown()), found)
    assert len(report.failed) == 1 and "Ollama is not running" in report.failed[0]
    assert report.documents == (ARDEN,)


class FakeDown:
    def __call__(self, url, body, timeout):
        raise ConnectionRefusedError("Ollama is not running")


def test_building_again_reads_only_the_new_documents(folder, store, tmp_path):
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    llm, post = local_model(answer(FACT))
    build(store, tmp_path, llm, found)
    asked = len(post.requests)
    report = build(store, tmp_path, llm, found)
    assert len(post.requests) == asked and report.extracted == 0 and report.passages == 0
    assert report.documents == (ARDEN,)


# --- search passages --------------------------------------------------------------------------

BODY = (
    "The applicant was arrested at his home on 3 May and taken to the police station, where he was\n"
    "questioned for nine hours without a lawyer and was not told of the charges against him."
)
NOTES = (
    "58 Trial Monitor's Notes, October 27, 2020.\n59 Id.\n60 Id.\n61 Human Rights Committee, General Comment No. 32,\n"
    "U.N. Doc. CCPR/C/GC/32, August 23, 2007, para. 33.\n62 Available at https://example.invalid/report."
)


def test_footnote_and_citation_blocks_are_not_searched():
    assert note_share(NOTES) > 0.5 > note_share(BODY) == 0.0
    text = f"{BODY}\n\n{NOTES}"
    assert passage_spans(BODY) == [(0, len(BODY))]
    assert all(note_share(text[a:b]) <= 0.5 for a, b in passage_spans(text))
    assert passage_spans(NOTES) == []


def test_numbered_paragraphs_are_not_mistaken_for_footnotes():
    paragraph = "2.1 The author was arrested on 5 March 2019 and held for six days before\nbeing brought before a judge, without counsel."
    assert note_share(paragraph) == 0.0 and note_share("23. " + paragraph) == 0.0


def test_passages_are_embedded_again_only_when_their_cut_changes(folder, store, tmp_path, monkeypatch):
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    build(store, tmp_path, local_model(answer(FACT))[0], found)
    assert embed_passages(store, FakeEmbedder()) == 0
    monkeypatch.setattr("corpus_builder.embed.passage_spans", lambda text: [(0, 80)])
    assert embed_passages(store, FakeEmbedder()) == 1
    assert store.passage_cuts() == {ARDEN: [(0, 80)]}


# --- privacy: a private collection never reaches a cloud model ----------------------------------


def private_document_in(store, folder) -> str:
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    normalize_all(store, [found])
    return ARDEN


def test_extraction_never_gives_a_private_document_to_a_cloud_model(folder, store):
    pid = private_document_in(store, folder)
    cloud = FakeLLM(answer=answer(FACT))
    report = extract_all(store, TAXONOMY, cloud)
    assert cloud.calls == [] and report.private_refused == (pid,) and report.extracted == ()


def test_the_extract_step_refuses_a_named_private_document_for_gemini(folder, store, monkeypatch):
    pid = private_document_in(store, folder)
    cloud = FakeLLM(answer=answer(FACT))
    monkeypatch.setattr(extract, "GeminiLLM", lambda model_id: cloud)
    assert cli.main(["--work-db", str(store.path), "extract", "--model", "gemini", "--model-id", "gemini-x", "--only", pid]) == 2
    assert cli.main(["--work-db", str(store.path), "extract", "--model", "gemini", "--model-id", "gemini-x"]) == 0
    assert cloud.calls == []


def test_a_private_document_is_given_to_the_local_model(folder, store):
    pid = private_document_in(store, folder)
    llm, post = local_model(answer(FACT))
    report = extract_all(store, TAXONOMY, llm)
    assert report.extracted == (pid,) and report.private_refused == ()
    assert [url for url, _ in post.requests] == ["http://127.0.0.1:11434/api/show", "http://127.0.0.1:11434/api/chat"]


def test_an_ollama_cloud_model_is_refused_by_its_name():
    with pytest.raises(CloudModelRefused):
        OllamaLLM(model="gpt-oss:120b-cloud")


def test_a_model_the_server_serves_from_a_cloud_host_never_sees_a_private_document(folder, store):
    private_document_in(store, folder)
    post = FakePost(answer(FACT))
    remote = {"remote_host": "https://ollama.com:443", "remote_model": "gpt-oss:120b"}
    llm = OllamaLLM(model="alias", post=lambda url, body, timeout: remote if url.endswith("/api/show") else post(url, body, timeout))
    with pytest.raises(CloudModelRefused):
        extract_all(store, TAXONOMY, llm)
    assert post.requests == []  # no question was asked


# --- several collections ----------------------------------------------------------------------


def test_building_one_collection_keeps_the_others_in_the_corpus(folder, store, tmp_path):
    first, second = col.create_collection("Alpha"), col.create_collection("Beta")
    col.add_file(first, "arden.txt", DOCUMENT.encode(), store)
    col.add_file(second, "arden.txt", DOCUMENT.encode() + b"\nA second hearing was held a month later.\n", store)
    build(store, tmp_path, local_model(answer(FACT))[0], first, second)
    report = build(store, tmp_path, local_model(answer(FACT))[0], first)
    assert report.documents == (f"col-alpha-arden-{tag('alpha', 'arden.txt')}",)
    ids = [doc.id for doc in SqlitePrecedentIndex(tmp_path / "precedents.db", TAXONOMY).documents()]
    assert len(ids) == 2 and any(pid.startswith("col-beta-arden-") for pid in ids)


def test_a_deleted_collection_leaves_the_corpus_at_the_next_build(folder, store, tmp_path):
    first, second = col.create_collection("Alpha"), col.create_collection("Beta")
    col.add_file(first, "arden.txt", DOCUMENT.encode(), store)
    col.add_file(second, "arden.txt", DOCUMENT.encode() + b"\nA second hearing was held a month later.\n", store)
    build(store, tmp_path, local_model(answer(FACT))[0], first, second)
    shutil.rmtree(folder / "beta")
    build(store, tmp_path, local_model(answer(FACT))[0], first)
    assert [doc.id for doc in SqlitePrecedentIndex(tmp_path / "precedents.db", TAXONOMY).documents()] == [f"col-alpha-arden-{tag('alpha', 'arden.txt')}"]


def test_files_with_the_same_name_become_separate_documents(folder, store):
    found = col.create_collection("Indonesia")
    for name, data in (("judgment.txt", DOCUMENT), ("judgment.md", DOCUMENT + "\nA later hearing.\n"), ("judgment.html", f"<p>{DOCUMENT} Appeal.</p>")):
        col.add_file(found, name, data.encode(), store)
    report = normalize_all(store, [found])
    assert report.skipped == () and len(report.documents) == 3


def test_a_long_collection_and_file_name_still_give_distinct_ids(folder, store):
    found = col.create_collection("Constitutional Court of the Republic of Somewhere")
    name = "decision-of-the-constitutional-court-on-the-detention-of-journalists-" * 2
    first = col.add_file(found, f"{name}.txt", DOCUMENT.encode(), store)
    second = col.add_file(found, f"{name}.txt", (DOCUMENT + "\nAppeal.\n").encode(), store)
    report = normalize_all(store, [found])
    assert first.path != second.path and report.skipped == () and len(set(report.documents)) == 2
    assert all(len(pid) <= 80 for pid in report.documents)


# --- footnotes ----------------------------------------------------------------------------------


def test_a_wrapped_line_that_starts_with_a_number_stays_in_its_paragraph():
    text = (
        "The Committee recalled that the arrest of the applicant for his articles about the provincial\n"
        "government had violated Article\n21 of the ICCPR. The court had sentenced him to\n"
        "12 years in prison after a trial that lasted a single day and heard no defence witnesses at all."
    )
    assert passage_spans(text) == [(0, len(text))]


# --- round 2: privacy fails closed, removed files leave, the public corpus is never dropped ------


def test_a_stale_public_flag_never_lets_a_now_private_collection_reach_a_cloud_model(folder, store):
    public = col.create_collection("Indonesia", private=False)
    col.add_file(public, "arden.txt", DOCUMENT.encode(), store)
    normalize_all(store, [public])  # stored as public
    shutil.rmtree(folder / "indonesia")
    col.create_collection("Indonesia")  # re-created private; the stored flag is stale
    cloud = FakeLLM(answer=answer(FACT))
    report = extract_all(store, TAXONOMY, cloud)
    assert cloud.calls == [] and len(report.private_refused) == 1


def test_an_uploaded_document_whose_collection_is_gone_never_reaches_a_cloud_model(folder, store):
    public = col.create_collection("Indonesia", private=False)
    col.add_file(public, "arden.txt", DOCUMENT.encode(), store)
    normalize_all(store, [public])
    shutil.rmtree(folder / "indonesia")
    cloud = FakeLLM(answer=answer(FACT))
    assert len(extract_all(store, TAXONOMY, cloud).private_refused) == 1 and cloud.calls == []


def test_a_collection_folder_without_its_manifest_is_never_silently_re_created(folder, store):
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    (folder / "indonesia" / "collection.yaml").unlink()
    with pytest.raises(col.CollectionError, match="no collection.yaml"):
        col.create_collection("Indonesia", private=False)


def test_a_removed_file_leaves_the_corpus_at_the_next_build(folder, store, tmp_path):
    found = col.create_collection("Indonesia")
    kept = col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    gone = col.add_file(found, "other.txt", (DOCUMENT + "\nA later hearing.\n").encode(), store)
    build(store, tmp_path, local_model(answer(FACT))[0], found)
    gone.path.unlink()
    report = build(store, tmp_path, local_model(answer(FACT))[0], found)
    ids = [doc.id for doc in SqlitePrecedentIndex(tmp_path / "precedents.db", TAXONOMY).documents()]
    assert ids == [f"col-{kept.symbol}"] and report.documents == (f"col-{kept.symbol}",)


def test_a_re_created_collection_does_not_bring_back_the_old_documents(folder, store, tmp_path):
    first = col.create_collection("Beta")
    old = col.add_file(first, "old.txt", DOCUMENT.encode(), store)
    build(store, tmp_path, local_model(answer(FACT))[0], first)
    shutil.rmtree(folder / "beta")
    again = col.create_collection("Beta", private=False)
    new = col.add_file(again, "new.txt", (DOCUMENT + "\nA new file.\n").encode(), store)
    report = build(store, tmp_path, local_model(answer(FACT))[0], again)
    ids = [doc.id for doc in SqlitePrecedentIndex(tmp_path / "precedents.db", TAXONOMY).documents()]
    assert ids == [f"col-{new.symbol}"] and f"col-{old.symbol}" not in report.documents


def test_the_same_bytes_in_two_collections_get_two_ids(folder, store):
    first, second = col.create_collection("a"), col.create_collection("a b")
    one = col.add_file(first, "b-c.txt", DOCUMENT.encode(), store)
    two = col.add_file(second, "c.txt", DOCUMENT.encode(), store)
    assert one.symbol != two.symbol


def test_building_a_collection_never_drops_a_public_corpus_this_computer_cannot_rebuild(folder, tmp_path):
    from tests.precedent_fixtures import FakeEmbedder as CorpusEmbedder
    from tests.precedent_fixtures import build_fixture_corpus

    corpus = build_fixture_corpus(tmp_path / "precedents.db", CorpusEmbedder(), TAXONOMY)
    empty = BuildStore(tmp_path / "fresh-build.db")  # a laptop with the corpus file but no work database
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), empty)
    with pytest.raises(col.CollectionError, match="Copy data/corpus/build.db"):
        build(empty, tmp_path, local_model(answer(FACT))[0], found)
    assert len(SqlitePrecedentIndex(corpus, TAXONOMY).documents()) == 6  # untouched


def test_collection_build_refuses_gemini_for_a_private_collection_before_reading_any_key(folder, store, monkeypatch):
    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)

    def no_client(model_id):
        raise AssertionError("the Gemini client must not be created for a private collection")

    monkeypatch.setattr(extract, "GeminiLLM", no_client)
    assert cli.main(["--work-db", str(store.path), "collection", "build", "--name", "Indonesia", "--model", "gemini", "--model-id", "x"]) == 2


def test_a_footnote_after_a_blank_line_is_left_out():
    body = "The applicant was arrested at his home on 3 May and taken to the police station, where he was questioned for nine hours."
    after = "He was released the next evening without charge, and the police kept his telephone and his two notebooks for three whole weeks."
    text = f"{body}\n\n2 See opinions No. 60/2013.\n\n{after}"
    assert [text[a:b] for a, b in passage_spans(text)] == [body, after]


def test_a_snapshot_is_refused_while_the_corpus_holds_a_private_collection(folder, store, tmp_path):
    from corpus_builder import opensearch

    found = col.create_collection("Indonesia")
    col.add_file(found, "arden.txt", DOCUMENT.encode(), store)
    build(store, tmp_path, local_model(answer(FACT))[0], found)
    assert opensearch.private_documents(tmp_path / "precedents.db") == 1
    with pytest.raises(ValueError, match="private collections"):
        opensearch.snapshot("http://127.0.0.1:1", "s", corpus_path=tmp_path / "precedents.db")
    assert opensearch.private_documents(tmp_path / "missing.db") == 0
