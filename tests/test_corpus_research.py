"""Local research: exact original offsets, filters, rebuilds and altered text refusal."""

import sqlite3

import pytest
from fixtures.corpus_b.synthetic import FakeEmbedder, FakeSource, make_doc

from corpus_builder.build_db import build_db
from corpus_builder.store import BuildStore
from ratio.config import default_config
from ratio.corpus_research import CorpusResearch


@pytest.fixture
def research(tmp_path):
    store = BuildStore(tmp_path / "build.db")
    sources = []
    for pid, kind, state in (
        ("ccpr-example", "ccpr_views", "Synthetic State A"),
        ("tw-example", "trialwatch_report", "Synthetic State B"),
    ):
        text = (
            "SYNTHETIC: invented reference for tests only.\n\nThe presumption of\ninnocence was discussed in the fictional hearing. The phrase [proof] is literal.\n"
            * 6
        )
        doc = make_doc(pid, text, kind=kind, state=state)
        store.put_document(doc, doc.url)
        store.put_passages(pid, [(0, len(text), FakeEmbedder().encode([text])[0])])
        sources.append(FakeSource(pid, kind, "Synthetic source", doc.attribution))
    path = tmp_path / "precedents.db"
    meta = build_db(
        store,
        default_config().fact_patterns,
        sources,
        path,
        extraction_model="none; wording search only",
        prompt_sha="",
    )
    assert (meta.documents, meta.facets) == (2, 0)
    return CorpusResearch(path)


def test_public_sources_without_model_facets_are_searchable_with_original_offsets(
    research,
):
    assert len(research.catalogue()["documents"]) == 2
    result = research.search("presumption of innocence")
    assert result["total_documents"] == 2
    assert len(result["hits"]) == 6  # capped at three passages per document
    for hit in result["hits"]:
        doc = research.document(hit["document"]["id"])
        assert hit["text"] == doc["text"][hit["start"] : hit["end"]]
        assert (
            doc["text"][hit["match_start"] : hit["match_end"]]
            == "presumption of\ninnocence"
        )
        assert "outcome" not in hit


def test_filters_and_literal_regex_characters(research):
    assert research.search("[proof]", kind="ccpr_views")["total_documents"] == 1
    assert (
        research.search("presumption", state="Synthetic State B")["total_documents"]
        == 1
    )
    assert research.search("not recorded")["hits"] == []
    with pytest.raises(ValueError):
        research.search("x" * 301)


def test_corrupted_text_is_not_shown_or_searched_after_rebuild(research):
    assert research.document("ccpr-example")
    with sqlite3.connect(research.path) as conn:
        conn.execute(
            "UPDATE documents SET text='Edited after verification' WHERE id='ccpr-example'"
        )
    assert research.document("ccpr-example") is None
    assert research.search("presumption")["total_documents"] == 1
    assert len(research.catalogue()["documents"]) == 1


def test_missing_corpus_has_directories_and_no_invented_results(tmp_path):
    research = CorpusResearch(tmp_path / "missing.db")
    assert research.catalogue()["available"] is False
    assert len(research.catalogue()["directories"]) == 4
    assert research.search("presumption")["hits"] == []
    assert research.document("missing") is None


def test_original_pdf_requires_the_saved_checksum_and_cannot_escape_raw_directory(
    research,
):
    import hashlib

    from ratio.paths import REPO_ROOT

    raw = (REPO_ROOT / "tests/fixtures/corpus/report.pdf").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    with sqlite3.connect(research.path) as conn:
        conn.execute(
            "UPDATE documents SET raw_sha256=? WHERE id='ccpr-example'", (sha,)
        )
    folder = research.path.parent / "raw" / "synthetic"
    folder.mkdir(parents=True)
    path = folder / f"{sha}.pdf"
    path.write_bytes(raw)
    assert research.original_pdf("ccpr-example")[1] == raw
    assert research.document("ccpr-example")["local_pdf"]
    path.write_bytes(b"replaced after download")
    assert research.original_pdf("ccpr-example") is None
    path.unlink()
    outside = research.path.parent / "outside.pdf"
    outside.write_bytes(raw)
    path.symlink_to(outside)
    assert research.original_pdf("ccpr-example") is None


def test_meaning_search_retains_original_passages_and_respects_source_filter(research):
    research._embedder = FakeEmbedder()
    result = research.search("innocence", mode="meaning", kind="ccpr_views")
    assert result["hits"]
    assert "shown results only" in result["method"]
    for hit in result["hits"]:
        assert hit["document"]["kind"] == "ccpr_views"
        doc = research.document(hit["document"]["id"])
        assert hit["text"] == doc["text"][hit["start"] : hit["end"]]
    with pytest.raises(ValueError):
        research.search("innocence", mode="unrecognized")
