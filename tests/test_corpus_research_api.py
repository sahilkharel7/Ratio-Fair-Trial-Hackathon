"""A saved corpus PDF can enter case intake without re-uploading or losing provenance."""

import hashlib
import sqlite3
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from test_corpus_research import research  # noqa: F401 - pytest fixture registration
from test_workspace_api import (
    api,  # noqa: F401 - pytest fixture registration
    call,
    server_module,
)

from ratio.paths import REPO_ROOT


@pytest.fixture
def corpus_api(monkeypatch, request):
    corpus_reader = request.getfixturevalue("research")
    raw = (REPO_ROOT / "tests/fixtures/corpus/report.pdf").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    with sqlite3.connect(corpus_reader.path) as conn:
        conn.execute(
            "UPDATE documents SET raw_sha256=? WHERE id='ccpr-example'", (sha,)
        )
    path = corpus_reader.path.parent / "raw" / "synthetic" / f"{sha}.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    monkeypatch.setattr(server_module, "CorpusResearch", lambda: corpus_reader)
    return request.getfixturevalue("api"), raw


def test_search_read_download_and_stage_original_with_publication_source(corpus_api):
    (root, library), raw = corpus_api
    assert len(call(root, "/api/precedents")["documents"]) == 2
    assert call(root, "/api/precedents/search?q=presumption")["total_documents"] == 2
    assert call(root, "/api/precedents/ccpr-example")["local_pdf"]
    with urlopen(root + "/api/precedents/ccpr-example/original") as response:
        assert response.read() == raw
        assert response.headers["Content-Type"] == "application/pdf"
    call(root, "/api/precedents/ccpr-example/stage", {})
    staged = library.pending_files()[0]
    assert staged["source_note"] == "https://example.invalid/ccpr-example"
    assert staged["provenance"] == "public"
    assert staged["sha256"] == hashlib.sha256(raw).hexdigest()
    assert not library.list()  # lawyer must confirm case grouping and metadata


def test_unknown_reference_and_oversized_query_are_clear_errors(corpus_api):
    (root, _), _ = corpus_api
    with pytest.raises(HTTPError) as missing:
        call(root, "/api/precedents/unknown/original")
    assert missing.value.code == 404
    with pytest.raises(HTTPError) as long_query:
        call(root, "/api/precedents/search?q=" + "x" * 301)
    assert long_query.value.code == 400


def test_one_pager_download_refuses_missing_or_tampered_saved_sheet(corpus_api):
    pytest.importorskip("reportlab")
    from test_case_briefs import builder

    (root, _), _ = corpus_api
    with pytest.raises(HTTPError) as absent:
        call(root, "/api/precedents/ccpr-example/brief.pdf")
    assert absent.value.code == 404
    # The fixture installs this index for the HTTP handler.
    research_instance = server_module.CorpusResearch()
    doc = next(
        d for d in research_instance.index().documents() if d.id == "ccpr-example"
    )
    pdf = builder.build(doc, research_instance.path.parent / "briefs")
    with urlopen(root + "/api/precedents/ccpr-example/brief.pdf") as response:
        assert response.read() == pdf.read_bytes()
    pdf.write_bytes(b"changed")
    with pytest.raises(HTTPError) as changed:
        call(root, "/api/precedents/ccpr-example/brief.pdf")
    assert changed.value.code == 404
