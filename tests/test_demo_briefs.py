"""Case drafts are source-bound, queued, cached and isolated from private notes."""

import base64
import io
import json

import pytest
from pypdf import PdfReader, PdfWriter

from ratio.demo_briefs import CaseBriefDraft, DemoBriefService, verified_sections
from ratio.library import LibraryStore
from ratio.paths import REPO_ROOT
from ratio.store import CaseStore
from ratio.testing import make_record
from scripts import groq_case_worker as worker

QUOTE = "The accused is charged with a synthetic offence, and the court examines the presumption of innocence."


def draft(doc_id, quote=QUOTE):
    return CaseBriefDraft.model_validate(
        {
            "sections": [
                {
                    "label": "Charge",
                    "summary": "Synthetic test case only.",
                    "quote": quote,
                    "doc_id": doc_id,
                    "pinpoint": "Invented pinpoint must be replaced",
                }
            ]
        }
    )


def one_page_pdf():
    pdf = PdfWriter()
    pdf.add_page(PdfReader(REPO_ROOT / "tests/fixtures/corpus/report.pdf").pages[0])
    output = io.BytesIO()
    pdf.write(output)
    return output.getvalue()


@pytest.fixture
def source(tmp_path):
    library = LibraryStore(CaseStore(tmp_path / "cases.db"))
    record = make_record([("judgment.txt", "judgment", QUOTE)])
    library.cases.save_case(record)
    library.register(record)
    library.add_note(record.case_id, "Private working note must stay on this computer.")
    return library, record


def test_draft_citation_is_exact_and_pinpoint_is_computed(source):
    _, record = source
    result = verified_sections(record, draft(record.documents[0].id))[0]
    span = result["span"]
    assert record.documents[0].text[span["start"] : span["end"]] == QUOTE
    assert "stored text line 1" in result["pinpoint"]
    assert "Invented" not in result["pinpoint"]
    with pytest.raises(ValueError, match="absent"):
        verified_sections(
            record,
            draft(record.documents[0].id, "An invented quote that does not occur."),
        )


def test_generated_draft_is_cached_without_notes_or_database_path(source):
    library, record = source
    seen = []
    raw = one_page_pdf()

    def runner(payload):
        seen.append(payload)
        return {
            "sections": draft(record.documents[0].id).model_dump()["sections"],
            "pdf": base64.b64encode(raw).decode(),
        }

    service = DemoBriefService(library, runner=runner)
    try:
        service.start(record.case_id)
        service._pool.submit(lambda: None).result(timeout=3)
        result = service.get(record.case_id)
        assert result["status"] == "ready"
        assert service.pdf(record.case_id) == raw
        service.start(record.case_id)
        assert len(seen) == 1
        assert "Private working note" not in json.dumps(seen)
        assert str(library.cases.path) not in json.dumps(seen)
        assert "API_KEY" not in json.dumps(seen)
        with library.connect() as conn:
            conn.execute(
                "UPDATE case_ai_briefs SET pdf=? WHERE case_id=?",
                (b"changed", record.case_id),
            )
        assert service.get(record.case_id)["status"] == "failed"
        assert service.pdf(record.case_id) is None
        changed = record.model_copy(
            update={
                "meta": record.meta.model_copy(
                    update={"charge_type": "Changed source metadata"}
                )
            }
        )
        library.cases.save_case(changed)
        assert service.get(record.case_id)["status"] == "not_generated"
        assert service.pdf(record.case_id) is None
    finally:
        service.close()


def test_failed_citation_does_not_publish_a_pdf(source):
    library, record = source

    def runner(_):
        return {
            "sections": draft(
                record.documents[0].id, "An invented quote not present in this source."
            ).model_dump()["sections"],
            "pdf": "",
        }

    service = DemoBriefService(library, runner=runner)
    try:
        service.start(record.case_id)
        service._pool.submit(lambda: None).result(timeout=3)
        assert service.get(record.case_id)["status"] == "failed"
        assert service.pdf(record.case_id) is None
        assert library.cases.load_case(record.case_id) == record
    finally:
        service.close()


def test_worker_refuses_private_input_before_model_or_renderer():
    class Cloud:
        def complete_json(self, **kwargs):
            pytest.fail("Private text reached the cloud")

    with pytest.raises(ValueError, match="Private"):
        worker.generate({"provenance": "private"}, llm=Cloud())


def test_excerpt_selection_is_bounded_and_preserves_original_characters():
    text = (QUOTE + "\n") * 300
    selected = worker.excerpts(
        [
            {
                "id": "fixture",
                "title": "Synthetic fixture",
                "type": "judgment",
                "text": text,
            }
        ],
        budget=1800,
    )
    snippets = selected[0]["excerpts"]
    assert sum(len(e["text"]) for e in snippets) <= 1800
    assert all(e["text"] in text for e in snippets)


def test_model_selects_source_passages_without_writing_quotations(monkeypatch):
    text = "Opening heading.\n\n" + QUOTE
    payload = {
        "provenance": "synthetic",
        "title": "Synthetic test",
        "court": "Test court",
        "source_note": "Synthetic test",
        "sources": [
            {"id": "doc-1", "title": "Test source", "type": "judgment", "text": text}
        ],
    }

    class Selector:
        def complete_json(self, *, user, schema, **kwargs):
            passages = json.loads(user)["passages"]
            return schema.model_validate(
                {
                    "sections": [
                        {
                            "label": "Charge",
                            "summary": "Synthetic example only.",
                            "quote_id": passages[0]["quote_id"],
                        }
                    ]
                }
            )

    monkeypatch.setattr(worker, "render", lambda *args: one_page_pdf())
    result = worker.generate(payload, llm=Selector())
    section = result["sections"][0]
    assert (
        text[section["source_start"] : section["source_start"] + len(section["quote"])]
        == section["quote"]
    )
    assert "stored text line" in section["pinpoint"]


def test_model_cannot_select_an_invented_source_passage(monkeypatch):
    class Selector:
        def complete_json(self, *, schema, **kwargs):
            return schema.model_validate(
                {
                    "sections": [
                        {
                            "label": "Charge",
                            "summary": "Synthetic example only.",
                            "quote_id": "invented",
                        }
                    ]
                }
            )

    with pytest.raises(ValueError):
        worker.generate(
            {
                "provenance": "synthetic",
                "title": "Test",
                "court": "Test",
                "sources": [
                    {"id": "doc-1", "title": "Test", "type": "judgment", "text": QUOTE}
                ],
            },
            llm=Selector(),
        )
