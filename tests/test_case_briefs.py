"""Reading sheets reject invented citations, changed sources and tampered PDFs."""

import hashlib
import importlib.util
import io
import json

import pytest
from fixtures.corpus_b.synthetic import make_doc
from pypdf import PdfReader

from ratio.case_briefs import BriefDraft, load_brief, verify_draft
from ratio.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "brief_builder", REPO_ROOT / "scripts/build_case_briefs.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def draft(quote):
    return BriefDraft.model_validate(
        {
            "sections": [
                {
                    "label": "Synthetic test",
                    "summary": "Invented fixture only; never an actual legal outcome.",
                    "quote": quote,
                    "pinpoint": "Synthetic text line 1",
                }
            ]
        }
    )


def test_quote_verification_is_contiguous_and_exact():
    doc = make_doc(
        "ccpr-example",
        "SYNTHETIC legal example. A charge was recorded. The court considered presumption of innocence.",
    )
    section = verify_draft(doc, draft("A charge was recorded."))[0]
    assert doc.text[section["start"] : section["end"]] == section["quote"]
    with pytest.raises(ValueError, match="absent"):
        verify_draft(doc, draft("A charge ... innocence."))


def test_one_page_pdf_and_manifest_refuse_changed_source_and_pdf(tmp_path):
    pytest.importorskip("reportlab")
    doc = make_doc(
        "ccpr-example",
        "SYNTHETIC legal example. A charge was recorded. The court considered presumption of innocence.",
    )
    pdf = builder.build(doc, tmp_path)
    assert len(PdfReader(io.BytesIO(pdf.read_bytes())).pages) == 1
    assert "SYNTHETIC" in PdfReader(pdf).pages[0].extract_text()
    assert load_brief(doc, tmp_path)
    pdf.write_bytes(b"tampered")
    assert load_brief(doc, tmp_path) is None
    builder.build(doc, tmp_path)
    changed = doc.model_copy(
        update={"text_sha256": hashlib.sha256(b"changed").hexdigest()}
    )
    assert load_brief(changed, tmp_path) is None
    manifest = tmp_path / (doc.id + ".json")
    data = json.loads(manifest.read_text())
    data["sections"][0]["quote"] = "Invented quotation that never appears."
    manifest.write_text(json.dumps(data))
    assert load_brief(doc, tmp_path) is None


def test_private_source_never_reaches_cloud_or_renderer(tmp_path):
    doc = make_doc("ccpr-example", "SYNTHETIC private case.").model_copy(
        update={"private": True}
    )

    class Cloud:
        def complete_json(self, **kwargs):
            pytest.fail("Private document reached a cloud model")

    with pytest.raises(ValueError, match="Private"):
        builder.build(doc, tmp_path, llm=Cloud())


def test_cloud_draft_must_have_a_real_source_quote(tmp_path):
    doc = make_doc("ccpr-example", "SYNTHETIC public case with invented legal facts.")

    class Cloud:
        def complete_json(self, **kwargs):
            return draft("This invented conclusion is absent from the source.")

    with pytest.raises(ValueError, match="absent"):
        builder.build(doc, tmp_path, llm=Cloud())
    assert not list(tmp_path.iterdir())
