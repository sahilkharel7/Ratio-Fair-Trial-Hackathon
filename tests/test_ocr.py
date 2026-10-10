"""On-device OCR of scanned PDFs (macOS Vision), on SYNTHETIC page images rendered here."""

import io
import shutil
import sys

import pytest
from PIL import Image, ImageDraw, ImageFont

from ratio.extraction.loader import decode_document
from ratio.library import LibraryStore
from ratio.store import CaseStore

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swiftc") is None, reason="on-device OCR needs macOS and Swift"
)

SENTENCE = "The accused must prove that she did not organise the assembly."


def scanned_pdf(lines):
    """A PDF that is only a picture of the page: no text layer, as from a scanner."""
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((120, 160 + i * 90), line, fill="black", font=font)
    out = io.BytesIO()
    image.save(out, "PDF", resolution=200)
    return out.getvalue()


def test_a_scanned_pdf_is_read_with_on_device_ocr(monkeypatch):
    monkeypatch.setenv("RATIO_OCR", "1")
    text = decode_document("scan.pdf", scanned_pdf(["SYNTHETIC: fictional case.", SENTENCE]))
    assert SENTENCE in text


def test_without_ocr_a_scanned_pdf_has_no_text(monkeypatch):
    monkeypatch.setenv("RATIO_OCR", "0")
    assert decode_document("scan.pdf", scanned_pdf(["SYNTHETIC: fictional case.", SENTENCE])).strip() == ""


def test_an_ocr_helper_that_cannot_run_leaves_the_page_unread(monkeypatch):
    import ratio.ocr

    def broken():
        raise PermissionError("helper is not executable")

    monkeypatch.setenv("RATIO_OCR", "1")
    monkeypatch.setattr(ratio.ocr, "_helper", broken)
    ratio.ocr._ocr_cached.cache_clear()
    with pytest.raises(ratio.ocr.OCRUnavailable):
        ratio.ocr.ocr_pages(b"%PDF-1.4 unique bytes for this test")
    assert decode_document("scan.pdf", scanned_pdf(["SYNTHETIC: a page that stays unread."])).strip() == ""
    ratio.ocr._ocr_cached.cache_clear()


def test_a_scanned_upload_is_staged_with_its_recognised_text(tmp_path, monkeypatch):
    monkeypatch.setenv("RATIO_OCR", "1")
    library = LibraryStore(CaseStore(tmp_path / "cases.db"))
    library.stage({"scan.pdf": scanned_pdf(["SYNTHETIC: fictional case.", SENTENCE])}, provenance="synthetic")
    assert library.pending_files()[0]["problem"] is None
