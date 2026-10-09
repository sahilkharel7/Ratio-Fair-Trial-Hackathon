"""Writes report.pdf, the SYNTHETIC two-page PDF the normaliser tests read. Everything in it is invented.

    .venv/bin/python tests/fixtures/corpus/make_fixture_pdf.py

The text is drawn with the standard Helvetica font in WinAnsi encoding, so pypdf extracts it back
exactly; byte 0xAD is a soft hyphen, which the normaliser must remove.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

PAGES = (
    (
        "SYNTHETIC: an invented fairness report, written for Ratio's tests. No real person, court or trial.",
        "Fairness Report on the Trial of Tamsin Orlo in the Republic of Quillmark",
        "The monitor attended every hearing of the trial. The accused, a journalist, was charged with",
        "spreading false information after publishing an article about a water contract.",
        "She was arrested at her home and was brought before a judge six days later. Her lawyer",
        "was not present at the remand hearing, and the court refused to hear two defence witnes-",
        "ses without giving reasons. The monitor relied on inter\xadnational standards.",
    ),
    (
        "The court extended the detention three times, each time with the same wording as the first",
        "order, and did not address the arguments of the defence about the length of the detention.",
        "This page exists so that the text is long enough to pass the length check of the normaliser.",
    ),
)
OUT = Path(__file__).with_name("report.pdf")


def _escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build(path: Path = OUT) -> None:
    writer = PdfWriter()
    font = writer._add_object(  # noqa: SLF001 - pypdf has no public call for an indirect object
        DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        })
    )
    for lines in PAGES:
        page = writer.add_blank_page(612, 792)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        content = DecodedStreamObject()
        drawn = " ".join(f"({_escape(line)}) Tj T*" for line in lines)
        content.set_data(f"BT /F1 10 Tf 14 TL 50 740 Td {drawn} ET".encode("cp1252"))
        page[NameObject("/Contents")] = writer._add_object(content)  # noqa: SLF001
    with path.open("wb") as handle:
        writer.write(handle)


if __name__ == "__main__":
    build()
