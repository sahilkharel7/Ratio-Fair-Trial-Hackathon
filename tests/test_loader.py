"""Loading a case folder: manifest, SYNTHETIC markers, headers, rulings, and input safety."""

import datetime as dt
from pathlib import Path

import pytest

from ratio.extraction.build import load_case
from ratio.extraction.loader import LoaderError, decode_document

MARKER = "SYNTHETIC: fictional case; all people, courts, country, language and laws are invented.\n"

NOTE = MARKER + (
    "TRIAL MONITORING NOTE\nHearing: 1\nHearing date: 2 June 2025\nMonitor: R.T.\n\n"
    "Mr. Venn was present in the courtroom. Defence counsel was present.\n"
)
JUDGMENT = MARKER + (
    "TEST COURT\nJUDGMENT\nPresiding Judge: Ilena Varda\n\nI. ASSESSMENT\n\n"
    "The Court finds the accused guilty of the offence under Article 214(2) of the Penal Code.\n"
)
MANIFEST = """synthetic: true
case_id: test-case
title: "Republic v. Test"
court: "Test Court"
charge_type: "Test charge"
data_provenance: synthetic
documents:
  - {path: notes/hearing_1.txt, type: monitoring_note, title: "Hearing 1"}
  - {path: judgment.txt, type: judgment, title: "Judgment"}
rulings: rulings.yaml
"""
RULINGS = """synthetic: true
rulings:
  - {code: verdict_conviction, doc: judgment.txt, quote: "finds the accused guilty", judge: "Ilena Varda", date: 2025-07-14}
"""


def write_case(folder: Path, *, manifest: str = MANIFEST, note: str = NOTE, judgment: str = JUDGMENT, rulings: str = RULINGS) -> Path:
    (folder / "notes").mkdir(parents=True)
    (folder / "case.yaml").write_text(manifest, encoding="utf-8")
    (folder / "notes" / "hearing_1.txt").write_text(note, encoding="utf-8")
    (folder / "judgment.txt").write_text(judgment, encoding="utf-8")
    (folder / "rulings.yaml").write_text(rulings, encoding="utf-8")
    return folder


def test_loads_documents_with_case_scoped_ids_and_strips_the_marker(tmp_path):
    record = load_case(write_case(tmp_path / "case"))
    assert [doc.id for doc in record.documents] == ["test-case/notes/hearing_1.txt", "test-case/judgment.txt"]
    assert all(doc.synthetic for doc in record.documents)
    assert not record.documents[0].text.startswith("SYNTHETIC")
    assert record.meta.synthetic and record.meta.data_provenance == "synthetic"


def test_hearing_date_header_sets_document_date_and_observations(tmp_path):
    record = load_case(write_case(tmp_path / "case"))
    note = record.document("test-case/notes/hearing_1.txt")
    assert note.date == dt.date(2025, 6, 2)
    assert note.date_span is not None and note.date_span.text == "2 June 2025"
    assert [obs.text for obs in record.observations] == [
        "Mr. Venn was present in the courtroom.",
        "Defence counsel was present.",
    ]
    assert all(obs.hearing_date == dt.date(2025, 6, 2) for obs in record.observations)
    hearings = [event for event in record.events if event.type == "hearing"]
    assert len(hearings) == 1 and hearings[0].precision == "date"


def test_presiding_judge_comes_from_the_judgment_header_with_a_span(tmp_path):
    record = load_case(write_case(tmp_path / "case"))
    assert record.meta.presiding_judge == "Ilena Varda"
    assert record.meta.presiding_judge_span is not None
    assert record.meta.presiding_judge_span.text == "Ilena Varda"


def test_rulings_are_aligned_to_exact_spans(tmp_path):
    record = load_case(write_case(tmp_path / "case"))
    (ruling,) = record.rulings
    assert ruling.code == "verdict_conviction"
    assert ruling.span.text == "finds the accused guilty"
    assert ruling.judge_name == "Ilena Varda"


def test_passages_and_citations_are_built_for_court_documents(tmp_path):
    record = load_case(write_case(tmp_path / "case"))
    kinds = {passage.kind for passage in record.passages}
    assert {"header", "heading", "body"} <= kinds
    assert any(citation.text == "Article 214(2) of the Penal Code" for citation in record.citations)


def test_synthetic_document_without_marker_is_refused(tmp_path):
    folder = write_case(tmp_path / "case", note=NOTE.replace(MARKER, ""))
    with pytest.raises(LoaderError, match="SYNTHETIC"):
        load_case(folder)


def test_ruling_quote_not_in_document_is_refused(tmp_path):
    folder = write_case(tmp_path / "case", rulings=RULINGS.replace("finds the accused guilty", "acquits the accused"))
    with pytest.raises(LoaderError, match="not found"):
        load_case(folder)


def test_manifest_paths_cannot_escape_the_case_folder(tmp_path):
    folder = write_case(tmp_path / "case", manifest=MANIFEST.replace("path: judgment.txt", "path: ../judgment.txt"))
    with pytest.raises(LoaderError):
        load_case(folder)


def test_windows_line_endings_are_normalised(tmp_path):
    folder = write_case(tmp_path / "case", note=NOTE.replace("\n", "\r\n"))
    record = load_case(folder)
    assert "\r" not in record.document("test-case/notes/hearing_1.txt").text


def test_decode_rejects_unsupported_types_and_bad_encoding():
    with pytest.raises(LoaderError, match="not supported"):
        decode_document("evil.exe", b"MZ")
    with pytest.raises(LoaderError, match="UTF-8"):
        decode_document("note.txt", b"\xff\xfe\x00bad")


@pytest.mark.parametrize("value", ["2 June", "02/06/25", "today", "10:05"])
def test_a_hearing_date_header_without_a_full_date_is_refused(tmp_path, value):
    folder = write_case(tmp_path / "case", note=NOTE.replace("Hearing date: 2 June 2025", f"Hearing date: {value}"))
    with pytest.raises(LoaderError, match="Hearing date"):
        load_case(folder)


def test_yaml_anchors_and_aliases_are_refused(tmp_path):
    bomb = MANIFEST + 'extra: &a ["x", "x"]\nmore: [*a, *a]\n'
    with pytest.raises(LoaderError, match="aliases"):
        load_case(write_case(tmp_path / "case", manifest=bomb))


def test_a_note_without_a_caption_keeps_its_first_paragraph(tmp_path):
    note = MARKER + "Mr. Venn was present in the courtroom.\n\nDefence counsel was present.\n"
    record = load_case(write_case(tmp_path / "case", note=note))
    assert [obs.text for obs in record.observations] == ["Mr. Venn was present in the courtroom.", "Defence counsel was present."]
