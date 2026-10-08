"""Deterministic segmentation: sentences with exact offsets, sections, header and signature blocks."""

from ratio.extraction.segment import find_citations, header_value, segment_document, sentence_spans

JUDGMENT = """MIREVO DISTRICT COURT, CRIMINAL CHAMBER
JUDGMENT
Case no. MDC-2025-0147
Presiding Judge: Ilena Varda

I. THE CHARGE

The accused is charged with an offence under Article 214(2) of the Penal Code.

II. ASSESSMENT OF THE COURT

Mr. Venn met Judge I. Varda on 19 February 2025. The Court notes the testimony of Ms. Brask. Art. 9 applies, e.g. to detention.
Article 214(2) of the Penal Code provides: "Whoever disseminates false information shall be punished. This includes rumours."

(signed) Ilena Varda, Presiding Judge
"""


def segments_by_kind(kind):
    return [seg for seg in segment_document(JUDGMENT) if seg.kind == kind]


def test_offsets_are_exact():
    for seg in segment_document(JUDGMENT):
        assert JUDGMENT[seg.start : seg.end] == JUDGMENT[seg.start : seg.end].strip()
        assert seg.end > seg.start


def test_header_lines_and_signature_are_marked():
    header = [JUDGMENT[s.start : s.end] for s in segments_by_kind("header")]
    assert header[0] == "MIREVO DISTRICT COURT, CRIMINAL CHAMBER"
    assert "Presiding Judge: Ilena Varda" in header
    signature = [JUDGMENT[s.start : s.end] for s in segments_by_kind("signature")]
    assert signature == ["(signed) Ilena Varda, Presiding Judge"]


def test_headings_set_the_section_of_following_sentences():
    headings = [JUDGMENT[s.start : s.end] for s in segments_by_kind("heading")]
    assert headings == ["I. THE CHARGE", "II. ASSESSMENT OF THE COURT"]
    body = segments_by_kind("body")
    assert body[0].section == "I. THE CHARGE"
    assert all(seg.section == "II. ASSESSMENT OF THE COURT" for seg in body[1:])


def test_abbreviations_initials_and_quoted_statutes_do_not_split_sentences():
    body = [JUDGMENT[s.start : s.end] for s in segments_by_kind("body")]
    assert "Mr. Venn met Judge I. Varda on 19 February 2025." in body
    assert "The Court notes the testimony of Ms. Brask." in body
    assert "Art. 9 applies, e.g. to detention." in body
    assert any(sentence.startswith("Article 214(2)") and sentence.endswith('rumours."') for sentence in body)


def test_sentence_spans_cover_every_segment():
    spans = sentence_spans(JUDGMENT)
    assert len(spans) == len(segment_document(JUDGMENT))


def test_citations_are_found_with_exact_offsets():
    found = [JUDGMENT[s:e] for s, e in find_citations(JUDGMENT)]
    assert "Article 214(2) of the Penal Code" in found
    assert "Art. 9" in found


def test_header_value_reads_labelled_header_lines():
    text = "TRIAL MONITORING NOTE\nHearing: 2\nHearing date: 16 June 2025\n\nBody text here."
    value = header_value(text, "Hearing date")
    assert value is not None
    start, end = value
    assert text[start:end] == "16 June 2025"
    assert header_value(text, "Presiding Judge") is None


def test_plain_note_paragraphs_split_into_sentences():
    note = "HEADER LINE\nHearing date: 2 June 2025\n\nMr. Venn was present. Counsel was present.\nThe hearing ended at 13:40."
    body = [note[s.start : s.end] for s in segment_document(note) if s.kind == "body"]
    assert body == ["Mr. Venn was present.", "Counsel was present.", "The hearing ended at 13:40."]
