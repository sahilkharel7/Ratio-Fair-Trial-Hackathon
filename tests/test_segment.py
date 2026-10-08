"""Deterministic segmentation: sentences with exact offsets, sections, header and signature blocks."""

import pytest

from ratio.extraction.segment import find_citations, header_value, segment_document, sentence_spans, split_sentences

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


# --- structure of real-world text: PDF extraction, sub-headings, heading styles, quotations ---------


def kinds(text: str) -> list[tuple[str, str, str | None]]:
    return [(seg.kind, text[seg.start : seg.end], seg.chapter) for seg in segment_document(text)]


def test_pdf_text_without_blank_lines_keeps_its_caption_headings_and_sentences():
    text = (
        "MIREVO DISTRICT COURT\nJUDGMENT\nPresiding Judge: Ilena Varda\n"
        "I. PROCEDURAL HISTORY\nThe accused was arrested on 14 February 2025. He was brought before the\n"
        "court on 19 February 2025.\nV. ASSESSMENT OF THE COURT\nThe accused knew that the allegations were false."
    )
    segments = kinds(text)
    assert [k for k, _, _ in segments] == ["header", "header", "header", "heading", "body", "body", "heading", "body"]
    assert segments[5][1] == "He was brought before the\ncourt on 19 February 2025."
    assert segments[-1][2] == "V. ASSESSMENT OF THE COURT"
    assert header_value(text, "Presiding Judge") is not None


def test_a_sub_heading_stays_inside_its_chapter():
    text = (
        "JUDGMENT\nCase no. T-1\n\nIII. SUBMISSIONS OF THE PARTIES\n\nA. The prosecution\n\n"
        "The prosecution argues that the accused knew.\n\nB. The defence\n\nThe defence contends otherwise.\n\n"
        "C. The third party\n\nNo third party appeared.\n\nV. ASSESSMENT OF THE COURT\n\nThe witness was credible.\n"
    )
    chapters = {body: chapter for kind, body, chapter in kinds(text) if kind == "body"}
    assert set(list(chapters.values())[:3]) == {"III. SUBMISSIONS OF THE PARTIES"}  # "C." is a letter here
    assert chapters["The witness was credible."] == "V. ASSESSMENT OF THE COURT"


@pytest.mark.parametrize("heading", ["V. ASSESSMENT OF THE COURT.", "4.1 Assessment of the evidence", "Submissions of the Parties"])
def test_heading_styles_are_recognised(heading):
    text = f"JUDGMENT\nCase no. T-1\n\n{heading}\n\nThe witness was credible.\n"
    assert ("heading", heading, heading) in kinds(text)


def test_a_numbered_paragraph_is_body_text_not_a_heading():
    text = "JUDGMENT\nCase no. T-1\n\n12. The Court notes that the accused was absent.\n"
    assert [k for k, _, _ in kinds(text)] == ["header", "header", "body"]


def test_a_stray_quote_mark_does_not_merge_the_rest_of_the_paragraph():
    text = 'The officer described a 6" folder of printouts. The defence argues that they were never seized. The accused knew.'
    assert len(split_sentences(text, 0, len(text))) == 3


def test_a_quotation_in_guillemets_stays_one_passage():
    text = "Article 5 provides: «No one shall be tried twice. No exception applies.» The court applied it."
    assert len(split_sentences(text, 0, len(text))) == 2


def test_pdf_pages_join_mid_sentence_and_break_after_a_full_stop():
    from ratio.extraction.loader import join_pdf_pages

    assert join_pdf_pages(["He was brought before the", "court on 19 February.", "", "Next page."]) == (
        "He was brought before the\ncourt on 19 February.\n\nNext page."
    )
