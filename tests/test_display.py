"""Display helpers: excerpts with exact offsets, escaped HTML highlights, markdown-safe text."""

from ratio.display import Mark, duration_text, excerpt, excerpt_html, guarantee_badge, marked_html, md_escape, percent
from ratio.schema import SourceSpan

TEXT = "Line one.\nThe accused was arrested on 14 February 2025 at his home.\nLine three is here."


def span_of(quote: str, text: str = TEXT, doc_id: str = "c/doc.txt") -> SourceSpan:
    start = text.index(quote)
    return SourceSpan(doc_id=doc_id, start=start, end=start + len(quote), text=quote)


def test_excerpt_keeps_the_exact_span_and_cuts_context_at_word_boundaries():
    span = span_of("arrested on 14 February 2025")
    part = excerpt(TEXT, span, context_chars=15)
    assert part.match == span.text and part.line == 2
    assert part.clipped_before and part.clipped_after
    assert not part.before.startswith(("ne", "e ")) and part.before.endswith("was ")
    assert part.after.startswith(" at his") and not part.after.endswith("ho")


def test_excerpt_of_a_short_document_is_not_clipped():
    span = span_of("Line one.")
    part = excerpt(TEXT, span, context_chars=10_000)
    assert (part.before, part.clipped_before, part.clipped_after) == ("", False, False)
    assert part.before + part.match + part.after == TEXT


def test_document_text_is_escaped_so_it_cannot_inject_markup():
    text = 'He wrote <script>alert("x")</script> & <b>bold</b> at noon.'
    html = excerpt_html(excerpt(text, span_of("<b>bold</b>", text), context_chars=100))
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert '<mark class="ratio-span">&lt;b&gt;bold&lt;/b&gt;</mark>' in html


def test_marks_wrap_exact_offsets_and_overlaps_merge():
    text = "abcdefghij"
    html = marked_html(text, [Mark(2, 5, "paraphrase", "2"), Mark(4, 7, "verbatim", "1"), Mark(8, 9, "excluded", "statute quote")])
    assert html == (
        '<div class="ratio-doc">ab<mark class="ratio-verbatim">cdefg</mark><sup class="ratio-label">2,1</sup>h'
        '<span class="ratio-excluded">i</span><span class="ratio-tag">statute quote</span>j</div>'
    )


def test_marks_use_document_offsets_when_rendering_a_slice():
    html = marked_html("defg", [Mark(4, 6, "verbatim")], offset=3)
    assert html == '<div class="ratio-doc">d<mark class="ratio-verbatim">ef</mark>g</div>'


def test_markdown_escape_keeps_text_literal():
    escaped = md_escape("Costs: $100 [link](http://x) :red[alarm] *a* <b>")
    for literal in (r"\$100", r"\[link\]\(http\:", r"\:red\[alarm\]", r"\*a\*", r"\<b\>"):
        assert literal in escaped
    assert md_escape("plain words") == "plain words"


def test_durations_and_percentages():
    assert duration_text(96, 144) == "between 4 and 6 days (96 to 144 hours; dates are day-level)"
    assert duration_text(30, 30) == "30 hours"
    assert (percent(0.6001), percent(None)) == ("60%", "n/a")


def test_reuse_marks_stay_on_the_document_each_pair_belongs_to():
    from ratio.display import reuse_marks
    from ratio.results import CharRange, ExcludedPassage, ReusePair, ReuseResult

    judgment = "The court finds the accused stole the car. Article 5 reads: theft is a crime."
    first, amended = "Count 1: the accused stole the car.", "Particular 12: the date of count 1 is amended."
    pair = ReusePair(
        id="p1", kind="verbatim", flag_id="f1",
        judgment=span_of("the accused stole the car", judgment, "c/judgment.txt"),
        indictment=span_of("the accused stole the car", first, "c/indictment.txt"),
        indictment_ranges=(CharRange(start=9, end=34),),
    )  # fmt: skip
    statute = span_of("Article 5 reads: theft is a crime.", judgment, "c/judgment.txt")
    reuse = ReuseResult(
        judgment_doc_id="c/judgment.txt", indictment_doc_id="c/indictment.txt",
        indictment_doc_ids=("c/indictment.txt", "c/amended.txt"), pairs=(pair,),
        excluded=(ExcludedPassage(passage_id="x", span=statute, reason="statute_quote"),),
    )  # fmt: skip
    numbers = {"f1": "1"}
    assert reuse_marks(reuse, numbers, "c/amended.txt", str.upper) == []
    assert reuse_marks(reuse, numbers, "c/indictment.txt", str.upper) == [Mark(9, 34, "verbatim", "1")]
    on_judgment = reuse_marks(reuse, numbers, "c/judgment.txt", str.upper)
    assert Mark(statute.start, statute.end, "excluded", "STATUTE_QUOTE") in on_judgment
    assert Mark(pair.judgment.start, pair.judgment.end, "verbatim", "1") in on_judgment
    assert amended  # the amended indictment exists but has no match: nothing may be painted on it


def test_a_guarantee_with_unlabelled_notes_is_shown_as_incomplete_not_as_no_evidence():
    from types import SimpleNamespace

    assert guarantee_badge(SimpleNamespace(status="no_evidence", unlabelled_notes=1)) == "incomplete"
    assert guarantee_badge(SimpleNamespace(status="no_evidence", unlabelled_notes=0)) == "no_evidence"
    assert guarantee_badge(SimpleNamespace(status="evidence_of_violation", unlabelled_notes=2)) == "evidence_of_violation"
