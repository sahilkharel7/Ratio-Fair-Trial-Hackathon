"""Display helpers: excerpts with exact offsets, escaped HTML highlights, markdown-safe text."""

from ratio.display import Mark, duration_text, excerpt, excerpt_html, marked_html, md_escape, percent
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
