"""Quote alignment: model quotes are only used to locate source text, never shown as evidence."""

import pytest

from ratio.extraction.align import (
    find_all,
    find_unique,
    locate_quote,
    negation_mismatch,
    sentence_bounds,
)

TEXT = (
    "Mr. Venn was present in the courtroom. No interpreter was present at the hearing. "
    "The presiding judge refused the request and ordered the trial to continue. "
    "Counsel said she met Mr. Venn on 21 February 2025, a week after his arrest."
)


def test_find_all_and_unique():
    assert find_all(TEXT, "Mr. Venn") == (0, TEXT.index("Mr. Venn", 1))
    assert find_unique(TEXT, "Mr. Venn") is None
    assert find_unique(TEXT, "ordered the trial to continue") == TEXT.index("ordered the trial")


def test_exact_quote_is_located():
    located = locate_quote(TEXT, "refused the request and ordered the trial")
    assert located is not None
    assert located.match == "exact"
    assert TEXT[located.start : located.end] == "refused the request and ordered the trial"


def test_ambiguous_quote_prefers_the_occurrence_inside_the_window():
    text = "The hearing was adjourned without reasons. Later, the hearing was adjourned without reasons."
    second = text.rindex("hearing was adjourned")
    located = locate_quote(text, "hearing was adjourned without reasons", window=(second - 4, len(text)))
    assert located is not None and located.start == second


def test_ambiguous_quote_without_window_is_dropped():
    text = "The court was open. The court was open."
    assert locate_quote(text, "The court was open", min_chars=5) is None


def test_whitespace_and_curly_quote_differences_are_normalised():
    text = "Counsel said:  “the file\nwas late” to the court."
    located = locate_quote(text, 'said: "the file was late" to the court', min_chars=5)
    assert located is not None
    assert located.match == "normalized"
    assert text[located.start : located.end].startswith("said:")
    assert text[located.start : located.end].endswith("to the court")


def test_long_quote_with_a_changed_word_matches_fuzzily_and_snaps_to_words():
    quote = "The presiding judge rejected the request and ordered the trial to continue"
    located = locate_quote(TEXT, quote)
    assert located is not None
    assert located.match == "fuzzy"
    found = TEXT[located.start : located.end]
    assert found.startswith("The presiding judge")
    assert found.endswith("continue")


def test_fuzzy_match_requires_identical_digits():
    quote = "Counsel said she met Mr. Venn on 22 February 2025, a week after"
    assert locate_quote(TEXT, quote) is None


def test_short_and_placeholder_quotes_are_rejected():
    assert locate_quote(TEXT, "Mr. Venn") is None  # shorter than the minimum
    assert locate_quote(TEXT, "unknown") is None
    assert locate_quote(TEXT, "   ") is None


def test_missing_quote_returns_none():
    assert locate_quote(TEXT, "an interpreter translated every word of the hearing") is None


def test_sentence_bounds_expand_a_quote_to_its_sentence():
    start = TEXT.index("interpreter was present")
    s, e = sentence_bounds(TEXT, start, start + len("interpreter was present"))
    assert TEXT[s:e] == "No interpreter was present at the hearing."


def test_negation_dropped_from_quote_is_detected():
    sentence = "No interpreter was present at the hearing."
    assert negation_mismatch(sentence, "interpreter was present at the hearing")
    assert not negation_mismatch(sentence, "No interpreter was present")
    assert not negation_mismatch("Mr. Venn was present.", "Mr. Venn was present")


@pytest.mark.parametrize("quote", ["Mr. Venn was present in the courtroom.", "Mr. Venn was present in the courtroom"])
def test_trailing_punctuation_does_not_matter(quote):
    located = locate_quote(TEXT, quote)
    assert located is not None and located.start == 0


def test_typographic_apostrophes_still_count_as_negation():
    assert negation_mismatch("The defendant didn\u2019t have an interpreter.", "The defendant have an interpreter")
