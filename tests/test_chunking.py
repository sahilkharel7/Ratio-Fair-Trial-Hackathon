"""Chunks keep absolute offsets, end on sentence boundaries and overlap their neighbours."""

from ratio.extraction.chunking import chunk_spans


def sentences(count: int) -> str:
    return " ".join(f"Sentence number {i} says something about the hearing on day {i}." for i in range(count))


def test_short_document_is_a_single_chunk():
    text = "Mr. Venn was present. Counsel was present."
    (chunk,) = chunk_spans(text, max_chars=2500, overlap_chars=200)
    assert (chunk.start, chunk.end) == (0, len(text))


def test_long_document_is_split_on_sentence_boundaries_with_overlap():
    text = sentences(40)
    chunks = chunk_spans(text, max_chars=500, overlap_chars=150)
    assert len(chunks) > 1
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for chunk in chunks:
        assert chunk.end - chunk.start <= 500
        assert text[chunk.end - 1] == "."
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start < previous.end  # overlap
        assert current.start > previous.start  # progress


def test_a_sentence_longer_than_the_limit_gets_its_own_chunk():
    long_sentence = "This sentence is very long " * 40 + "and ends here."
    text = f"Short one. {long_sentence} Short two."
    chunks = chunk_spans(text, max_chars=200, overlap_chars=50)
    assert any(text[c.start : c.end] == long_sentence for c in chunks)
    assert chunks[-1].end == len(text)


def test_empty_text_has_no_chunks():
    assert chunk_spans("   \n", max_chars=100, overlap_chars=10) == ()
