"""Event dates from real-world model replies and real-world documents: joined dates, dates split
by line breaks, court date formats, a date taken from the wrong clause, partial dates, times."""

import datetime as dt
import re

import pytest

from ratio.config import load_config
from ratio.extraction.chunking import chunk_spans
from ratio.extraction.extract import _dedupe, _event_key, extract_record
from ratio.extraction.prompts import ChunkExtraction
from ratio.llm import PromptTooLong
from ratio.testing import FakeLLM, make_record

SETTINGS = load_config().settings.extraction
CAPTION = "JUDGMENT\nCase no. T-1\n\n"


def numbered_sentences(user: str) -> dict[int, str]:
    return {int(n): text for n, text in re.findall(r"^\[(\d+)\] (.*)$", user, re.MULTILINE)}


def extract(body: str, *events: dict, doc_type: str = "judgment"):
    """Run extraction on one document; the scripted model returns ``events`` for every chunk."""
    path = "note.txt" if doc_type == "monitoring_note" else "judgment.txt"
    header = "Hearing date: 2 June 2025\n\n" if doc_type == "monitoring_note" else CAPTION
    record = make_record([(path, doc_type, header + body)])

    def respond(system, user, schema, purpose):
        assert schema is ChunkExtraction
        sentences = numbered_sentences(user)
        numbered = [{**e, "sentence": next((n for n, s in sentences.items() if e["quote"][:20] in s), 0)} for e in events]
        return {"events": numbered, "arguments": []}

    llm = FakeLLM(respond)
    extracted, report = extract_record(record, llm, SETTINGS)
    return [e for e in extracted.events if e.type != "hearing"], report, llm


def model_event(type_: str, quote: str, date_text: str, iso_date: str) -> dict:
    return {"type": type_, "quote": quote, "date_text": date_text, "iso_date": iso_date, "actors": []}


EXTENSION = "His detention was extended on 19 March 2025 and again on 14 May 2025."


def test_joined_dates_in_one_answer_give_one_event_each():
    # The shape the real model returned for the demo judgment (one event, two dates).
    events, _, _ = extract(EXTENSION, model_event("detention_extension", EXTENSION[:-1], "19 March 2025, 14 May 2025", "2025-03-19, 2025-05-14"))
    assert [(e.parsed_date.date(), e.date_span.text, e.needs_review) for e in events] == [
        (dt.date(2025, 3, 19), "19 March 2025", False),
        (dt.date(2025, 5, 14), "14 May 2025", False),
    ]


def test_joined_dates_without_matching_model_dates_are_kept_for_review():
    events, _, _ = extract(EXTENSION, model_event("detention_extension", EXTENSION[:-1], "19 March 2025 and 14 May 2025", "2025-03-19"))
    assert len(events) == 2
    assert all("several dates" in " ".join(e.review_reasons) for e in events)


def test_a_date_broken_by_a_line_break_is_found_and_the_prompt_stays_one_line():
    body = "The accused was arrested in the early morning of 14\nFebruary 2025 at his home."
    events, _, llm = extract(body, model_event("arrest", "The accused was arrested in the early morning of 14 February 2025", "14 February 2025", "2025-02-14"))
    (arrest,) = events
    assert arrest.parsed_date == dt.datetime(2025, 2, 14) and arrest.date_span.text == "14\nFebruary 2025"
    assert "[1] The accused was arrested in the early morning of 14 February 2025 at his home." in llm.calls[0].user


@pytest.mark.parametrize("written", ["14th February, 2025", "the 14th day of February, 2025", "14-02-2025", "2025/02/14"])
def test_court_date_formats_are_sent_to_the_model_and_parsed(written):
    body = f"The accused was arrested on {written} at his home."
    events, _, llm = extract(body, model_event("arrest", f"The accused was arrested on {written}", written, "2025-02-14"))
    assert len(llm.calls) == 1
    assert [e.parsed_date for e in events] == [dt.datetime(2025, 2, 14)]


def test_a_date_from_another_clause_is_replaced_by_the_date_in_the_quote():
    body = "The accused was arrested on 14 February 2025 and was brought before the court on 19 February 2025."
    events, _, _ = extract(body, model_event("first_appearance", "was brought before the court on 19 February 2025", "14 February 2025", "2025-02-14"))
    (appearance,) = events
    assert appearance.parsed_date == dt.datetime(2025, 2, 19)
    assert appearance.needs_review and "quote's date was used" in " ".join(appearance.review_reasons)


def test_part_of_a_longer_date_is_not_accepted():
    body = "The accused was arrested on 14 February 2025."
    events, report, _ = extract(body, model_event("arrest", "The accused was arrested on 14 February 2025", "4 February 2025", "2025-02-04"))
    assert events == []
    assert any("no date written in its sentence" in reason for reason in report.dropped)


def test_a_time_is_never_turned_into_a_date():
    body = "The hearing began at 10:10 before the presiding judge. Mr. Venn was present."
    events, _, _ = extract(body, model_event("first_appearance", "The hearing began at 10:10 before the presiding judge", "10:10", ""), doc_type="monitoring_note")
    (start,) = events
    assert start.parsed_date is None and "could not be parsed" in " ".join(start.review_reasons)


def test_a_court_document_without_recognisable_dates_is_reported():
    _, report, llm = extract("The Court has examined the evidence. The accused is guilty.")
    assert llm.calls == ()
    assert any("no written date was recognised" in reason for reason in report.dropped)


def test_a_part_too_long_for_the_model_is_skipped_and_reported_while_the_rest_is_read():
    record = make_record(
        [
            ("judgment.txt", "judgment", CAPTION + "The accused was arrested on 14 February 2025."),
            ("note.txt", "monitoring_note", "Hearing date: 2 June 2025\n\nMr. Venn was present."),
        ]
    )

    def respond(system, user, schema, purpose):
        if "Title: judgment.txt" in user:
            raise PromptTooLong("extraction prompt too long for the context window")
        return {"events": [], "arguments": []}

    _, report = extract_record(record, FakeLLM(respond), SETTINGS)
    assert report.model_calls == 2
    assert any(reason.startswith("judgment.txt: part 1 skipped") for reason in report.dropped)


def test_chunks_never_repeat_a_subset_of_the_previous_chunk():
    text = "Short one. Short two here. Short three is here. " + "Long " * 700 + "end."
    chunks = chunk_spans(text, max_chars=2500, overlap_chars=200)
    for earlier, later in zip(chunks, chunks[1:]):
        assert not (later.start >= earlier.start and later.end <= earlier.end)


def test_the_clean_copy_of_a_repeated_event_is_kept():
    events, _, _ = extract(EXTENSION, model_event("detention_extension", EXTENSION[:-1], "19 March 2025", "2025-03-19"))
    clean = events[0]
    reviewed = clean.model_copy(update={"needs_review": True, "review_reasons": ("model and dateparser disagree",)})
    assert _dedupe([reviewed, clean], _event_key) == [clean]


def written_date(sentence: str) -> str:
    return re.search(r"\d{1,2} [A-Z][a-z]+ \d{4}", sentence).group(0)


@pytest.mark.parametrize(
    ("event_type", "sentence"),
    [
        ("counsel_access", "The screenshots offered by the prosecution were excluded on 1 September 2025."),
        ("trial_start", "The trial was first set for 12 May 2025."),
        ("trial_start", "Time spent in detention since 9 September 2024 counts toward the sentence."),
        ("charge", "On 4 March 2026 he signed a written statement prepared by investigators."),
        ("first_appearance", "On 15 February 2025 he was brought before the Public Prosecutor."),
        ("first_appearance", "On 15 February 2025 he appeared before the Public Prosecutor, who questioned him."),
    ],
)
def test_an_event_whose_sentence_does_not_name_that_kind_of_event_is_dropped(event_type, sentence):
    events, report, _ = extract(sentence, model_event(event_type, sentence[:-1], written_date(sentence), ""))
    assert events == []
    assert any("does not mention" in reason for reason in report.dropped)


@pytest.mark.parametrize(
    ("event_type", "sentence"),
    [
        ("counsel_access", "She stated that she was first able to meet Mr. Venn on 21 February 2025."),
        ("trial_start", "The trial opened on 2 June 2025 before Judge Ilena Varda."),
        ("first_appearance", "On 10 September 2024 she was brought before Judge Petra Hollin."),
        ("first_appearance", "A detention hearing was held on 15 February 2025, at which his detention was ordered."),
        ("charge", "The indictment was filed on 2 December 2024."),
    ],
)
def test_an_event_whose_sentence_names_it_is_kept(event_type, sentence):
    events, _, _ = extract(sentence, model_event(event_type, sentence[:-1], written_date(sentence), ""))
    assert [e.type for e in events] == [event_type]
