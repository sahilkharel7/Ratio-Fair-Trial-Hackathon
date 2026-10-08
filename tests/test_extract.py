"""LLM extraction with a scripted model: quotes become source spans, dates are parsed in code."""

import datetime as dt
import re

import pytest

from ratio.config import load_config
from ratio.extraction.build import load_case
from ratio.extraction.extract import extract_record
from ratio.extraction.prompts import ChunkExtraction
from ratio.paths import DEMO_CASE_DIR
from ratio.testing import FakeLLM

SETTINGS = load_config().settings.extraction


def event(type_, quote, date_text, iso_date, actors=()):
    return {"type": type_, "quote": quote, "date_text": date_text, "iso_date": iso_date, "actors": list(actors)}


SCRIPT = {
    "Judgment (14 July 2025)": {
        "events": [
            event("detention_extension", "His detention was extended on 19 March 2025 and again on 14 May 2025", "19 March 2025", "2025-03-19"),
            event("detention_extension", "His detention was extended on 19 March 2025 and again on 14 May 2025", "14 May 2025", "2025-05-14"),
        ],
        "arguments": [],
    },
    "Monitoring note, hearing 4 (14 July 2025)": {
        "events": [],
        "arguments": [{"party": "defense", "quote": "Ms. Brask announced that the defence would appeal"}],
    },
    "Indictment (3 March 2025)": {
        "events": [
            event("arrest", "The accused was arrested at his home in Mirevo in the early morning of 14 February 2025", "14 February 2025", "2025-02-14"),
            event("first_appearance", "On 19 February 2025 the accused was brought before Judge Ilena Varda", "19 February 2025", "2025-02-18"),
            event("charge", "The Public Prosecutor filed this indictment on a date in spring", "", ""),
        ],
        "arguments": [{"party": "prosecution", "quote": "The conduct of the accused was deliberate"}],
    },
    "Monitoring note, hearing 2 (16 June 2025)": {
        "events": [],
        "arguments": [
            {
                "party": "defense",
                "quote": "Defence counsel argued that the messages presented by the prosecution had been extracted from Mr. Venn's mobile telephone",
            }
        ],
    },
}


def sentence_number(user: str, quote: str) -> int:
    for line in user.splitlines():
        match = re.match(r"\[(\d+)\] (.*)", line)
        if match and quote[:25] in match.group(2):
            return int(match.group(1))
    return 0


def responder(system, user, schema, purpose):
    assert schema is ChunkExtraction and purpose == "extraction"
    for title, reply in SCRIPT.items():
        if f"Title: {title}" in user:
            events = [{**e, "sentence": sentence_number(user, e["quote"])} for e in reply["events"]]
            return {"events": events, "arguments": reply["arguments"]}
    return {"events": [], "arguments": []}


@pytest.fixture(scope="module")
def extracted():
    base = load_case(DEMO_CASE_DIR)
    llm = FakeLLM(responder)
    record, report = extract_record(base, llm, SETTINGS)
    return base, record, report, llm


def events_of(record, type_):
    return [e for e in record.events if e.type == type_]


def test_each_chunk_is_sent_at_most_once(extracted):
    base, _, report, llm = extracted
    assert report.documents == len(base.documents)
    assert report.model_calls == len(llm.calls) <= report.chunks


def test_the_prompt_lists_only_sentences_with_written_dates(extracted):
    _, _, _, llm = extracted
    indictment_call = next(call for call in llm.calls if "Title: Indictment" in call.user)
    numbered = [line for line in indictment_call.user.splitlines() if re.match(r"\[\d+\] ", line)]
    assert numbered and all(re.search(r"\d{4}", line) for line in numbered)
    assert "FULL TEXT" not in indictment_call.user  # arguments come from notes only


def test_two_dates_in_one_sentence_give_two_events(extracted):
    _, record, _, _ = extracted
    extensions = sorted(e.parsed_date.date() for e in events_of(record, "detention_extension"))
    assert extensions == [dt.date(2025, 3, 19), dt.date(2025, 5, 14)]


def test_announcement_without_an_argument_verb_is_not_an_argument(extracted):
    _, record, report, _ = extracted
    assert not any("appeal" in a.text for a in record.arguments)
    assert any("no argument verb" in reason for reason in report.dropped)


def test_exact_quote_becomes_a_sentence_span_with_a_code_parsed_date(extracted):
    _, record, _, _ = extracted
    (arrest,) = events_of(record, "arrest")
    assert arrest.span.text.startswith("The accused was arrested at his home")
    assert arrest.quote_span.text == "The accused was arrested at his home in Mirevo in the early morning of 14 February 2025"
    assert arrest.date_span is not None and arrest.date_span.text == "14 February 2025"
    assert arrest.parsed_date == dt.datetime(2025, 2, 14)
    assert arrest.precision == "date"
    assert not arrest.needs_review


def test_model_date_that_disagrees_with_the_text_is_kept_and_marked_for_review(extracted):
    _, record, _, _ = extracted
    (appearance,) = events_of(record, "first_appearance")
    assert appearance.parsed_date == dt.datetime(2025, 2, 19)  # the Clock uses this
    assert appearance.model_date == "2025-02-18"  # kept for the reviewer
    assert appearance.needs_review
    assert any("disagree" in reason for reason in appearance.review_reasons)


def test_quote_not_in_the_source_is_dropped_and_counted(extracted):
    _, record, report, _ = extracted
    assert events_of(record, "charge") == []
    assert any("not found" in reason for reason in report.dropped)


def test_arguments_are_kept_only_from_monitoring_notes(extracted):
    _, record, report, _ = extracted
    assert [a.party for a in record.arguments] == ["defense"]
    argument = record.arguments[0]
    assert argument.text == argument.span.text
    assert argument.hearing_date == dt.date(2025, 6, 16)
    assert any("argument" in reason and "note" in reason for reason in report.dropped)


def test_deterministic_parts_of_the_record_are_unchanged(extracted):
    base, record, _, _ = extracted
    assert record.observations == base.observations
    assert record.passages == base.passages
    assert record.rulings == base.rulings
    assert [e for e in record.events if e.type == "hearing"] == list(base.events)


def test_progress_callback_reports_each_chunk():
    base = load_case(DEMO_CASE_DIR)
    seen = []
    extract_record(base, FakeLLM(responder), SETTINGS, progress=lambda label, done, total: seen.append((done, total)))
    assert seen[-1][0] == seen[-1][1] == len(seen)
