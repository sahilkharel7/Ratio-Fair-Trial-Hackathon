"""LLM extraction with a scripted model: quotes become source spans, dates are parsed in code."""

import datetime as dt
import re

import pytest

from ratio.config import load_config
from ratio.extraction.build import load_case
from ratio.extraction.extract import extract_record
from ratio.extraction.prompts import ChunkExtraction
from ratio.paths import DEMO_CASE_DIR
from ratio.testing import FakeLLM, make_record

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
    base, record, _, _ = extracted
    from_model = [e for e in events_of(record, "detention_extension") if e not in base.events]  # the orders' own are built in code
    extensions = sorted(e.parsed_date.date() for e in from_model)
    assert extensions == [dt.date(2025, 3, 19), dt.date(2025, 5, 14)]


def test_announcement_without_an_argument_verb_is_not_an_argument(extracted):
    _, record, report, _ = extracted
    assert not any("appeal" in a.text for a in record.arguments)
    assert any("no argument verb" in reason for reason in report.dropped)


CLOSING = (
    "Defence counsel argued two points. First, four of the six posts were written by other people. "
    "The defendant cannot be convicted for them. Second, the two posts he wrote were opinions.\n\n"
    "First, the judge thanked the parties.\n\n"
    "Defence counsel objected to the exhibit. First, the clerk read out the exhibit list.\n\n"
    'The prosecutor argued three points. "First, the posts were false," she said.\n\n'
    "Defence counsel argued that the exhibit was late. The judge gave two reasons. First, the exhibit was filed in time."
)


def note_arguments(body: str, *quotes: str):
    """The arguments kept from what the model named (``quotes``, as defence arguments) in one monitoring
    note; sentences the backstop adds on its own are tested separately below."""
    record = make_record([("note.txt", "monitoring_note", f"Hearing date: 1 July 2026\n\n{body}")])

    def respond(system, user, schema, purpose):
        return {"events": [], "arguments": [{"party": "defense", "quote": quote} for quote in quotes]}

    extracted, report = extract_record(record, FakeLLM(respond), SETTINGS)
    return sorted(a.text for a in extracted.arguments if any(quote in a.text for quote in quotes)), report


def test_numbered_points_after_an_argument_verb_in_the_same_paragraph_are_arguments():
    texts, _ = note_arguments(CLOSING, "First, four of the six posts were written by other people", "Second, the two posts he wrote were opinions")
    assert texts == ["First, four of the six posts were written by other people.", "Second, the two posts he wrote were opinions."]


def test_a_numbered_point_in_a_paragraph_without_an_argument_verb_is_dropped():
    texts, report = note_arguments(CLOSING, "First, the judge thanked the parties")
    assert texts == []
    assert any("no argument verb" in reason for reason in report.dropped)


def test_a_numbered_point_needs_a_list_announced_with_the_argument_verb():
    # "objected to the exhibit" announces no points, so this "First," does not start an argument list.
    texts, _ = note_arguments(CLOSING, "First, the clerk read out the exhibit list")
    assert texts == []


def test_the_argument_verb_and_the_announced_list_must_be_in_the_same_sentence():
    # The judge, not counsel, announced these reasons.
    texts, _ = note_arguments(CLOSING, "First, the exhibit was filed in time")
    assert texts == []


def test_a_numbered_point_inside_quotation_marks_is_recognised():
    texts, _ = note_arguments(CLOSING, "First, the posts were false")
    assert texts == ['"First, the posts were false," she said.']


BACKSTOP_NOTE = (
    "Her lawyer, Mara Quint, was present. Prosecutor Edda Marn read the charge.\n\n"
    "In closing, Ms. Quint argued that the posts were true and that there was no time to answer the amended charge.\n\n"
    "Ms. Marn argued that the posts caused panic.\n\n"
    "Defence counsel argued two points.\n\n"
    "The witness argued that he had seen nothing."
)


def note_record_arguments(body: str, *quotes: str):
    record = make_record([("note.txt", "monitoring_note", f"Hearing date: 22 September 2025\n\n{body}")])

    def respond(system, user, schema, purpose):
        return {"events": [], "arguments": [{"party": "defense", "quote": quote} for quote in quotes]}

    extracted, _ = extract_record(record, FakeLLM(respond), SETTINGS)
    return extracted.arguments


def test_an_argument_the_model_missed_is_added_with_its_party_and_marked_for_review():
    arguments = note_record_arguments(BACKSTOP_NOTE)  # the model returns no arguments at all
    assert [(a.party, a.text[:28], a.needs_review) for a in arguments] == [
        ("defense", "In closing, Ms. Quint argued", True),  # Quint is "her lawyer" in the notes
        ("prosecution", "Ms. Marn argued that the pos", True),  # Marn is the prosecutor
    ]
    assert all(a.span.text == a.text for a in arguments)


def test_the_backstop_never_repeats_an_argument_the_model_found():
    arguments = note_record_arguments(BACKSTOP_NOTE, "In closing, Ms. Quint argued that the posts were true")
    quint = [a for a in arguments if "Quint argued" in a.text]
    assert len(quint) == 1 and quint[0].quote_span.text == "In closing, Ms. Quint argued that the posts were true"  # the model's own


def test_an_unnumbered_sentence_still_needs_its_own_argument_verb():
    texts, _ = note_arguments(CLOSING, "The defendant cannot be convicted for them")
    assert texts == []


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
    assert {record.document(a.span.doc_id).type for a in record.arguments} == {"monitoring_note"}
    argument = next(a for a in record.arguments if "extracted from Mr. Venn" in a.text)  # the one the model named
    assert argument.text == argument.span.text
    assert argument.hearing_date == dt.date(2025, 6, 16)
    assert any("argument" in reason and "note" in reason for reason in report.dropped)


def test_deterministic_parts_of_the_record_are_unchanged(extracted):
    base, record, _, _ = extracted
    assert record.observations == base.observations
    assert record.passages == base.passages
    assert record.rulings == base.rulings
    assert record.events[: len(base.events)] == base.events  # header events (hearings, the verdict) and order extensions, built in code, come first


def test_detention_orders_are_read_in_code_and_never_sent_to_the_model(extracted):
    base, record, _, llm = extracted
    assert not any("Detention" in call.user and "order" in call.user.lower() for call in llm.calls)
    assert record.orders == base.orders and len(base.orders) == 3
    in_code = [e for e in base.events if e.type == "detention_extension"]
    assert sorted(e.parsed_date.date() for e in in_code) == [dt.date(2025, 3, 19), dt.date(2025, 5, 14)]


def test_progress_callback_reports_each_chunk():
    base = load_case(DEMO_CASE_DIR)
    seen = []
    extract_record(base, FakeLLM(responder), SETTINGS, progress=lambda label, done, total: seen.append((done, total)))
    assert seen[-1][0] == seen[-1][1] == len(seen)
