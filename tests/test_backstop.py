"""The argument backstop: a note sentence becomes an argument only when a party is the whole subject of
"argued that ..." (or a like verb). The rejected sentences are the ones review found misread."""

from ratio.extraction.backstop import counsel_roster, missed_arguments
from ratio.testing import make_record

ROSTER = (
    "Her lawyer, Mara Quint, was present. Prosecutor Edda Marn read the charge. The victim's lawyer, Ann Pell, sat behind the prosecution. "
    "Prosecution counsel Ivo Holt joined her. Defence counsel Ada Lune and the prosecutor, Teo Lune, were both new."
)
ARGUMENTS = [
    ("defense", "In closing, Ms. Quint argued that the posts were true."),
    ("prosecution", "The prosecutor argued that the posts caused panic."),
    ("defense", "Defense counsel objected that the statement was signed without a lawyer."),
    ("prosecution", "Ms. Marn contended that the screenshots were reliable."),
    ("prosecution", "Mr. Holt submitted that the exhibit was admissible."),  # "prosecution counsel Ivo Holt"
]
NOT_ARGUMENTS = [
    "The defence witness insisted that he had been at home that night.",
    "The defence's motion was challenged by the prosecutor.",
    "After the defence rested, the presiding judge challenged the witness.",
    "The defence left the screenshots unchallenged.",
    "Defence counsel never argued that the posts were true.",
    "The presiding judge remarked that the defence had argued nothing new.",
    "Defence counsel objected to the question and the judge overruled the objection.",
    "According to her counsel, Inspector Dorn, the search was lawful. Inspector Dorn maintained that the search was lawful.",
    "Ms. Pell argued that the posts harmed her client.",  # the victim's lawyer is not a party
    "Ms. Marn, for the State, argued two points.",
    "Defence counsel argued two points.",
    "Ms. Lune argued that the hearing should go on.",  # a surname named for both sides
]


def note(*paragraphs: str):
    return make_record([("note.txt", "monitoring_note", "Hearing date: 22 September 2025\n\n" + "\n\n".join(paragraphs))])


def test_named_counsel_are_read_from_the_notes_by_side():
    assert counsel_roster(note(ROSTER)) == {"Quint": "defense", "Marn": "prosecution", "Holt": "prosecution"}


def test_a_name_never_runs_across_a_line_and_a_title_is_never_a_name():
    roster = counsel_roster(note("Closing submissions of defence counsel\nMs. Quint spoke first.", "According to her counsel, Inspector Dorn, nothing was seized."))
    assert roster == {}


def test_only_a_party_that_is_the_whole_subject_of_argued_that_makes_an_argument():
    record = note(ROSTER, *(text for _, text in ARGUMENTS), *NOT_ARGUMENTS)
    added = missed_arguments(record, [])
    assert [(a.party, a.text) for a in added] == ARGUMENTS
    assert all(a.needs_review and a.span.text == a.text and a.quote_span == a.span for a in added)


def test_the_backstop_reads_only_the_documents_of_the_run():
    record = note(ROSTER, *(text for _, text in ARGUMENTS))
    assert missed_arguments(record.model_copy(update={"documents": ()}), []) == []
