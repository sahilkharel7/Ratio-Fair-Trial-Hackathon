"""Reuse Detector, step 1: which judgment passages are the court's own reasoning."""

import pytest

from ratio.config import load_config
from ratio.modules.reuse_exclusions import classify, compile_rules
from ratio.paths import GOLD_DIR
from ratio.schema import CaseRecord
from ratio.testing import make_record

CONFIG = load_config()
RULES = compile_rules(CONFIG.settings.reuse)
MOCK = CaseRecord.model_validate_json((GOLD_DIR / "mock_record.json").read_text(encoding="utf-8"))
CAPTION = "JUDGMENT\nCase no. T-1\n\n"


def reasons_of(record: CaseRecord, doc_type: str = "judgment") -> dict[str, str | None]:
    """Sentence text -> exclusion reason (None for reasoning), headings omitted."""
    doc = record.documents_of_type(doc_type)[0]
    passages = record.passages_of(doc.id)
    reasons = classify(passages, doc.text, record.citations, RULES)
    return {p.span.text: reasons[p.id] for p in passages if p.id in reasons}


def judgment(body: str) -> dict[str, str | None]:
    return reasons_of(make_record([("judgment.txt", "judgment", CAPTION + body)]))


def test_demo_reasoning_is_exactly_the_courts_assessment():
    doc = MOCK.documents_of_type("judgment")[0]
    passages = MOCK.passages_of(doc.id)
    reasons = classify(passages, doc.text, MOCK.citations, RULES)
    by_index = {p.index: reasons.get(p.id, "heading") for p in passages}
    assert [index for index, reason in by_index.items() if reason is None] == list(range(21, 31))
    assert by_index[13] == "charge_recital"
    assert by_index[19] == "statute_quote"
    assert {by_index[i] for i in (15, 16, 17)} == {"party_position"}
    assert {by_index[i] for i in (8, 9, 10, 11, 32, 33)} == {"non_reasoning_section"}
    assert {by_index[i] for i in (0, 1, 2, 3, 4, 5, 6, 34)} == {"header_or_signature"}
    assert {by_index[i] for i in (7, 12, 14, 18, 20, 31)} == {"heading"}


def test_party_position_is_carried_through_its_paragraph_until_the_court_speaks():
    reasons = judgment(
        "V. ASSESSMENT\n\n"
        "The prosecution argues that the accused planned the scheme for months. "
        "It further submits that the transfers were concealed. "
        "The Court finds that the transfers were recorded openly in the ledger.\n\n"
        "The defence contends that the ledger was altered. In its view, the ledger cannot be trusted.\n\n"
        "The ledger was signed by the accused on 3 May 2024.\n"
    )
    assert list(reasons.values())[2:] == [
        "party_position",
        "party_position",  # carried: "It further submits" names no party
        None,  # the court's own voice ends the carry
        "party_position",
        "party_position",
        None,  # a new paragraph ends it too
    ]


def test_heading_ends_a_carried_position():
    reasons = judgment(
        "IV. THE CASE FOR THE PROSECUTION\n\nThe prosecutor argued that the funds were moved abroad.\n\n"
        "V. ASSESSMENT\n\nThe funds were moved abroad on 4 May 2024.\n"
    )
    assert reasons["The prosecutor argued that the funds were moved abroad."] == "party_position"
    assert reasons["The funds were moved abroad on 4 May 2024."] is None


@pytest.mark.parametrize(
    "sentence",
    [
        "The defence rightly argues that the warrant was never served.",
        "The Court agrees with the prosecution that the funds were concealed.",
        "This Court finds, as the prosecution argued, that the funds were concealed.",
    ],
)
def test_a_position_the_court_endorses_is_reasoning(sentence):
    assert judgment(f"V. ASSESSMENT\n\n{sentence}\n")[sentence] is None


@pytest.mark.parametrize(
    "sentence",
    [
        "According to the prosecution, the accused fled abroad.",
        "Defence counsel submitted that the confession was coerced.",
        "Counsel for the accused maintained that no transfer took place.",
    ],
)
def test_attribution_forms(sentence):
    assert judgment(f"V. ASSESSMENT\n\n{sentence}\n")[sentence] == "party_position"


def test_statute_marker_needs_a_quotation_or_a_citation():
    reasons = judgment(
        "V. ASSESSMENT\n\n"
        "Section 12 of the Criminal Code provides that no person may be tried twice for one offence. "
        "The ledger provides that the transfer was approved by the board.\n"
    )
    assert reasons["Section 12 of the Criminal Code provides that no person may be tried twice for one offence."] == "statute_quote"
    assert reasons["The ledger provides that the transfer was approved by the board."] is None


def test_charge_recital_and_non_reasoning_sections():
    reasons = judgment(
        "II. SUBMISSIONS OF THE PARTIES\n\nBoth parties filed written submissions on 2 May 2024.\n\n"
        "VII. OBSERVATIONS ON THE EVIDENCE\n\nThe accused stands charged with fraud under the Criminal Code. "
        "The witness gave a consistent account of the meeting.\n"
    )
    assert reasons["Both parties filed written submissions on 2 May 2024."] == "non_reasoning_section"
    assert reasons["The accused stands charged with fraud under the Criminal Code."] == "charge_recital"
    assert reasons["The witness gave a consistent account of the meeting."] is None  # unknown heading: reasoning


def test_a_judgment_without_headings_is_all_reasoning_except_caption_and_signature():
    reasons = judgment("The witness was credible. The accused was present.\n\n(signed) A. Judge\n")
    assert reasons == {
        "JUDGMENT": "header_or_signature",
        "Case no. T-1": "header_or_signature",
        "The witness was credible.": None,
        "The accused was present.": None,
        "(signed) A. Judge": "header_or_signature",
    }


# --- review fixes: real-world wording of court voice, party positions, recitals and headings --------

ASSESSMENT = "V. ASSESSMENT OF THE COURT\n\n"


@pytest.mark.parametrize(
    "rebuttal",
    [
        "That argument is rejected: the amendment entered into force on 1 January 2025.",
        "This objection cannot succeed.",
        "The Court does not accept this.",
        "The Court therefore finds that the amendment applied.",
        "The Trial Chamber finds that the amendment applied.",
        "The Court, having examined the file, finds that the amendment applied.",
        "We find that the amendment applied.",
        "However, the amendment entered into force on 1 January 2025.",
    ],
)
def test_the_courts_answer_after_a_party_position_is_reasoning(rebuttal):
    reasons = judgment(ASSESSMENT + f"The defence argues that the first article predates the amendment. {rebuttal}\n")
    assert reasons["The defence argues that the first article predates the amendment."] == "party_position"
    assert reasons[rebuttal] is None


def test_a_report_continues_across_sentences_and_paragraphs_until_the_court_speaks():
    reasons = judgment(
        ASSESSMENT + "The prosecution argues that the accused acted alone.\n\n"
        "It further submits that the messages were sent abroad. In its view, the timing was planned.\n\n"
        "The messages were sent from the accused's telephone.\n"
    )
    assert list(reasons.values())[2:] == ["party_position", "party_position", "party_position", None]


@pytest.mark.parametrize(
    "sentence",
    [
        "Contrary to what the defence argues, the accused knew that the allegations were false.",
        "Although the defence argues that the search was unlawful, the warrant is in the file.",
        "The defence claim that the warrant was never issued is unfounded.",
        "The defence request to exclude the messages is rejected.",
        "As to the defence claim that the search was unlawful, the warrant is in the file.",
        "The defence witness claimed that he saw nothing.",
        "The witnesses correctly identified the accused.",
    ],
)
def test_the_court_mentioning_a_party_is_not_a_party_position(sentence):
    assert judgment(ASSESSMENT + sentence + "\n")[sentence] is None


@pytest.mark.parametrize(
    "sentence",
    [
        "The prosecution, in its closing speech, argued that the accused knew the allegations were false.",
        "It was argued by the prosecution that the accused knew the allegations were false.",
        "In the prosecution's view, the accused knew the allegations were false.",
        "The prosecution's case is that the accused knew the allegations were false.",
        "The prosecution states that the accused knew the allegations were false.",
        "The Public Prosecutor maintained that the accused knew the allegations were false.",
        "The Court notes that the prosecution argues that the accused knew the allegations were false.",
        "The prosecution submits that the investigators correctly seized the telephone.",
        "The prosecution accepts that the telephone was seized without a warrant.",
    ],
)
def test_ways_of_reporting_a_partys_case_are_party_positions(sentence):
    assert judgment(ASSESSMENT + sentence + "\n")[sentence] == "party_position"


@pytest.mark.parametrize(
    "heading",
    [
        "V. FINDINGS ON THE CHARGES",
        "V. ASSESSMENT OF THE CHARGE",
        "V. THE COURT'S ASSESSMENT OF THE PARTIES' SUBMISSIONS",
        "V. ASSESSMENT OF THE ARGUMENTS OF THE PARTIES",
        "V. SUBMISSIONS OF THE PARTIES AND ASSESSMENT OF THE COURT",
    ],
)
def test_reasoning_chapters_named_after_the_charge_or_the_submissions_stay_reasoning(heading):
    sentence = "The accused knew that the allegations were false."
    assert judgment(f"{heading}\n\n{sentence}\n")[sentence] is None


@pytest.mark.parametrize("heading", ["VI. VERDICT", "VI. DECISION", "ON THESE GROUNDS", "FOR THESE REASONS, THE COURT, UNANIMOUSLY,", "III. THE PARTIES' ARGUMENTS"])
def test_operative_and_submission_chapters_are_not_reasoning(heading):
    sentence = "Finds the accused guilty of publishing the articles."
    assert judgment(f"{heading}\n\n{sentence}\n")[sentence] == "non_reasoning_section"


def test_a_sub_heading_inside_the_submissions_chapter_is_still_submissions():
    reasons = judgment(
        "III. SUBMISSIONS OF THE PARTIES\n\nA. The prosecution\n\nMessages recovered from the telephone show the plan.\n\n"
        "V. ASSESSMENT OF THE COURT\n\nA. The charge of disseminating false information\n\nThe witness was credible.\n"
    )
    assert reasons["Messages recovered from the telephone show the plan."] == "non_reasoning_section"
    assert reasons["The witness was credible."] is None


@pytest.mark.parametrize(
    "sentence, reason",
    [
        ("As to count 1, the accused knew that the allegations were false.", None),
        ("On Count 2, the messages show that he coordinated the timing.", None),
        ("The Court finds that the accused, who is charged with fraud, knew the allegations were false.", None),
        ("By indictment of 3 March 2025, the accused was charged with disseminating false information.", "charge_recital"),
        ("According to the indictment, the accused published a series of articles.", "charge_recital"),
    ],
)
def test_charge_recitals_and_count_by_count_reasoning(sentence, reason):
    assert judgment(ASSESSMENT + sentence + "\n")[sentence] == reason


def test_a_provision_quoted_as_a_block_after_its_marker_is_excluded_with_it():
    reasons = judgment(
        "IV. THE APPLICABLE LAW\n\nArticle 214(2) of the Penal Code provides:\n\n"
        "Whoever disseminates information that he knows to be false shall be punished. "
        "The same penalty applies where the dissemination causes public alarm.\n\n"
        "The accused published the articles knowing them to be false.\n"
    )
    assert list(reasons.values())[2:] == ["statute_quote", "statute_quote", "statute_quote", None]


@pytest.mark.parametrize("opening, closing", [("‘", "’"), ("«", "»"), ('"', '"')])
def test_a_multi_sentence_provision_in_any_quotation_style_is_one_statute_quote(opening, closing):
    provision = f"Rule 11 provides: {opening}An accused shall be informed. No exception applies.{closing}"
    assert judgment(f"IV. THE APPLICABLE LAW\n\n{provision}\n")[provision] == "statute_quote"


def test_line_breaks_inside_markers_change_nothing():
    reasons = judgment(ASSESSMENT + "The defence contends that the allegations were true. The Court\nfinds that the accused knew they were false.\n")
    assert reasons["The Court\nfinds that the accused knew they were false."] is None
    recital = "The accused is\ncharged with disseminating false information."
    assert judgment(ASSESSMENT + recital + "\n")[recital] == "charge_recital"
