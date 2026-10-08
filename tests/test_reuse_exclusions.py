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
