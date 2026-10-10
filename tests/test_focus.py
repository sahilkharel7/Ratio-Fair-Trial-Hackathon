import pytest

from ratio.extraction.build import load_case
from ratio.focus import penalty_summary, screen
from ratio.paths import DEMO_CASE_DIR, DEMO_DIR
from ratio.testing import make_record


def check_spans(record, focus):
    spans = (
        [p.span for p in focus.penalties]
        + [p.span for p in focus.prompts]
        + ([focus.charge_span] if focus.charge_span else [])
    )
    for span in spans:
        assert record.document(span.doc_id).text[span.start : span.end] == span.text


def test_actual_demo_keeps_prosecution_request_range_and_sentence_separate():
    record = load_case(DEMO_CASE_DIR)
    focus = screen(record)
    assert penalty_summary(focus, "requested")["label"] == "5 years imprisonment"
    assert penalty_summary(focus, "imposed")["label"] == "4 years imprisonment"
    assert penalty_summary(focus, "statutory")["label"] == "2–6 years imprisonment"
    assert not focus.prompts  # accusation, copied reasoning and a conviction are not themselves proof of Article 14(2)
    check_spans(record, focus)


def test_focused_demo_and_normal_burden_statement_are_not_confused():
    sorin = load_case(DEMO_DIR / "focused" / "sorin-2025")
    maren = load_case(DEMO_DIR / "focused" / "maren-2025")
    found = screen(sorin)
    assert [p.pattern for p in found.prompts] == ["burden_shift"]
    assert found.prompts[0].status == "needs_review"
    assert penalty_summary(found, "statutory")["label"] == "Up to 5 years imprisonment"
    assert not screen(maren).prompts
    check_spans(sorin, found)


@pytest.mark.parametrize(
    "term,months",
    [
        ("twenty-five", 300),
        ("one hundred and twenty-five", 1500),
        ("six", 72),
        ("2.5", 30),
    ],
)
def test_compound_numbers_preserve_the_actual_requested_term(term, months):
    record = make_record(
        [
            (
                "indictment.txt",
                "indictment",
                f"The prosecutor requests {term} years imprisonment.",
            )
        ]
    )
    focus = screen(record)
    assert focus.penalties[0].max_months == months
    check_spans(record, focus)


def test_conflicting_specific_requests_require_source_review():
    focus = screen(
        make_record(
            [
                (
                    "note.txt",
                    "monitoring_note",
                    "The prosecutor asks for three years imprisonment. The prosecutor later requests five years imprisonment.",
                )
            ]
        )
    )
    summary = penalty_summary(focus, "requested")
    assert summary["ambiguous"] and summary["statement"] is None


@pytest.mark.parametrize(
    "text",
    [
        "The defendant is not required to prove innocence.",
        "The court said it would be wrong to require the accused to prove innocence.",
        "The accused is charged with an offence and the prosecutor requests a conviction.",
        "The Court finds the defendant guilty after hearing the evidence.",
    ],
)
def test_normal_or_negated_statements_do_not_become_violation_prompts(text):
    assert not screen(make_record([("judgment.txt", "judgment", text)])).prompts


def test_pretrial_public_guilt_statement_is_only_a_source_prompt():
    record = make_record(
        [
            (
                "note.txt",
                "monitoring_note",
                "Before the trial the police spokesman publicly announced on television that the accused was guilty.",
            )
        ]
    )
    focus = screen(record)
    assert focus.prompts[0].pattern == "official_guilt_statement"
    assert focus.prompts[0].status == "needs_review"
    check_spans(record, focus)


def test_probation_is_not_fabricated_as_a_fine_and_negative_terms_are_not_positive():
    focus = screen(
        make_record(
            [
                (
                    "indictment.txt",
                    "indictment",
                    "The prosecutor requests a sentence of probation. The prosecutor did not request five years imprisonment.",
                )
            ]
        )
    )
    assert len(focus.penalties) == 1 and focus.penalties[0].kind == "other"


@pytest.mark.parametrize(
    "term,label",
    [
        ("five years and six months", "5 years 6 months imprisonment"),
        ("five and a half years", "5.5 years imprisonment"),
    ],
)
def test_pdf_line_wrapping_preserves_a_complete_imposed_term(term, label):
    record = make_record(
        [
            (
                "judgment.txt",
                "judgment",
                f"The applicant was sentenced to {term}\n imprisonment.",
            )
        ]
    )
    focus = screen(record)
    assert focus.penalties[0].label == label
    assert focus.penalties[0].min_months == focus.penalties[0].max_months == 66
    check_spans(record, focus)


def test_review_judgment_discussion_is_a_reading_lead_even_when_no_breach_found():
    record = make_record(
        [
            (
                "judgment.txt",
                "judgment",
                "The Court examined the presumption of\ninnocence and found no violation.",
            )
        ]
    )
    focus = screen(record)
    assert focus.prompts[0].pattern == "presumption_discussion"
    assert focus.prompts[0].status == "needs_review"
    assert "allegation" in focus.prompts[0].question
    check_spans(record, focus)


def test_statutory_liability_in_a_review_judgment_is_not_an_imposed_sentence():
    record = make_record(
        [
            (
                "judgment.txt",
                "judgment",
                "Simple defamation entails liability to a sentence of imprisonment for not more than six months.",
            )
        ]
    )
    focus = screen(record)
    assert not [p for p in focus.penalties if p.stage == "imposed"]
    assert penalty_summary(focus, "statutory")["label"] == "Up to 6 months imprisonment"
    check_spans(record, focus)
