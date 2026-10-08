"""Contract tests for the case record and flag models."""

from datetime import date

import pytest
from pydantic import ValidationError

from ratio.schema import (
    CaseMeta,
    CaseRecord,
    Document,
    Evidence,
    Flag,
    FollowUp,
    SourceSpan,
    stable_id,
)

TEXT = "Hearing date: 2 June 2025\n\nMr. Venn was present in the courtroom."


def make_doc(text: str = TEXT, doc_path: str = "notes/hearing_1.txt") -> Document:
    return Document(
        id=f"case-1/{doc_path}",
        case_id="case-1",
        path=doc_path,
        type="monitoring_note",
        title="Hearing 1",
        text=text,
        synthetic=True,
        sha256="0" * 64,
        date=date(2025, 6, 2),
    )


def make_span(start: int, end: int, text: str = TEXT, doc_id: str = "case-1/notes/hearing_1.txt") -> SourceSpan:
    return SourceSpan(doc_id=doc_id, start=start, end=end, text=text[start:end])


def make_meta() -> CaseMeta:
    return CaseMeta(
        case_id="case-1",
        title="Republic v. Test",
        court="Test Court",
        charge_type="Test charge",
        data_provenance="synthetic",
        synthetic=True,
    )


class TestSourceSpan:
    def test_valid_span_keeps_offsets_and_text(self):
        span = make_span(27, 35)
        assert span.text == "Mr. Venn"
        assert span.case_id == "case-1"

    def test_rejects_text_length_mismatch(self):
        with pytest.raises(ValidationError):
            SourceSpan(doc_id="case-1/a.txt", start=0, end=5, text="abc")

    def test_rejects_empty_or_reversed_span(self):
        with pytest.raises(ValidationError):
            SourceSpan(doc_id="case-1/a.txt", start=5, end=5, text="")
        with pytest.raises(ValidationError):
            SourceSpan(doc_id="case-1/a.txt", start=6, end=5, text="x")

    def test_rejects_blank_text(self):
        with pytest.raises(ValidationError):
            SourceSpan(doc_id="case-1/a.txt", start=0, end=3, text="   ")

    def test_is_immutable(self):
        span = make_span(27, 35)
        with pytest.raises(ValidationError):
            span.start = 0  # type: ignore[misc]

    def test_overlap_requires_same_document(self):
        a = make_span(27, 35)
        b = make_span(30, 40)
        other_doc = SourceSpan(doc_id="case-1/other.txt", start=30, end=40, text=TEXT[30:40])
        assert a.overlaps(b)
        assert not a.overlaps(other_doc)
        assert not a.overlaps(make_span(35, 40))


class TestDocument:
    def test_id_must_combine_case_and_path(self):
        with pytest.raises(ValidationError):
            Document(
                id="wrong",
                case_id="case-1",
                path="a.txt",
                type="judgment",
                title="t",
                text="x",
                synthetic=True,
                sha256="0" * 64,
            )

    def test_rejects_unknown_document_type(self):
        with pytest.raises(ValidationError):
            Document(
                id="case-1/a.txt",
                case_id="case-1",
                path="a.txt",
                type="email",  # type: ignore[arg-type]
                title="t",
                text="x",
                synthetic=True,
                sha256="0" * 64,
            )


class TestFlag:
    def evidence(self) -> tuple[Evidence, ...]:
        return (Evidence(role="supporting", span=make_span(27, 35)),)

    def test_flag_requires_at_least_one_evidence_span(self):
        with pytest.raises(ValidationError):
            Flag(
                id="f1",
                case_id="case-1",
                module="absence",
                standard_id="iccpr_14_3_d",
                standard_label="ICCPR Art. 14(3)(d)",
                status="evidence_of_compliance",
                message="m",
                evidence=(),
            )

    def test_status_must_belong_to_module_vocabulary(self):
        with pytest.raises(ValidationError):
            Flag(
                id="f1",
                case_id="case-1",
                module="clock",
                standard_id="gc35_48h",
                standard_label="GC35",
                status="evidence_of_violation",
                message="m",
                evidence=self.evidence(),
            )

    def test_valid_flag_exposes_spans(self):
        flag = Flag(
            id="f1",
            case_id="case-1",
            module="absence",
            standard_id="iccpr_14_3_d",
            standard_label="ICCPR Art. 14(3)(d)",
            status="evidence_of_compliance",
            message="m",
            evidence=self.evidence(),
        )
        assert flag.spans[0].text == "Mr. Venn"
        assert flag.review_status == "needs_legal_review"

    def test_follow_up_is_not_a_flag_and_needs_no_span(self):
        follow_up = FollowUp(id="u1", case_id="case-1", rubric_id="iccpr_14_3_f", question="Was an interpreter present?")
        assert follow_up.context == ()


class TestCaseRecord:
    def test_document_lookup(self):
        record = CaseRecord(meta=make_meta(), documents=(make_doc(),))
        assert record.document("case-1/notes/hearing_1.txt").title == "Hearing 1"
        with pytest.raises(KeyError):
            record.document("case-1/missing.txt")

    def test_rejects_duplicate_document_ids(self):
        with pytest.raises(ValidationError):
            CaseRecord(meta=make_meta(), documents=(make_doc(), make_doc()))

    def test_rejects_documents_from_another_case(self):
        stranger = Document(
            id="case-2/a.txt",
            case_id="case-2",
            path="a.txt",
            type="judgment",
            title="t",
            text="x",
            synthetic=True,
            sha256="0" * 64,
        )
        with pytest.raises(ValidationError):
            CaseRecord(meta=make_meta(), documents=(stranger,))

    def test_round_trips_through_json(self):
        record = CaseRecord(meta=make_meta(), documents=(make_doc(),))
        assert CaseRecord.model_validate_json(record.model_dump_json()) == record


def test_stable_id_is_deterministic_and_order_sensitive():
    assert stable_id("a", 1, "b") == stable_id("a", 1, "b")
    assert stable_id("a", 1, "b") != stable_id("b", 1, "a")
    assert len(stable_id("x")) == 16


def test_only_a_confirmed_benchmark_can_be_exceeded():
    with pytest.raises(ValidationError, match="hard rule 5"):
        Flag(
            id="f",
            case_id="case-1",
            module="clock",
            standard_id="counsel_access",
            standard_label="x",
            status="exceeds_benchmark",
            message="m",
            evidence=(Evidence(role="from_event", span=make_span(27, 35)),),
            review_status="needs_legal_review",
        )
