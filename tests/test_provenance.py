"""Hard rule 2: a flag survives only if every one of its spans is found in the original text."""

from ratio.provenance import enforce, filter_follow_up, resolver_for, span_is_valid
from ratio.schema import CaseMeta, CaseRecord, Document, Evidence, Flag, FollowUp, SourceSpan

TEXT_A = "The defendant was present. No interpreter was mentioned."
TEXT_B = "Judge Varda ordered detention on 19 February 2025."


def record(case_id: str, text: str) -> CaseRecord:
    meta = CaseMeta(
        case_id=case_id,
        title="t",
        court="c",
        charge_type="x",
        data_provenance="synthetic",
        synthetic=True,
    )
    doc = Document(
        id=f"{case_id}/a.txt",
        case_id=case_id,
        path="a.txt",
        type="monitoring_note",
        title="a",
        text=text,
        synthetic=True,
        sha256="0" * 64,
    )
    return CaseRecord(meta=meta, documents=(doc,))


def span(doc_id: str, text: str, start: int, end: int, override: str | None = None) -> SourceSpan:
    return SourceSpan(doc_id=doc_id, start=start, end=end, text=override if override is not None else text[start:end])


def flag(flag_id: str, *spans: SourceSpan) -> Flag:
    return Flag(
        id=flag_id,
        case_id="c1",
        module="absence",
        standard_id="iccpr_14_3_d",
        standard_label="ICCPR Art. 14(3)(d)",
        status="evidence_of_compliance",
        message="m",
        evidence=tuple(Evidence(role="supporting", span=s) for s in spans),
    )


RESOLVE = resolver_for([record("c1", TEXT_A), record("c2", TEXT_B)])
GOOD = span("c1/a.txt", TEXT_A, 0, 26)
OTHER_CASE = span("c2/a.txt", TEXT_B, 0, 11)


def test_valid_span_in_any_case_resolves():
    assert span_is_valid(GOOD, RESOLVE)
    assert span_is_valid(OTHER_CASE, RESOLVE)


def test_span_with_altered_text_is_invalid():
    tampered = span("c1/a.txt", TEXT_A, 0, 26, override="The defendant was ABSENT.!")
    assert not span_is_valid(tampered, RESOLVE)


def test_span_out_of_range_or_unknown_document_is_invalid():
    beyond = SourceSpan(doc_id="c1/a.txt", start=50, end=90, text="x" * 40)
    unknown = SourceSpan(doc_id="c9/a.txt", start=0, end=3, text="The")
    assert not span_is_valid(beyond, RESOLVE)
    assert not span_is_valid(unknown, RESOLVE)


def test_enforce_keeps_valid_flags_and_drops_any_flag_with_a_bad_span():
    tampered = span("c1/a.txt", TEXT_A, 27, 56, override="An interpreter was mentioned.")
    result = enforce([flag("ok", GOOD, OTHER_CASE), flag("bad", GOOD, tampered)], RESOLVE)
    assert [f.id for f in result.kept] == ["ok"]
    assert [f.id for f, _ in result.dropped] == ["bad"]
    assert "c1/a.txt" in result.dropped[0][1]
    assert result.rate == 0.5


def test_enforce_on_no_flags_reports_full_rate():
    assert enforce([], RESOLVE).rate == 1.0


def test_follow_up_keeps_only_valid_context_spans():
    bad = span("c1/a.txt", TEXT_A, 0, 26, override="x" * 26)
    follow_up = FollowUp(
        id="u",
        case_id="c1",
        rubric_id="iccpr_14_3_f",
        question="q",
        context=(Evidence(role="context", span=GOOD), Evidence(role="context", span=bad)),
    )
    cleaned = filter_follow_up(follow_up, RESOLVE)
    assert [e.span for e in cleaned.context] == [GOOD]
    assert follow_up.context[1].span == bad  # the original is not modified


def test_a_flag_without_evidence_is_dropped_even_if_validation_was_bypassed():
    hollow = Flag.model_construct(**{**flag("hollow", GOOD).model_dump(), "evidence": ()})
    result = enforce([hollow], RESOLVE)
    assert result.kept == () and result.dropped[0][1] == "flag has no evidence span"
