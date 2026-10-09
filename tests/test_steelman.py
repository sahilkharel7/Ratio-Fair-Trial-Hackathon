"""The steelman-the-state agent: which findings get a reply, what the model is shown, and the checks
every argument must pass (a reviewed ground, a passage it was shown, an exact quote of that passage,
the block list) before it is kept. The real model's replies are checked on the recorded demo cache."""

import re

import pytest

from ratio import steelman
from ratio.config import load_config
from ratio.context import AnalysisContext
from ratio.llm import LLMResponseError
from ratio.messages import REMOVED
from ratio.provenance import resolver_for
from ratio.schema import Evidence, Flag, SourceSpan
from ratio.testing import FakeEmbedder, FakeLLM, make_record

CONFIG = load_config()
NOTE = (
    "Hearing date: 2 June 2025\n\n"
    "Defence counsel said she had received the case file three days before the hearing. "
    "She did not ask the court for an adjournment. "
    "The presiding judge ordered the trial to continue."
)
RECORD = make_record([("notes/one.txt", "monitoring_note", NOTE)])
DOC = RECORD.documents[0]


def span_of(text: str) -> SourceSpan:
    start = DOC.text.index(text)
    return SourceSpan(doc_id=DOC.id, start=start, end=start + len(text), text=text)


def flag(status="evidence_of_violation", module="absence", standard_id="iccpr_14_3_b", flag_id="f1") -> Flag:
    return Flag(
        id=flag_id, case_id=RECORD.case_id, module=module, standard_id=standard_id, standard_label="ICCPR Art. 14(3)(b)",
        status=status, message="The notes describe facts contrary to ICCPR Art. 14(3)(b).",
        review_status="confirmed" if status == "exceeds_benchmark" else "needs_legal_review",
        evidence=(Evidence(role="contradicting", span=span_of("Defence counsel said she had received the case file three days before the hearing.")),),
    )  # fmt: skip


def passages(user: str) -> dict[str, str]:
    return {pid: text for pid, text in re.findall(r"^(E\d+) \([^)]*\)(?: \[part of the finding\])?: (.*)$", user, re.MULTILINE)}


def answering(*arguments, rejects=()):
    """The State's arguments, then a check that accepts every quote except those containing ``rejects``."""

    def responder(system, user, schema, purpose):
        if purpose == "steelman_check":
            return {"note": "Checked the quote.", "shows": not any(text in user for text in rejects)}
        assert purpose == "steelman"
        return {"arguments": [dict(arg(passages(user))) if callable(arg) else arg for arg in arguments]}

    return responder


def no_request(shown: dict[str, str]) -> dict:
    pid = next(p for p, text in shown.items() if "adjournment" in text)
    return {"ground": "no_adjournment_request", "passage": pid, "quote": "She did not ask the court for an adjournment.", "argument": "Counsel never asked for more time, which she had to do."}


def run(responder, flags=None, record=RECORD):
    llm = FakeLLM(responder)
    result = steelman.run(record, flags or [flag()], AnalysisContext(llm=llm, embedder=FakeEmbedder(), config=CONFIG))
    return result, llm


# --- what the model is shown --------------------------------------------------------------


def test_the_prompt_offers_only_the_standards_grounds_and_numbered_record_passages():
    _, llm = run(answering())
    [call] = llm.calls  # no argument, so no check
    assert "- no_adjournment_request: " in call.user and "- record_incomplete: " in call.user
    assert "exceptional_circumstances" not in call.user  # a ground of another standard
    assert set(passages(call.user)) == {"E1", "E2", "E3"}  # the note's three sentences
    assert "E1 (notes/one.txt) [part of the finding]: Defence counsel said" in call.user  # the finding's own evidence is marked
    assert "cured_later: Any shortfall was made good at a later stage of the proceedings. Use only if a passage shows" in call.user
    assert "The passage must not be part of the finding." in call.user  # cured_later is independent
    assert 'The finding rests on:\n- "Defence counsel said she had received the case file' in call.user
    assert "Do not describe anyone's character or motives." in call.system


def test_compliance_findings_and_judge_patterns_get_no_reply():
    flags = [flag("evidence_of_compliance", flag_id="c"), flag("pattern_warrants_review", "judges", "judicial_pattern", "j"), flag(flag_id="v")]
    result, llm = run(answering(), flags)
    assert [r.flag_id for r in result.replies] == ["v"] and len(llm.calls) == 1


# --- the checks ---------------------------------------------------------------------------


def test_an_argument_on_an_offered_ground_with_an_exact_quote_is_kept():
    result, _ = run(answering(no_request))
    [reply] = result.replies
    [argument] = reply.arguments
    assert argument.ground_id == "no_adjournment_request" and argument.span == span_of("She did not ask the court for an adjournment.")
    assert reply.unsupported_grounds == ("time_adequate", "cured_later", "record_incomplete") and reply.dropped == ()


def test_unknown_grounds_unseen_passages_missing_quotes_and_repeats_are_dropped():
    bad = [
        {"ground": "the_court_was_right", "passage": "E1", "quote": "She did not ask the court for an adjournment.", "argument": "x"},
        {"ground": "time_adequate", "passage": "E9", "quote": "She did not ask the court for an adjournment.", "argument": "x"},
        {"ground": "cured_later", "passage": "E1", "quote": "The defence was later given two more weeks.", "argument": "x"},
    ]
    result, _ = run(answering(no_request, *bad, no_request))
    [reply] = result.replies
    assert [a.ground_id for a in reply.arguments] == ["no_adjournment_request"]
    reasons = " | ".join(reply.dropped)
    for expected in ("is not one of the grounds offered", "'E9' was not shown", "quote not found", "argued twice"):
        assert expected in reasons


def test_an_independent_ground_cannot_rest_on_the_findings_own_evidence():
    def circular(shown):
        pid = next(p for p, text in shown.items() if "three days" in text)
        return {"ground": "cured_later", "passage": pid, "quote": "she had received the case file three days before the hearing", "argument": "x"}

    result, _ = run(answering(circular))
    assert result.replies[0].arguments == () and "must rest on other evidence" in result.replies[0].dropped[0]


def test_an_argument_the_check_rejects_is_dropped_and_each_kept_one_was_checked():
    result, llm = run(answering(no_request, rejects=("did not ask",)))
    assert result.replies[0].arguments == () and "does not show what the ground requires" in result.replies[0].dropped[0]
    check = next(c for c in llm.calls if c.purpose == "steelman_check")
    assert 'Quote from the record: "She did not ask the court for an adjournment."' in check.user
    assert "the quote shows that the defence did not ask for an adjournment or more time" in check.user
    kept, llm = run(answering(no_request))
    assert len(kept.replies[0].arguments) == 1 and [c.purpose for c in llm.calls] == ["steelman", "steelman_check"]


def test_a_quote_differing_only_in_whitespace_or_quotation_marks_is_located_exactly():
    span = span_of("She did not ask the court for an adjournment.")
    located = steelman.locate("  “She did not   ask the court\nfor an adjournment.”", span)
    assert located == span
    assert steelman.locate("adjournment", span) is None  # too short to mean anything
    assert steelman.locate("She asked the court for an adjournment.", span) is None


def test_model_wording_that_characterises_a_person_is_removed():
    def biased(shown):
        return no_request(shown) | {"argument": "The defence lawyer was incompetent and never asked for time."}

    result, _ = run(answering(biased))
    assert REMOVED in result.replies[0].arguments[0].argument and "incompetent" not in result.replies[0].arguments[0].argument


def test_no_more_than_the_configured_number_of_arguments_is_kept():
    def on(ground):
        return lambda shown: no_request(shown) | {"ground": ground}

    result, _ = run(answering(*(on(g) for g in ("no_adjournment_request", "time_adequate", "cured_later", "record_incomplete"))))
    reply = result.replies[0]
    assert len(reply.arguments) == CONFIG.settings.steelman.max_arguments == 3
    assert any("beyond the first 3" in reason for reason in reply.dropped)


def test_an_unusable_model_answer_is_reported_not_raised():
    def broken(system, user, schema, purpose):
        raise LLMResponseError("truncated reply")

    result, _ = run(broken)
    [reply] = result.replies
    assert not reply.checked and "truncated reply" in reply.note and reply.arguments == ()


def test_an_argument_whose_quote_fails_the_source_check_is_dropped():
    result, _ = run(answering(no_request))
    forged = result.replies[0].arguments[0].model_copy(update={"span": SourceSpan(doc_id=DOC.id, start=0, end=5, text="FORGE")})
    tampered = result.model_copy(update={"replies": (result.replies[0].model_copy(update={"arguments": (forged,)}),)})
    checked = steelman.check_steelman(tampered, resolver_for([RECORD]))
    assert checked.replies[0].arguments == () and "failed the source check" in checked.replies[0].dropped[0]
    assert steelman.check_steelman(result, resolver_for([RECORD])) == result


def test_a_case_without_adverse_findings_makes_no_model_call():
    result, llm = run(answering(), [flag("evidence_of_compliance")])
    assert result.replies == () and llm.calls == ()


@pytest.mark.parametrize("module_status", [("clock", "exceeds_benchmark"), ("renewal", "repeated_grounds"), ("reuse", "unaddressed_argument")])
def test_every_adverse_status_is_contested(module_status):
    module, status = module_status
    standard = {"clock": "gc35_48h", "renewal": "renewal_review", "reuse": "unaddressed_defense_argument"}[module]
    result, _ = run(answering(), [flag(status, module, standard)])
    assert len(result.replies) == 1


# --- the catalogue ------------------------------------------------------------------------


def test_every_standard_ratio_flags_has_grounds_of_its_own_and_quotes_have_sources():
    standards = {item.id for item in CONFIG.rubric.items} | {"gc35_48h", "reasoning_reuse", "unaddressed_defense_argument", "renewal_review", "renewal_gap"}
    for standard_id in standards:
        assert CONFIG.steelman.grounds.get(standard_id), standard_id
    quoted = [g for grounds in CONFIG.steelman.grounds.values() for g in grounds if g.quote]
    assert quoted and all(g.source in CONFIG.jurisprudence.sources for g in quoted)
    assert "judicial_pattern" not in CONFIG.steelman.grounds  # never a reply for a judge (hard rule 3)


# --- the real model's recorded replies on the demo ----------------------------------------


@pytest.fixture(scope="module")
def demo():
    from ratio.extraction.build import load_case
    from ratio.paths import DEMO_CASE_DIR, MINILM_DIR
    from ratio.pipeline import analysis_context, analyze, demo_llm, ingest

    if not (MINILM_DIR / "modules.json").exists():
        pytest.skip("run scripts/fetch_models.py once")
    llm = demo_llm(CONFIG)
    record, _ = ingest(load_case(DEMO_CASE_DIR), llm, CONFIG)
    return record, analyze(record, analysis_context(CONFIG, llm))


@pytest.mark.embed
def test_the_recorded_replies_contest_every_adverse_finding_and_rest_on_exact_quotes(demo):
    record, analysis = demo
    adverse = [f for f in analysis.all_flags() if f.status in CONFIG.steelman.adverse_statuses]
    assert [r.flag_id for r in analysis.steelman.replies] == [f.id for f in adverse] and len(adverse) == 11
    resolve = resolver_for([record])
    from ratio.provenance import span_is_valid

    for reply in analysis.steelman.replies:
        offered = {g.id for g in CONFIG.steelman.for_standard(reply.standard_id)}
        assert reply.checked and {a.ground_id for a in reply.arguments} <= offered
        assert set(reply.unsupported_grounds) == offered - {a.ground_id for a in reply.arguments}
        assert all(span_is_valid(a.span, resolve) for a in reply.arguments)


@pytest.mark.embed
def test_the_recorded_replies_keep_the_states_real_points_and_nothing_for_the_five_day_delay(demo):
    _, analysis = demo
    by_standard = {}
    for reply in analysis.steelman.replies:
        by_standard.setdefault(reply.standard_id, []).append(reply)
    [renewal] = by_standard["renewal_review"]
    assert [a.ground_id for a in renewal.arguments] == ["investigation_pending"]
    assert "forensic examination of the telephone" in renewal.arguments[0].span.text
    [witnesses] = by_standard["iccpr_14_3_e"]
    assert witnesses.arguments == () and witnesses.dropped  # this recording's attempt failed the checks, so nothing unverified is shown
    [delay] = by_standard["gc35_48h"]
    assert delay.arguments == () and "exceptional_circumstances" in delay.unsupported_grounds  # the record shows no justification
