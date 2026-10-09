"""The jurisprudence linker: the corpus's rules (official sources only, a decision only with the footnote
that cites it, known standards and statuses, no characterisation of people), the choice of entries for a
finding or a question for the monitor, and what the demo shows. The quotes themselves are checked
against the official documents by scripts/check_jurisprudence.py, which needs the network."""

import pytest
from pydantic import ValidationError

from ratio.config import Jurisprudence, JurisprudenceEntry, load_config
from ratio.jurisprudence import Reference, cited_entries, for_finding, for_follow_up, for_standard
from ratio.messages import contains_blocked_term
from ratio.schema import FLAG_STATUSES, Evidence, Flag, FollowUp, SourceSpan

CONFIG = load_config()
CORPUS = CONFIG.jurisprudence
SPAN = SourceSpan(doc_id="c/judgment.txt", start=0, end=4, text="The ")


def flag(standard_id: str, status: str, module: str = "absence") -> Flag:
    return Flag(
        id="f", case_id="c", module=module, standard_id=standard_id, standard_label="x", status=status, message="m",
        evidence=(Evidence(role="judgment", span=SPAN),), review_status="confirmed" if status == "exceeds_benchmark" else "needs_legal_review",
    )  # fmt: skip


def ids(references) -> list[str]:
    return [ref.entry.id for ref in references]


# --- the corpus -----------------------------------------------------------------------------


def test_every_entry_names_known_standards_and_statuses():
    known = {item.id for item in CONFIG.rubric.items} | {b.id for b in CONFIG.benchmarks.benchmarks} | set(CONFIG.standards.standards)
    statuses = {status for values in FLAG_STATUSES.values() for status in values}
    for entry in CORPUS.entries:
        assert set(entry.standards) <= known, entry.id
        assert set(entry.statuses) <= statuses, entry.id


def test_every_standard_ratio_flags_has_a_general_comment_paragraph():
    standards = {item.id for item in CONFIG.rubric.items} | {"gc35_48h"} | set(CONFIG.standards.standards)
    for standard_id in standards:
        assert any(ref.entry.kind == "general_comment" for ref in for_standard(standard_id, None, CORPUS)), standard_id


def test_sources_are_official_and_decisions_carry_the_footnote_that_cites_them():
    assert all(source.url.startswith("https://documents.un.org/") for source in CORPUS.sources.values())
    views = [e for e in CORPUS.entries if e.kind == "views"]
    assert views and all(e.communication in e.cited_as and e.pinpoint.startswith("footnote ") for e in views)
    with pytest.raises(ValidationError, match="footnote that cites it"):
        JurisprudenceEntry(id="x", kind="views", source="gc32", pinpoint="footnote 1", standards=["iccpr_14_3_b"], case="A v. B", communication="1/2000")
    with pytest.raises(ValidationError, match="a quote and no case"):
        JurisprudenceEntry(id="x", kind="general_comment", source="gc32", pinpoint="para. 1", standards=["iccpr_14_3_b"])
    bad = CORPUS.model_dump() | {"sources": {"gc32": {"symbol": "x", "title": "x", "url": "https://example.org/gc32.pdf"}}}
    with pytest.raises(ValidationError, match="official UN document"):
        Jurisprudence.model_validate(bad)


def test_no_entry_characterises_a_person_and_a_judge_pattern_is_never_shown_a_decision():
    for entry in CORPUS.entries:
        for text in (entry.quote, entry.note, entry.cited_as):
            assert text is None or not contains_blocked_term(text, CONFIG.messages.block_list), entry.id
    pattern = for_finding(flag("judicial_pattern", "pattern_warrants_review", "judges"), CORPUS)
    assert ids(pattern) == ["gc32_para21"]  # the paragraph its standard already cites, and no decision (hard rule 3)


# --- choosing entries -----------------------------------------------------------------------


def test_general_comment_paragraphs_come_first_and_decisions_only_for_their_statuses():
    violation = ids(for_finding(flag("iccpr_14_3_b", "evidence_of_violation"), CORPUS))
    assert violation == ["gc32_para32", "gc32_para33", "gc32_para34", "chan_v_guyana", "phillip_v_trinidad_and_tobago"]
    compliance = ids(for_finding(flag("iccpr_14_3_b", "evidence_of_compliance"), CORPUS))
    assert compliance == ["gc32_para32", "gc32_para33", "gc32_para34"]


def test_a_question_for_the_monitor_is_shown_general_comment_paragraphs_only():
    question = FollowUp(id="q", case_id="c", rubric_id="iccpr_14_3_f", question="Was an interpreter present?")
    assert ids(for_follow_up(question, CORPUS)) == ["gc32_para40"]  # Guesdon is for findings only
    assert ids(for_finding(flag("iccpr_14_3_f", "evidence_of_violation"), CORPUS)) == ["gc32_para40", "guesdon_v_france"]


def test_an_unknown_standard_gets_nothing():
    assert for_standard("no_such_standard", None, CORPUS) == ()


def test_a_reference_shows_its_citation_and_its_own_words():
    gc, view = (Reference(e, CORPUS.sources[e.source]) for e in (CORPUS.entries[4], CORPUS.entries[2]))
    assert gc.citation == "CCPR/C/GC/32, para. 33" and gc.shown.startswith("“‘Adequate facilities’ must include")
    assert view.citation == "Chan v. Guyana, communication No. 913/2000 (CCPR/C/GC/32, footnote 68, to para. 32)"
    assert view.shown == view.entry.note


# --- the demo -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo():
    from ratio.extraction.build import load_case
    from ratio.feedback import case_findings
    from ratio.history import load_alias_decisions, load_history
    from ratio.paths import DEMO_CASE_DIR, MINILM_DIR
    from ratio.pipeline import analysis_context, analyze, analyze_judges, demo_llm, ingest

    if not (MINILM_DIR / "modules.json").exists():
        pytest.skip("run scripts/fetch_models.py once")
    llm = demo_llm(CONFIG)
    record, _ = ingest(load_case(DEMO_CASE_DIR), llm, CONFIG)
    analysis = analyze(record, analysis_context(CONFIG, llm))
    history = [r for r in load_history() if r.case_id != record.case_id]
    judges = analyze_judges([*history, record], {record.case_id: analysis}, load_alias_decisions(), CONFIG)
    return analysis, case_findings(analysis, judges)


@pytest.mark.embed
def test_every_demo_finding_and_question_has_jurisprudence_and_the_planted_ones_their_decisions(demo):
    analysis, flags = demo
    assert all(for_finding(f, CORPUS) for f in flags)
    assert all(for_follow_up(q, CORPUS) for q in analysis.absence.follow_ups)
    by_status = {f.status: ids(for_finding(f, CORPUS)) for f in flags}
    assert "teran_jijon_v_ecuador" in by_status["exceeds_benchmark"]  # five days, planted
    assert {"gc35_para38", "taright_v_algeria", "smantser_v_belarus"} <= set(by_status["repeated_grounds"])
    assert by_status["order_gap"] == ["gc35_para22", "mclawrence_v_jamaica"]
    shown = cited_entries(flags, analysis.absence.follow_ups, CORPUS)
    assert len(shown) >= 20 and "freemantle_v_jamaica" in shown
