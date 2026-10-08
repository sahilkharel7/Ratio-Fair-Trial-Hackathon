"""The demo story (PDF demo steps 2 and 3), replayed from the committed cache of the real model:
the rights coverage grid, the interpreter follow-up, and the single red interval."""

import json

import pytest

from ratio.config import load_config
from ratio.expected import ExpectedFlags
from ratio.extraction.build import load_case
from ratio.paths import DEMO_CASE_DIR, GOLD_DIR
from ratio.pipeline import analysis_context, analyze, demo_llm, ingest

EXPECTED = ExpectedFlags.model_validate(json.loads((GOLD_DIR / "expected_flags.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def replayed():
    config = load_config()
    llm = demo_llm(config)
    record, _ = ingest(load_case(DEMO_CASE_DIR), llm, config)
    analysis = analyze(record, analysis_context(config, llm))
    return record, analysis, llm


def test_the_whole_story_replays_without_the_model(replayed):
    _, analysis, llm = replayed
    assert llm.replay_only and llm.stats.misses == 0
    assert analysis.llm_mode == "replay"


def test_every_guarantee_status_matches_the_gold_record(replayed):
    _, analysis, _ = replayed
    pinned = {item.standard_id: item.status for item in EXPECTED.expected if item.kind == "status"}
    assert {a.rubric_id: a.status for a in analysis.absence.assessments} == pinned


def test_interpreter_follow_up_shows_the_language_context(replayed):
    _, analysis, _ = replayed
    follow_up = next(u for u in analysis.absence.follow_ups if u.rubric_id == "iccpr_14_3_f")
    assert "interpreter" in follow_up.question.lower()
    assert any("first language is Ostric" in e.span.text for e in follow_up.context)


def test_exactly_one_red_interval_first_appearance_after_five_days(replayed):
    _, analysis, _ = replayed
    red = [f for f in analysis.clock.flags if f.status == "exceeds_benchmark"]
    assert [f.standard_id for f in red] == ["gc35_48h"]
    gc35 = next(i for i in analysis.clock.intervals if i.benchmark_id == "gc35_48h")
    assert (gc35.min_hours, gc35.max_hours) == (96, 144)


def test_no_flag_was_dropped_and_every_span_is_exact_source_text(replayed):
    record, analysis, _ = replayed
    assert analysis.dropped_flags == 0
    for flag in analysis.all_flags():
        assert flag.evidence
        for span in flag.spans:
            assert record.document(span.doc_id).text[span.start : span.end] == span.text
