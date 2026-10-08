"""Procedural Clock: intervals computed in code; red only when every reading of the record's
dates exceeds a confirmed benchmark."""

import datetime as dt

import pytest

from ratio.config import load_config
from ratio.context import AnalysisContext
from ratio.modules import clock
from ratio.paths import GOLD_DIR
from ratio.schema import CaseRecord, Event
from ratio.testing import FakeEmbedder, FakeLLM

CONFIG = load_config()
MOCK = CaseRecord.model_validate_json((GOLD_DIR / "mock_record.json").read_text(encoding="utf-8"))


def run(record: CaseRecord):
    def no_model(*_):
        pytest.fail("the clock must never call the model")

    return clock.run(record, AnalysisContext(llm=FakeLLM(no_model), embedder=FakeEmbedder(), config=CONFIG))


def interval(result, benchmark_id):
    return next(item for item in result.intervals if item.benchmark_id == benchmark_id)


def replace_events(events) -> CaseRecord:
    return MOCK.model_copy(update={"events": tuple(events)})


def stray(type_: str, date: dt.date, **changes) -> Event:
    template = next(e for e in MOCK.events if e.type == "first_appearance")
    fields = {"id": f"stray-{type_}-{date}", "type": type_, "parsed_date": dt.datetime.combine(date, dt.time())}
    return template.model_copy(update={**fields, **changes})


def test_first_appearance_five_days_after_arrest_is_red():
    result = run(MOCK)
    gc35 = interval(result, "gc35_48h")
    assert gc35.status == "exceeds_benchmark"
    assert (gc35.min_hours, gc35.max_hours) == (96, 144)  # 14 Feb to 19 Feb, day precision: 4 to 6 days
    assert gc35.threshold_hours == 48 and gc35.review_status == "confirmed"
    (flag,) = [f for f in result.flags if f.standard_id == "gc35_48h"]
    assert {e.role for e in flag.evidence} == {"from_event", "to_event"}
    assert "absolutely exceptional" in flag.message
    assert "violation" not in flag.message.lower()
    assert flag.citation and "CCPR/C/GC/35" in flag.citation and "33" in flag.citation


def test_exactly_one_red_interval_on_the_demo_record():
    assert [f.standard_id for f in run(MOCK).flags if f.status == "exceeds_benchmark"] == ["gc35_48h"]


def test_benchmarks_without_a_confirmed_threshold_are_measured_and_never_coloured():
    result = run(MOCK)
    for item in result.intervals:
        if item.benchmark_id != "gc35_48h":
            assert item.status in {"measured", "cannot_compute"}
            assert item.flag_id is None
    detention = interval(result, "pretrial_detention_length")
    assert detention.status == "measured" and detention.min_hours is not None


def test_timeline_merges_mentions_and_keeps_repeated_events_apart():
    timeline = run(MOCK).timeline
    (arrest,) = [t for t in timeline if t.type == "arrest"]
    assert len(arrest.mentions) == 3 and arrest.state == "ok"
    assert len([t for t in timeline if t.type == "detention_extension"]) == 2
    dated = [t.date for t in timeline if t.date is not None]
    assert dated == sorted(dated)


def test_a_stray_later_date_cannot_erase_the_red_interval():
    result = run(replace_events([*MOCK.events, stray("first_appearance", dt.date(2025, 3, 4))]))
    gc35 = interval(result, "gc35_48h")
    assert gc35.status == "exceeds_benchmark"
    assert "conflict" in gc35.note
    assert [t.state for t in result.timeline if t.type == "first_appearance"] == ["conflict"]


def test_a_stray_earlier_date_cannot_create_a_red_interval():
    result = run(replace_events([*MOCK.events, stray("first_appearance", dt.date(2025, 2, 15))]))
    assert interval(result, "gc35_48h").status in {"may_exceed", "within_benchmark"}
    assert not any(f.status == "exceeds_benchmark" for f in result.flags)


def test_dates_marked_for_review_give_amber_not_red():
    reviewed = [
        e.model_copy(update={"needs_review": True, "review_reasons": ("model and dateparser disagree",)})
        if e.type == "first_appearance"
        else e
        for e in MOCK.events
    ]
    result = run(replace_events(reviewed))
    assert interval(result, "gc35_48h").status == "needs_review"
    assert [f.status for f in result.flags] == ["needs_review"]


def test_missing_end_event_cannot_be_computed():
    gc35 = interval(run(replace_events(e for e in MOCK.events if e.type != "first_appearance")), "gc35_48h")
    assert gc35.status == "cannot_compute"
    assert gc35.min_hours is None


def test_an_end_date_before_the_start_is_ignored():
    events = [e for e in MOCK.events if e.type != "first_appearance"] + [stray("first_appearance", dt.date(2025, 2, 1))]
    assert interval(run(replace_events(events)), "gc35_48h").status == "cannot_compute"


def test_exact_times_give_exact_hours():
    arrest = stray("arrest", dt.date(2025, 2, 14), precision="datetime", parsed_date=dt.datetime(2025, 2, 14, 6, 0))
    appearance = stray(
        "first_appearance", dt.date(2025, 2, 15), precision="datetime", parsed_date=dt.datetime(2025, 2, 15, 18, 0)
    )
    events = [e for e in MOCK.events if e.type not in {"arrest", "first_appearance"}] + [arrest, appearance]
    gc35 = interval(run(replace_events(events)), "gc35_48h")
    assert gc35.min_hours == gc35.max_hours == 36
    assert gc35.status == "within_benchmark"
