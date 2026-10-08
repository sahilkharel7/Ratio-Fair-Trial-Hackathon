"""SQLite store: round trips, overwrite, listing and the last-viewed case."""

import pytest

from ratio.results import CaseAnalysis
from ratio.schema import CaseMeta, CaseRecord, Document
from ratio.store import CaseStore


def make_record(case_id: str, title: str = "Republic v. Test") -> CaseRecord:
    meta = CaseMeta(
        case_id=case_id,
        title=title,
        court="Mirevo District Court",
        charge_type="Art. 214",
        data_provenance="synthetic",
        synthetic=True,
    )
    doc = Document(
        id=f"{case_id}/judgment.txt",
        case_id=case_id,
        path="judgment.txt",
        type="judgment",
        title="Judgment",
        text="The Court finds the accused guilty.",
        synthetic=True,
        sha256="0" * 64,
    )
    return CaseRecord(meta=meta, documents=(doc,))


@pytest.fixture
def store(tmp_path):
    return CaseStore(tmp_path / "ratio.db")


def test_case_round_trip(store):
    record = make_record("case-1")
    store.save_case(record)
    assert store.load_case("case-1") == record


def test_missing_case_and_analysis_return_none(store):
    assert store.load_case("nope") is None
    assert store.load_analysis("nope") is None
    assert store.last_case() is None


def test_saving_again_overwrites(store):
    store.save_case(make_record("case-1", title="Old"))
    store.save_case(make_record("case-1", title="New"))
    assert store.load_case("case-1").meta.title == "New"
    assert [summary.case_id for summary in store.list_cases()] == ["case-1"]


def test_ids_with_quotes_are_stored_safely(store):
    record = make_record("o'brien-2025")
    store.save_case(record)
    assert store.load_case("o'brien-2025") == record


def test_analysis_round_trip_and_last_case(store):
    store.save_case(make_record("case-1"))
    analysis = CaseAnalysis(case_id="case-1", dropped_flags=0, llm_model="fake")
    store.save_analysis(analysis)
    store.set_last_case("case-1")
    assert store.load_analysis("case-1") == analysis
    assert store.last_case() == "case-1"


def test_all_records_lists_every_case(store):
    store.save_case(make_record("b"))
    store.save_case(make_record("a"))
    assert sorted(r.case_id for r in store.all_records()) == ["a", "b"]
    summaries = store.list_cases()
    assert all(summary.synthetic for summary in summaries)


def test_store_creates_parent_directories(tmp_path):
    store = CaseStore(tmp_path / "nested" / "dir" / "ratio.db")
    store.save_case(make_record("x"))
    assert store.load_case("x") is not None


def test_saving_a_case_again_discards_its_stale_analysis(store):
    store.save_case(make_record("case-1"))
    store.save_analysis(CaseAnalysis(case_id="case-1"))
    store.save_case(make_record("case-1", title="Changed"))
    assert store.load_analysis("case-1") is None
