"""Synthetic judicial history for Module 4: generated files in sync, every ruling quote aligned."""

import importlib.util
from collections import Counter

import pytest

from ratio.extraction.build import load_case
from ratio.paths import HISTORY_DIR, REPO_ROOT

spec = importlib.util.spec_from_file_location("build_history", REPO_ROOT / "scripts" / "build_history.py")
build_history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_history)


@pytest.fixture(scope="module")
def seed():
    return build_history.load_seed()


@pytest.fixture(scope="module")
def records(seed):
    return {case["id"]: load_case(HISTORY_DIR / case["id"]) for case in seed["cases"]}


def test_generated_files_are_in_sync_with_the_seed(seed):
    for relative, text in build_history.history_files(seed).items():
        assert (HISTORY_DIR / relative).read_text(encoding="utf-8") == text, (
            f"{relative} is stale: run `python scripts/build_history.py`"
        )


def test_every_case_loads_with_aligned_rulings_and_a_presiding_judge(seed, records):
    assert len(records) == 20
    for case in seed["cases"]:
        record = records[case["id"]]
        assert record.meta.synthetic
        assert record.meta.presiding_judge == case["judge"]
        assert record.rulings
        for ruling in record.rulings:
            doc = record.document(ruling.span.doc_id)
            assert doc.text[ruling.span.start : ruling.span.end] == ruling.span.text


def test_seed_matches_the_designed_judge_profile(seed):
    mirevo = [c for c in seed["cases"] if "court" not in c]
    varda = [c for c in mirevo if c["judge"] in {"Ilena Varda", "Iléna Várda", "Judge Ilena Varda", "Hon. Ilena Varda"}]
    baseline = [c for c in mirevo if c["judge"] in {"Petra Hollin", "Anwar Selik"}]
    assert len(varda) == 8 and len(baseline) == 10
    assert Counter(c["detention"] for c in varda) == {"ordered": 8}
    assert Counter(c["detention"] for c in baseline)["ordered"] == 4
    assert Counter(c["verdict"] for c in baseline)["conviction"] == 7
    assert Counter(c["hearings"] for c in baseline)["closed"] == 2
    assert sum(1 for c in varda if c.get("evidence")) == 2  # + demo case = 3, below the minimum of 5


def test_registry_edge_cases_are_present(seed):
    judges = {(c["judge"], c.get("court", "Mirevo")) for c in seed["cases"]}
    assert ("I. Varda", "Mirevo") in judges
    assert ("Ilena Varda", "Port Elsin Regional Court") in judges
