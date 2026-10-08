"""The demo data: marked SYNTHETIC everywhere, planted issues present, gold files in sync."""

import datetime as dt
import json

import pytest
import yaml

from ratio.expected import ExpectedFlags, GoldTimeline
from ratio.extraction.build import load_case
from ratio.extraction.dates import parse_date_text
from ratio.gold import build_mock_record, gold_files, load_annotations, resolve_anchor
from ratio.paths import DEMO_CASE_DIR, DEMO_DIR, GOLD_DIR
from ratio.schema import CaseRecord

MARKER = "SYNTHETIC: fictional case; all people, courts, country, language and laws are invented."


@pytest.fixture(scope="module")
def record():
    return load_case(DEMO_CASE_DIR)


@pytest.fixture(scope="module")
def annotations():
    return load_annotations(GOLD_DIR / "annotations.yaml")


def test_every_text_file_starts_with_the_synthetic_marker():
    files = sorted(DEMO_DIR.rglob("*.txt"))
    assert files
    for path in files:
        assert path.read_text(encoding="utf-8").splitlines()[0] == MARKER, path


def test_every_yaml_and_json_file_is_flagged_synthetic():
    for path in sorted(DEMO_DIR.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data.get("synthetic") is True, path
    json_files = sorted(DEMO_DIR.rglob("*.json"))  # gold files, the recorded model replies and their manifest
    assert len(json_files) > 10
    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert True in (data.get("synthetic"), data.get("meta", {}).get("synthetic")), path  # a case record keeps it in meta


def test_demo_case_shape(record):
    notes = record.documents_of_type("monitoring_note")
    assert [doc.date for doc in notes] == [
        dt.date(2025, 6, 2),
        dt.date(2025, 6, 16),
        dt.date(2025, 6, 30),
        dt.date(2025, 7, 14),
    ]
    assert record.meta.presiding_judge == "Ilena Varda"
    assert len(record.documents_of_type("indictment")) == 1
    assert len(record.documents_of_type("judgment")) == 1
    assert {ruling.code for ruling in record.rulings} >= {"detention_ordered", "verdict_conviction", "hearing_closed"}


def test_planted_interpreter_omission(record):
    notes_text = " ".join(doc.text for doc in record.documents_of_type("monitoring_note")).lower()
    assert "interpret" not in notes_text
    assert "first language is ostric" in notes_text


def test_planted_five_day_delay_and_prosecutor_negative_control(record):
    indictment = record.documents_of_type("indictment")[0].text
    assert "arrested at his home in Mirevo in the early morning of 14 February 2025" in indictment
    assert "On 15 February 2025 the accused was brought before the Public Prosecutor" in indictment
    assert "On 19 February 2025 the accused was brought before Judge Ilena Varda" in indictment


def test_every_annotation_quote_and_anchor_resolves_exactly_once(record, annotations):
    mock = build_mock_record(DEMO_CASE_DIR, annotations)
    assert len(mock.events) == len(record.events) + len(annotations.events)
    for item in annotations.expected + annotations.must_not_flag:
        for anchor in item.anchors:
            assert resolve_anchor(record, anchor).text == anchor.quote


def test_annotated_dates_agree_with_dateparser(annotations):
    for entry in annotations.events:
        parsed = parse_date_text(entry.date_text)
        assert parsed is not None, entry.id
        assert parsed.value.date() == entry.date, entry.id


def test_expected_flags_pin_every_guarantee_status(annotations):
    statuses = {item.standard_id for item in annotations.expected if item.kind == "status"}
    assert statuses == {f"iccpr_14_3_{c}" for c in "abcdefg"}


def test_committed_gold_files_match_the_annotations():
    for name, text in gold_files().items():
        committed = (GOLD_DIR / name).read_text(encoding="utf-8")
        assert committed == text, f"{name} is stale: run `python -m ratio.gold`"


def test_gold_files_validate_against_the_contract():
    CaseRecord.model_validate_json((GOLD_DIR / "mock_record.json").read_text(encoding="utf-8"))
    ExpectedFlags.model_validate(json.loads((GOLD_DIR / "expected_flags.json").read_text(encoding="utf-8")))
    timeline = GoldTimeline.model_validate(json.loads((GOLD_DIR / "gold_timeline.json").read_text(encoding="utf-8")))
    assert {event.type for event in timeline.events} >= {"arrest", "first_appearance", "verdict"}
