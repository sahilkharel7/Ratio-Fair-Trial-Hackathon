"""Dates are parsed in code (dateparser); the model's own date is kept only for comparison."""

import datetime as dt

import pytest

from ratio.extraction.dates import has_explicit_year, parse_date_text, parse_iso_date

BASE = dt.datetime(2025, 6, 2)


def test_day_month_year_has_date_precision():
    parsed = parse_date_text("14 February 2025", base=BASE)
    assert parsed is not None
    assert parsed.value.date() == dt.date(2025, 2, 14)
    assert parsed.precision == "date"


def test_time_of_day_gives_datetime_precision():
    parsed = parse_date_text("2 June 2025 at 10:05", base=BASE)
    assert parsed is not None
    assert parsed.value == dt.datetime(2025, 6, 2, 10, 5)
    assert parsed.precision == "datetime"


def test_month_only_has_month_precision():
    parsed = parse_date_text("February 2025", base=BASE)
    assert parsed is not None
    assert parsed.precision == "month"


def test_numeric_dates_are_read_day_first():
    parsed = parse_date_text("03/08/2025", base=BASE)
    assert parsed is not None
    assert parsed.value.date() == dt.date(2025, 8, 3)


@pytest.mark.parametrize("text", [None, "", "unknown", "null", "N/A", "not stated", "the hearing"])
def test_placeholders_and_non_dates_give_none(text):
    assert parse_date_text(text, base=BASE) is None


def test_explicit_year_detection():
    assert has_explicit_year("14 February 2025")
    assert not has_explicit_year("14 February")
    assert not has_explicit_year("two days later")


def test_parsing_does_not_depend_on_the_wall_clock():
    first = parse_date_text("14 February", base=BASE)
    second = parse_date_text("14 February", base=dt.datetime(2019, 6, 1))
    assert first is not None and second is not None
    assert first.value.year == 2025
    assert second.value.year == 2019


def test_model_iso_dates():
    assert parse_iso_date("2025-02-19") == dt.date(2025, 2, 19)
    assert parse_iso_date("unknown") is None
    assert parse_iso_date("19 Feb") is None
    assert parse_iso_date(None) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("2025-06-02", dt.date(2025, 6, 2)), ("2025-06-14", dt.date(2025, 6, 14)), ("2025-03-08", dt.date(2025, 3, 8))],
)
def test_iso_dates_are_read_year_first(text, expected):
    parsed = parse_date_text(text, base=BASE)
    assert parsed is not None and parsed.value.date() == expected and parsed.precision == "date"


def test_iso_datetime_keeps_its_time():
    parsed = parse_date_text("2025-06-02T10:05", base=BASE)
    assert parsed is not None and parsed.value == dt.datetime(2025, 6, 2, 10, 5) and parsed.precision == "datetime"


@pytest.mark.parametrize("text", ["9-12 February 2025", "2 and 3 June 2025", "02/2025", "6/2025"])
def test_ranges_and_numeric_month_year_are_not_guessed(text):
    assert parse_date_text(text, base=BASE) is None


def test_without_a_base_date_a_year_must_be_written():
    assert parse_date_text("2 June") is None
    assert parse_date_text("2 June 2025") is not None
