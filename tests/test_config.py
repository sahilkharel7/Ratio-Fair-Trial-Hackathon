"""Config validation. The legal-content rules (hard rule 5) are enforced in code and tested here."""

import shutil

import pytest
from pydantic import ValidationError

from ratio.config import (
    CONFIRMED_BENCHMARKS,
    Benchmark,
    ConfigError,
    RateDef,
    load_config,
)
from ratio.paths import CONFIG_DIR


@pytest.fixture(scope="module")
def config():
    return load_config()


class TestRubric:
    def test_covers_article_14_3_a_to_g_in_order(self, config):
        assert [item.id for item in config.rubric.items] == [f"iccpr_14_3_{c}" for c in "abcdefg"]

    def test_every_item_is_cited_and_marked_for_legal_review(self, config):
        for item in config.rubric.items:
            assert item.citation.instrument in config.rubric.sources
            assert item.citation.paras
            assert item.review_status == "needs_legal_review"
            assert item.follow_up.strip()
            assert item.parts

    def test_paragraphs_match_the_verified_gc32_mapping(self, config):
        verified = {"a": "31", "b": "32-34", "c": "35", "d": "36-38", "e": "39", "f": "40", "g": "41"}
        for letter, paras in verified.items():
            item = config.rubric_item(f"iccpr_14_3_{letter}")
            assert item.citation.instrument == "gc32"
            assert item.citation.paras == paras

    def test_indicator_ids_are_unique_across_the_rubric(self, config):
        ids = [
            indicator.id
            for item in config.rubric.items
            for indicators in ([part.compliance + part.violation for part in item.parts] + [item.context])
            for indicator in indicators
        ]
        assert len(ids) == len(set(ids))

    def test_interpreter_language_fact_is_context_not_evidence(self, config):
        item = config.rubric_item("iccpr_14_3_f")
        assert any("language" in context.text for context in item.context)
        evidence_ids = {i.id for part in item.parts for i in part.compliance + part.violation}
        assert evidence_ids.isdisjoint({context.id for context in item.context})

    def test_interpreter_must_be_evidenced_at_each_hearing(self, config):
        part = config.rubric_item("iccpr_14_3_f").parts[0]
        assert part.scope == "per_hearing"
        assert part.required

    def test_indicator_lookup_returns_part_and_label(self, config):
        part, label = config.indicator_index["d_defendant_present"]
        assert part.id == "d_presence"
        assert label == "supports"


class TestBenchmarks:
    def test_only_gc35_48h_is_confirmed(self, config):
        confirmed = [b.id for b in config.benchmarks.benchmarks if b.review_status == "confirmed"]
        assert confirmed == ["gc35_48h"]
        assert CONFIRMED_BENCHMARKS == frozenset({"gc35_48h"})

    def test_gc35_benchmark_matches_the_verified_source(self, config):
        benchmark = config.benchmark("gc35_48h")
        assert benchmark.threshold_hours == 48
        assert benchmark.from_event == "arrest"
        assert benchmark.to_event == "first_appearance"
        assert benchmark.citation.instrument == "gc35"
        assert benchmark.citation.paras == "33"
        assert "48 hours is ordinarily sufficient" in (benchmark.citation.quote or "")

    def test_other_benchmarks_are_cited_but_have_no_threshold(self, config):
        others = [b for b in config.benchmarks.benchmarks if b.id != "gc35_48h"]
        assert len(others) >= 2  # three cited benchmarks in total
        for benchmark in others:
            assert benchmark.review_status == "needs_legal_review"
            assert benchmark.threshold_hours is None
            assert benchmark.citation.paras
            assert benchmark.citation.instrument in config.benchmarks.sources

    def test_confirming_a_benchmark_outside_the_allowlist_is_rejected(self):
        with pytest.raises(ValidationError):
            Benchmark(
                id="counsel_access",
                name="Access to counsel",
                provision="ICCPR Art. 14(3)(b)",
                from_event="arrest",
                to_event="counsel_access",
                threshold_hours=24,
                citation={"instrument": "gc35", "paras": "35"},
                review_status="confirmed",
            )

    def test_a_confirmed_benchmark_needs_a_threshold(self):
        with pytest.raises(ValidationError):
            Benchmark(
                id="gc35_48h",
                name="Brought promptly before a judge",
                provision="ICCPR Art. 9(3)",
                from_event="arrest",
                to_event="first_appearance",
                threshold_hours=None,
                citation={"instrument": "gc35", "paras": "33"},
                review_status="confirmed",
            )


class TestRulingCodes:
    def test_rates_reference_known_codes(self, config):
        codes = {code.id for code in config.ruling_codes.codes}
        for rate in config.ruling_codes.rates:
            assert set(rate.denominator) <= codes

    def test_spec_ruling_codes_are_present(self, config):
        codes = {code.id for code in config.ruling_codes.codes}
        assert {
            "detention_ordered",
            "detention_refused",
            "defense_motion_granted",
            "defense_motion_denied",
            "hearing_closed",
            "contested_evidence_admitted",
            "verdict_conviction",
            "sentence_months",
        } <= codes

    def test_numerator_must_be_part_of_denominator(self):
        with pytest.raises(ValidationError):
            RateDef(id="r", label="r", numerator=("a",), denominator=("b",), combine="any")


class TestSettings:
    def test_llm_only_talks_to_loopback(self, config):
        assert config.settings.llm.host.startswith("http://127.0.0.1")

    def test_judge_minimum_case_count_defaults_to_five(self, config):
        assert config.settings.judges.min_case_count == 5

    def test_reuse_shingles_are_word_five_grams(self, config):
        assert config.settings.reuse.shingle_size == 5


def test_reuse_and_judge_standards_are_cited_and_need_legal_review(config):
    for standard_id in ("reasoning_reuse", "unaddressed_defense_argument", "judicial_pattern"):
        standard = config.standard(standard_id)
        assert standard.citation.strip()
        assert standard.review_status == "needs_legal_review"


def test_corrupt_config_raises_a_clear_error(tmp_path):
    broken = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, broken)
    (broken / "benchmarks.yaml").write_text("benchmarks: [ {id: x} ]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="benchmarks.yaml"):
        load_config(broken)


def test_a_key_written_twice_in_a_config_file_is_refused(tmp_path):
    broken = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, broken)
    messages = broken / "messages.yaml"
    text = messages.read_text(encoding="utf-8")
    messages.write_text(text.replace('  role_ruling: "Coded ruling"\n', '  role_ruling: "Coded ruling"\n  role_ruling: "Ruling"\n', 1), encoding="utf-8")
    with pytest.raises(ConfigError, match="messages.yaml.*duplicate key 'role_ruling'"):
        load_config(broken)


def test_a_coded_ruling_is_labelled_as_one(config):
    assert config.messages.labels["role_ruling"] == "Coded ruling"


def test_a_key_that_overrides_a_merged_mapping_is_still_allowed(tmp_path):
    from ratio.config import _read_yaml

    path = tmp_path / "merge.yaml"
    path.write_text("base: &base {a: 1, b: 2}\nderived:\n  <<: *base\n  b: 3\n", encoding="utf-8")
    assert _read_yaml(path)["derived"] == {"a": 1, "b": 3}


@pytest.mark.parametrize(
    "text",
    [
        "templates:\n  - &t\n    <<: {a: 1, b: 2}\n    b: 3\nitems:\n  x:\n    <<: *t\n",  # an anchor in a list, merged later
        "x:\n  <<: &m\n    <<: {a: 1}\n    a: 2\ny:\n  m: *m\n",  # a merge source that itself merges
        "=: x\n",  # the "=" key, read as a plain string
    ],
)
def test_valid_merges_and_special_keys_load_as_the_safe_loader_reads_them(tmp_path, text):
    import yaml

    from ratio.config import _read_yaml

    path = tmp_path / "valid.yaml"
    path.write_text(text, encoding="utf-8")
    assert _read_yaml(path) == yaml.safe_load(text)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("d:\n  <<: {a: 1, a: 2}\n", "duplicate key 'a'"),  # inside an inline merge source
        ("? [a, b]\n: x\n", "unhashable"),  # a list as a key
    ],
)
def test_broken_keys_are_refused_with_the_file_named(tmp_path, text, match):
    from ratio.config import _read_yaml

    path = tmp_path / "broken.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=f"(?s)broken.yaml.*{match}"):  # the message spans lines
        _read_yaml(path)
