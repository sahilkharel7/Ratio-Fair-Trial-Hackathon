"""Judge registry: names are transliterated and stripped of titles, matched only within one court,
and anything ambiguous (initials, near-misses, long gaps) waits for a person's decision."""

import random

import pytest

from ratio.config import default_config
from ratio.modules.judge_registry import build_registry, normalise_name
from ratio.schema import AliasDecision
from ratio.testing import CodedRuling, make_history_case

SETTINGS = default_config().settings.judges
TITLES = SETTINGS.title_words
OTHER_COURT = "Other Regional Court"


def case(case_id, judge, date="2024-03-01", court="Test District Court", public=False):
    return make_history_case(case_id, [CodedRuling("detention_ordered", judge, date)], court=court, public=public)


def names(registry):
    return sorted((judge.display_name, judge.court, judge.case_ids) for judge in registry.judges)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ilena Varda", "ilena varda"),
        ("Hon. Iléna Várda", "ilena varda"),
        ("Judge ILENA VARDA", "ilena varda"),
        ("The Hon. Mr Justice Ilena Varda", "ilena varda"),
        ("Mary Judge", "mary judge"),  # only leading titles are stripped
        ("I. Varda", "i varda"),
        ("Varda, Ilena", "varda ilena"),
        ("Mary O'Neill", "mary oneill"),  # an apostrophe does not split the name into an initial
        ("Mary O’Neill", "mary oneill"),
        ("The Presiding Judge", ""),  # titles only: no personal name to match
        ("Hon.", ""),
    ],
)
def test_normalise_name(raw, expected):
    assert normalise_name(raw, TITLES) == expected


def test_variants_at_the_same_court_are_one_judge():
    records = [case("c1", "Ilena Varda"), case("c2", "Iléna Várda"), case("c3", "Hon. Ilena Varda"), case("c4", "Ilena Varda")]
    registry = build_registry(records, (), SETTINGS)
    assert len(registry.judges) == 1 and not registry.pending
    judge = registry.judges[0]
    assert judge.display_name == "Ilena Varda"  # the most common spelling
    assert judge.case_ids == ("c1", "c2", "c3", "c4")
    assert set(judge.variants) == {"Ilena Varda", "Iléna Várda", "Hon. Ilena Varda"}
    assert {registry.judge_of(f"c{n}", name) for n, name in [(1, "Ilena Varda"), (2, "Iléna Várda")]} == {judge.judge_id}


def test_a_close_spelling_above_the_threshold_is_merged():
    registry = build_registry([case("c1", "Ilena Varda"), case("c2", "Ilena Vardah")], (), SETTINGS)
    assert len(registry.judges) == 1 and not registry.pending


def test_the_same_name_at_another_court_is_never_merged():
    registry = build_registry([case("c1", "Ilena Varda"), case("c2", "Ilena Varda", court=OTHER_COURT)], (), SETTINGS)
    assert names(registry) == [("Ilena Varda", OTHER_COURT, ("c2",)), ("Ilena Varda", "Test District Court", ("c1",))]
    assert not registry.pending


def test_initials_only_go_to_the_manual_list_and_count_nowhere():
    records = [case("c1", "Ilena Varda"), case("c2", "Ilena Varda"), case("c3", "I. Varda")]
    registry = build_registry(records, (), SETTINGS)
    assert names(registry) == [("Ilena Varda", "Test District Court", ("c1", "c2"))]
    [pending] = registry.pending
    assert (pending.case_id, pending.raw_name, pending.candidate_name) == ("c3", "I. Varda", "Ilena Varda")
    assert pending.candidate_judge_id == registry.judges[0].judge_id
    assert "initial" in pending.reason
    assert registry.judge_of("c3", "I. Varda") is None


def test_initials_matching_two_judges_name_no_candidate():
    records = [case("c1", "Ilena Varda"), case("c2", "Ivo Varda"), case("c3", "I. Varda")]
    registry = build_registry(records, (), SETTINGS)
    [pending] = registry.pending
    assert pending.candidate_judge_id is None and "2 judges" in pending.reason
    assert pending.candidate_name == "Ilena Varda or Ivo Varda"


def test_initials_with_no_matching_judge_are_a_judge_of_their_own():
    registry = build_registry([case("c1", "Ilena Varda"), case("c2", "P. Hollin")], (), SETTINGS)
    assert [judge.display_name for judge in registry.judges] == ["Ilena Varda", "P. Hollin"]
    assert not registry.pending


def test_a_near_miss_below_the_threshold_waits_for_a_decision():
    records = [case("c1", "Ilena Varda"), case("c2", "Ilena Varda"), case("c3", "Elena Varda")]
    registry = build_registry(records, (), SETTINGS)
    assert names(registry) == [("Ilena Varda", "Test District Court", ("c1", "c2"))]
    [pending] = registry.pending
    assert pending.case_id == "c3" and pending.candidate_name == "Ilena Varda"
    assert SETTINGS.ambiguous_floor <= pending.score < SETTINGS.name_match_threshold


def test_unrelated_names_are_different_judges():
    registry = build_registry([case("c1", "Ilena Varda"), case("c2", "Petra Hollin")], (), SETTINGS)
    assert len(registry.judges) == 2 and not registry.pending


def test_the_same_name_years_later_at_the_same_court_waits_for_a_decision():
    records = [case("c1", "Ilena Varda", "2010-01-05"), case("c2", "Ilena Varda", "2011-02-01"), case("c3", "Ilena Varda", "2024-06-01")]
    registry = build_registry(records, (), SETTINGS)
    assert names(registry) == [("Ilena Varda", "Test District Court", ("c1", "c2"))]
    [pending] = registry.pending
    assert pending.case_id == "c3" and "years" in pending.reason


def test_synthetic_and_public_cases_never_share_a_judge():
    records = [case("c1", "Ilena Varda"), case("c2", "Ilena Varda", public=True), case("c3", "I. Varda", public=True)]
    registry = build_registry(records, (), SETTINGS)
    assert sorted(judge.synthetic for judge in registry.judges) == [False, True]
    [pending] = registry.pending
    assert pending.case_id == "c3" and pending.synthetic is False
    assert pending.candidate_judge_id == next(j.judge_id for j in registry.judges if not j.synthetic)


def test_a_decision_merges_a_pending_name_into_the_judge():
    records = [case("c1", "Ilena Varda"), case("c2", "I. Varda")]
    decisions = [AliasDecision(case_id="c2", name="I. Varda", same_as="Ilena Varda")]
    registry = build_registry(records, decisions, SETTINGS)
    assert names(registry) == [("Ilena Varda", "Test District Court", ("c1", "c2"))]
    assert not registry.pending
    assert registry.judge_of("c2", "I. Varda") == registry.judges[0].judge_id


def test_a_decision_can_keep_a_name_apart():
    records = [case("c1", "Ilena Varda"), case("c2", "I. Varda")]
    registry = build_registry(records, [AliasDecision(case_id="c2", name="I. Varda", same_as=None)], SETTINGS)
    assert [judge.display_name for judge in registry.judges] == ["I. Varda", "Ilena Varda"]
    assert not registry.pending


def test_a_decision_that_matches_nothing_is_reported():
    decisions = [AliasDecision(case_id="c9", name="X. Nobody", same_as=None), AliasDecision(case_id="c1", name="Y. Other", same_as=None)]
    registry = build_registry([case("c1", "Ilena Varda")], decisions, SETTINGS)
    [absent] = [note for note in registry.notes if "c9" in note.text]
    assert absent.synthetic is None and "X. Nobody" not in absent.text  # names no person of a case it cannot see
    [unknown] = [note for note in registry.notes if "c1" in note.text]
    assert unknown.synthetic is True and "Y. Other" in unknown.text


def test_a_decision_naming_no_registered_judge_keeps_the_name_waiting():
    records = [case("c1", "Ilena Varda"), case("c2", "Ilena Varda"), case("x1", "I. Varda")]
    registry = build_registry(records, [AliasDecision(case_id="x1", name="I. Varda", same_as="Judge Varda")], SETTINGS)
    assert names(registry) == [("Ilena Varda", "Test District Court", ("c1", "c2"))]
    [pending] = registry.pending
    assert pending.case_id == "x1" and pending.candidate_judge_id is None
    assert any("x1" in note.text and "Judge Varda" in note.text and note.synthetic for note in registry.notes)


def test_surname_only_and_role_suffixed_forms_wait_for_a_decision():
    records = [case("c1", "Ilena Varda"), case("c2", "Ilena Varda"), case("c3", "Her Honour Judge Varda"), case("c4", "Ilena Varda, presiding judge")]
    registry = build_registry(records, (), SETTINGS)
    assert names(registry) == [("Ilena Varda", "Test District Court", ("c1", "c2"))]
    assert [(p.case_id, p.candidate_name) for p in registry.pending] == [("c3", "Ilena Varda"), ("c4", "Ilena Varda")]


def test_title_only_names_count_nowhere():
    records = [case("c1", "Ilena Varda"), case("c2", "The Presiding Judge"), case("c3", "Judge")]
    registry = build_registry(records, (), SETTINGS)
    assert [judge.display_name for judge in registry.judges] == ["Ilena Varda"]
    assert [(p.case_id, p.candidate_judge_id) for p in registry.pending] == [("c2", None), ("c3", None)]
    assert all("title" in p.reason for p in registry.pending)


def test_apostrophes_and_middle_initials_are_matched_like_other_names():
    records = [
        case("a1", "Mary O'Neill"), case("a2", "Mary O'Neill"), case("a3", "Mary O'Neil"), case("a4", "Mary ONeill"),
        case("b1", "Ilena M. Varda"), case("b2", "Ilena M. Varda"), case("b3", "Ilena M. Vardaa"),
        case("d1", "John A. Smith"), case("d2", "John A. Smith"), case("d3", "John A. Smyth"),
    ]  # fmt: skip
    registry = build_registry(records, (), SETTINGS)
    assert names(registry) == [
        ("Ilena M. Varda", "Test District Court", ("b1", "b2", "b3")),
        ("John A. Smith", "Test District Court", ("d1", "d2")),
        ("Mary O'Neill", "Test District Court", ("a1", "a2", "a3", "a4")),
    ]
    [pending] = registry.pending
    assert (pending.case_id, pending.candidate_name) == ("d3", "John A. Smith")


def test_the_registry_does_not_depend_on_the_order_of_cases():
    records = [
        case("c1", "Ilena Varda"), case("c2", "Iléna Várda"), case("c3", "I. Varda"), case("c4", "Elena Varda"),
        case("c5", "Petra Hollin"), case("c6", "Ilena Varda", court=OTHER_COURT),
    ]  # fmt: skip
    expected = build_registry(records, (), SETTINGS)
    for seed in range(5):
        shuffled = records[:]
        random.Random(seed).shuffle(shuffled)
        assert build_registry(shuffled, (), SETTINGS) == expected


def test_role_suffixed_initials_wait_for_a_decision():
    records = [case("c1", "Ilena Varda"), case("c2", "Ilena Varda"), case("c3", "Judge I. Varda (presiding)"), case("c4", "I. Varda, presiding judge")]
    registry = build_registry(records, (), SETTINGS)
    assert names(registry) == [("Ilena Varda", "Test District Court", ("c1", "c2"))]
    assert [(p.case_id, p.candidate_name) for p in registry.pending] == [("c3", "Ilena Varda"), ("c4", "Ilena Varda")]


def test_names_with_different_initials_are_never_merged():
    records = [case(f"m{n}", "Ilena M. Varda") for n in range(3)] + [case(f"p{n}", "Ilena P. Varda") for n in range(2)]
    records += [case("x1", "Ilena Mira Varda")]  # compatible with M., not with P.
    registry = build_registry(records, (), SETTINGS)
    merged = {judge.display_name: judge.case_ids for judge in registry.judges}
    assert merged == {"Ilena M. Varda": ("m0", "m1", "m2"), "Ilena P. Varda": ("p0", "p1")}  # M. and P. are two people
    [pending] = registry.pending  # 'Mira' may be the M.: close, so a person decides
    assert (pending.case_id, pending.candidate_name) == ("x1", "Ilena M. Varda")


def test_a_title_only_name_can_be_decided_a_judge_of_its_own():
    records = [case("c1", "Ilena Varda"), case("l1", "Judge Lord"), case("l2", "Judge Lord")]
    decisions = [AliasDecision(case_id=case_id, name="Judge Lord", same_as=None) for case_id in ("l1", "l2")]
    registry = build_registry(records, decisions, SETTINGS)
    assert names(registry) == [("Ilena Varda", "Test District Court", ("c1",)), ("Judge Lord", "Test District Court", ("l1", "l2"))]
    assert not registry.pending


def test_a_decision_naming_only_a_title_keeps_the_name_waiting_with_a_note():
    records = [case("c1", "Ilena Varda"), case("x1", "I. Varda")]
    registry = build_registry(records, [AliasDecision(case_id="x1", name="I. Varda", same_as="Judge")], SETTINGS)
    [pending] = registry.pending
    assert pending.case_id == "x1" and pending.candidate_judge_id is None
    assert any("x1" in note.text and "title" in note.text for note in registry.notes)


def test_initials_names_merge_spelling_variants_like_full_names_and_keep_the_common_spelling():
    records = [case(f"h{n}", "P. Hollin") for n in range(6)] + [case("t1", "P. Holin")]
    registry = build_registry(records, (), SETTINGS)
    assert names(registry) == [("P. Hollin", "Test District Court", (*(f"h{n}" for n in range(6)), "t1"))]
    assert not registry.pending


def test_a_decision_matching_two_same_named_judges_uses_the_one_of_its_period():
    records = [case("c1", "Ilena Varda", "2010-01-05"), case("c2", "Ilena Varda", "2010-06-01"), case("c3", "Ilena Varda", "2024-06-01"),
               case("c4", "I. Varda", "2024-07-01")]  # fmt: skip
    decisions = [AliasDecision(case_id="c3", name="Ilena Varda", same_as=None), AliasDecision(case_id="c4", name="I. Varda", same_as="Ilena Varda")]
    registry = build_registry(records, decisions, SETTINGS)
    assert sorted(judge.case_ids for judge in registry.judges) == [("c1", "c2"), ("c3", "c4")]
    assert not registry.pending


def test_an_initials_name_with_a_misspelt_surname_waits_for_a_decision():
    records = [case("c1", "Ilena Varda"), case("c2", "Ilena Varda"), case("v1", "I. Vardah"), case("v2", "Vardah, I.")]
    registry = build_registry(records, (), SETTINGS)
    assert names(registry) == [("Ilena Varda", "Test District Court", ("c1", "c2"))]
    assert [(p.case_id, p.candidate_name) for p in registry.pending] == [("v1", "Ilena Varda"), ("v2", "Ilena Varda")]


def test_a_decision_from_a_file_of_the_other_kind_of_data_is_not_used():
    records = [case("c1", "Maren Tull", public=True), case("c2", "M. Tull", public=True)]
    recorded_as_synthetic = AliasDecision(case_id="c2", name="M. Tull", same_as="Maren Tull", synthetic=True)
    registry = build_registry(records, [recorded_as_synthetic], SETTINGS)
    assert [p.case_id for p in registry.pending] == ["c2"]
    [note] = registry.notes
    assert note.synthetic is False and "c2" in note.text and "synthetic" in note.text
    absent = build_registry(records, [AliasDecision(case_id="c9", name="X", same_as=None, synthetic=True)], SETTINGS)
    assert [note.synthetic for note in absent.notes] == [True]  # the file's kind, so the page shows it on its own view
