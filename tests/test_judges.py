"""Module 4, Judicial History Tracker: per-case outcomes, the baseline of the same court and charge
type, hidden indicators below the minimum case count, the pattern rule, and the designed outcome
of the synthetic demo history (one indicator fires, two are shown plainly, two are hidden)."""

import pytest

from ratio.config import default_config
from ratio.extraction.build import load_case
from ratio.history import load_alias_decisions, load_history
from ratio.messages import contains_blocked_term
from ratio.modules import judges
from ratio.paths import DEMO_CASE_DIR
from ratio.pipeline import analyze_judges
from ratio.provenance import check_judges, resolver_for, span_is_valid
from ratio.results import CaseAnalysis, ReuseResult
from ratio.schema import AliasDecision
from ratio.testing import CodedRuling, make_history_case

CONFIG = default_config()
DETENTION = CONFIG.rate("detention_at_first_appearance")


def ruling(code, judge="Ilena Varda", date=None, value=None):
    return CodedRuling(code, judge, date, value)


def outcome(rulings, rate_id):
    record = make_history_case("c1", rulings)
    found = judges.case_outcome(record.rulings, CONFIG.rate(rate_id), record)
    return None if found is None else (found[0], found[1].code, found[1].date and found[1].date.isoformat())


def test_first_counts_the_earliest_dated_ruling_once_per_case():
    rulings = [ruling("detention_refused", date="2024-03-02"), ruling("detention_ordered", date="2024-02-01"), ruling("detention_ordered")]
    assert outcome(rulings, "detention_at_first_appearance") == (True, "detention_ordered", "2024-02-01")


def test_last_counts_the_latest_ruling():
    rulings = [ruling("verdict_conviction", date="2024-05-01"), ruling("verdict_acquittal", date="2024-09-01")]
    assert outcome(rulings, "conviction") == (False, "verdict_acquittal", "2024-09-01")


def test_any_counts_a_case_with_one_closed_hearing():
    rulings = [ruling("hearing_public", date="2024-01-01"), ruling("hearing_closed", date="2024-02-01")]
    assert outcome(rulings, "hearing_closed_any") == (True, "hearing_closed", "2024-02-01")


def test_all_needs_every_motion_denied():
    denied, granted = ruling("defense_motion_denied", date="2024-01-01"), ruling("defense_motion_granted", date="2024-02-01")
    assert outcome([denied, granted], "defense_motions_all_denied") == (False, "defense_motion_granted", "2024-02-01")
    assert outcome([denied], "defense_motions_all_denied")[0] is True


def test_a_case_without_the_rate_codes_is_not_counted():
    assert outcome([ruling("verdict_conviction")], "detention_at_first_appearance") is None


def history(judge_counts, *, court="Test District Court", charge="Test charge", start=0):
    """Cases with one detention ruling each: {judge: (ordered, refused)}."""
    records, number = [], start
    for judge, (ordered, refused) in judge_counts.items():
        for code in ["detention_ordered"] * ordered + ["detention_refused"] * refused:
            number += 1
            records.append(make_history_case(f"h{number:02d}", [ruling(code, judge, "2024-01-10")], court=court, charge_type=charge))
    return records


def detention(report, name):
    profile = next(p for p in report.profiles if p.display_name == name)
    return profile, next(i for i in profile.indicators if i.rate_id == DETENTION.id)


def test_baseline_is_the_other_judges_of_the_same_court_and_charge_type_only():
    records = history({"Ilena Varda": (6, 0), "Petra Hollin": (1, 2), "Anwar Selik": (0, 3)})
    records += history({"Other Judge": (0, 9)}, court="Other Court", start=50)
    records += history({"Charge Judge": (0, 9)}, charge="Other charge", start=70)
    profile, indicator = detention(judges.run(records, {}, (), CONFIG), "Ilena Varda")
    assert (indicator.judge.k, indicator.judge.n) == (6, 6)
    assert (indicator.baseline.k, indicator.baseline.n, indicator.baseline_judges) == (1, 6, 2)
    assert [i.rate_id for i in profile.indicators] == [DETENTION.id]  # no rulings for the other rates
    assert profile.k_compared == 1


def test_indicator_is_hidden_below_the_minimum_and_carries_no_rate():
    report = judges.run(history({"Ilena Varda": (4, 0), "Petra Hollin": (1, 9)}), {}, (), CONFIG)
    _, indicator = detention(report, "Ilena Varda")
    assert not indicator.shown and indicator.judge is None and not indicator.outcomes
    assert indicator.message == "Not shown: fewer than 5 cases (4)."
    _, other = detention(report, "Petra Hollin")
    assert not other.shown and other.message == "Not shown: the baseline has fewer than 5 cases (4)."


def test_pattern_fires_only_when_the_difference_interval_excludes_zero():
    fires = judges.run(history({"Ilena Varda": (7, 0), "Petra Hollin": (1, 5)}), {}, (), CONFIG)
    profile, indicator = detention(fires, "Ilena Varda")
    assert indicator.pattern and indicator.difference_ci[0] > 0
    [flag] = profile.flags
    assert flag.id == indicator.flag_id and flag.status == "pattern_warrants_review" and flag.case_id is None
    assert flag.message.startswith("Pattern that warrants review: ") and "n=7 cases" in flag.message and "n=6 cases" in flag.message
    assert len(flag.evidence) == 7 and {e.role for e in flag.evidence} == {"ruling"}
    quiet = judges.run(history({"Ilena Varda": (5, 2), "Petra Hollin": (2, 4)}), {}, (), CONFIG)
    profile, indicator = detention(quiet, "Ilena Varda")
    assert indicator.shown and not indicator.pattern and not profile.flags
    assert indicator.message.startswith("Not distinguishable from the baseline at this sample size")


def test_pending_names_count_neither_for_the_judge_nor_in_the_baseline():
    records = history({"Ilena Varda": (6, 0), "Petra Hollin": (1, 5)})
    records.append(make_history_case("x1", [ruling("detention_refused", "I. Varda", "2024-01-10")]))
    report = judges.run(records, {}, (), CONFIG)
    profile, indicator = detention(report, "Ilena Varda")
    assert (indicator.judge.n, indicator.baseline.n) == (6, 6)
    assert [p.case_id for p in profile.pending] == ["x1"] == [c.case_id for c in report.manual_confirmations]
    confirmed = judges.run(records, {}, [AliasDecision(case_id="x1", name="I. Varda", same_as="Ilena Varda")], CONFIG)
    _, indicator = detention(confirmed, "Ilena Varda")
    assert (indicator.judge.k, indicator.judge.n) == (6, 7)


def test_unknown_ruling_codes_are_reported_not_silently_dropped():
    records = history({"Ilena Varda": (1, 0)}) + [make_history_case("x1", [ruling("detention_orderd")])]
    report = judges.run(records, {}, (), CONFIG)
    assert any("detention_orderd" in note.text and "x1" in note.text and note.synthetic for note in report.notes)


def two_judges(case_id, first, later, first_code="detention_refused"):
    """A case whose first appearance was held by one judge and whose detention another judge extended."""
    rulings = [ruling(first_code, first, "2024-01-10"), ruling("detention_ordered", later, "2024-03-10")]
    return make_history_case(case_id, rulings)


def test_first_appearance_is_credited_only_to_the_judge_who_held_it():
    records = [two_judges(f"t{n}", "Alpha Orren", "Beta Kask") for n in range(5)]
    records += history({"Gamma Lune": (2, 4)}, start=10)
    report = judges.run(records, {}, (), CONFIG)
    beta = next(p for p in report.profiles if p.display_name == "Beta Kask")
    assert DETENTION.id not in {i.rate_id for i in beta.indicators}  # Beta held no first appearance
    _, alpha = detention(report, "Alpha Orren")
    assert (alpha.judge.k, alpha.judge.n) == (0, 5)
    assert (alpha.baseline.k, alpha.baseline.n, alpha.baseline_judges) == (2, 6, 1)
    _, gamma = detention(report, "Gamma Lune")
    assert (gamma.baseline.k, gamma.baseline.n) == (0, 5)


def test_a_case_whose_first_ruling_awaits_confirmation_counts_nowhere():
    records = history({"Ilena Varda": (5, 0), "Petra Hollin": (1, 4)})
    records += [two_judges(f"p{n}", "I. Varda", "Petra Hollin") for n in range(3)]  # the pending name refused first
    report = judges.run(records, {}, (), CONFIG)
    _, varda = detention(report, "Ilena Varda")
    _, hollin = detention(report, "Petra Hollin")
    assert (varda.baseline.k, varda.baseline.n) == (1, 5)  # not 4 of 8: Hollin only extended in p0-p2
    assert (hollin.judge.k, hollin.judge.n) == (1, 5)


def test_document_order_places_undated_rulings_when_it_agrees_with_the_dates():
    undated_first = [ruling("detention_refused"), ruling("detention_ordered", date="2024-03-10")]  # in one summary, in this order
    records = [make_history_case(f"u{n}", undated_first) for n in range(5)] + history({"Petra Hollin": (0, 5)}, start=10)
    report = judges.run(records, {}, (), CONFIG)
    _, varda = detention(report, "Ilena Varda")
    assert (varda.judge.k, varda.judge.n) == (0, 5)  # refused at the first appearance, as the summary reads
    assert not [note for note in report.notes if "order" in note.text]
    agreeing = [ruling("detention_ordered"), ruling("detention_ordered", date="2024-03-10")]
    assert outcome(agreeing, "detention_at_first_appearance") == (True, "detention_ordered", None)


def test_a_case_whose_rulings_cannot_be_put_in_order_is_left_out_with_a_note():
    # the dated rulings are not in date order in the summary, so it cannot place the undated one
    tangled = [ruling("detention_refused", date="2024-03-10"), ruling("detention_ordered"), ruling("detention_refused", date="2024-01-10")]
    records = [make_history_case(f"u{n}", tangled) for n in range(5)] + history({"Petra Hollin": (0, 5)}, start=10)
    report = judges.run(records, {}, (), CONFIG)
    varda = next(p for p in report.profiles if p.display_name == "Ilena Varda")
    assert DETENTION.id not in {i.rate_id for i in varda.indicators}
    unknown = [note for note in report.notes if "order" in note.text]
    assert sorted(note.text.split()[1].rstrip(":") for note in unknown) == [f"u{n}" for n in range(5)]


def test_a_sentence_ruling_without_a_value_is_reported():
    records = [make_history_case("s1", [ruling("verdict_conviction", date="2024-05-01"), ruling("sentence_months", date="2024-05-01")])]
    report = judges.run(records, {}, (), CONFIG)
    assert any("s1" in note.text and "sentence_months" in note.text and "value" in note.text for note in report.notes)


def test_case_signals_link_other_module_outputs_without_attributing_them():
    records = history({"Ilena Varda": (1, 0)})
    analysis = CaseAnalysis(case_id="h01", reuse=ReuseResult(judgment_doc_id=None, indictment_doc_id=None, score=0.6))
    [profile] = judges.run(records, {"h01": analysis}, (), CONFIG).profiles
    [signal] = profile.case_signals
    assert (signal.case_id, signal.reuse_score, signal.flags) == ("h01", 0.6, 0)


# --- the synthetic demo history (data/demo/history plus the demo case) -------------------------


@pytest.fixture(scope="module")
def demo():
    records = load_history() + (load_case(DEMO_CASE_DIR),)
    return records, judges.run(records, {}, load_alias_decisions(), CONFIG)


def test_demo_judge_has_one_pattern_two_plain_and_two_hidden_indicators(demo):
    _, report = demo
    profile = next(p for p in report.profiles if p.display_name == "Ilena Varda" and p.court.startswith("Mirevo"))
    assert len(profile.case_ids) == 9 and profile.synthetic
    assert set(profile.name_variants) == {"Ilena Varda", "Iléna Várda", "Judge Ilena Varda", "Hon. Ilena Varda"}
    table = {i.rate_id: (i.shown, i.pattern, i.judge and (i.judge.k, i.judge.n), i.baseline and (i.baseline.k, i.baseline.n)) for i in profile.indicators}
    assert table == {
        "detention_at_first_appearance": (True, True, (9, 9), (4, 10)),
        "conviction": (True, False, (8, 9), (7, 10)),
        "hearing_closed_any": (True, False, (5, 9), (2, 10)),
        "contested_evidence_admitted_any": (False, False, None, None),
        "defense_motions_all_denied": (False, False, None, None),
    }
    assert profile.k_compared == 3
    [sentences] = profile.descriptive
    assert sentences.shown and sorted(sentences.values) == [24, 30, 30, 36, 36, 42, 48, 48]


def test_demo_registry_keeps_the_initials_case_and_the_other_court_apart(demo):
    _, report = demo
    [pending] = report.manual_confirmations
    assert (pending.case_id, pending.raw_name, pending.candidate_name) == ("hist-initials-01", "I. Varda", "Ilena Varda")
    elsewhere = [p for p in report.profiles if p.display_name == "Ilena Varda" and p.court == "Port Elsin Regional Court"]
    assert [p.case_ids for p in elsewhere] == [("hist-port-elsin-01",)]
    assert not any(i.shown for i in elsewhere[0].indicators)


def test_demo_baseline_judges_show_no_pattern(demo):
    _, report = demo
    for name in ("Petra Hollin", "Anwar Selik"):
        profile = next(p for p in report.profiles if p.display_name == name)
        assert not profile.flags and not any(i.pattern for i in profile.indicators)


def test_demo_flags_rest_on_exact_rulings_across_cases(demo):
    records, report = demo
    resolve = resolver_for(records)
    flags = [flag for profile in report.profiles for flag in profile.flags]
    assert len(flags) == 1
    [flag] = flags
    assert len({e.span.case_id for e in flag.evidence}) == 9
    assert all(span_is_valid(e.span, resolve) for e in flag.evidence)


def test_no_judge_output_characterises_a_person_and_every_one_states_its_sample_size(demo):
    _, report = demo
    block_list = CONFIG.messages.block_list
    for profile in report.profiles:
        for indicator in profile.indicators:
            assert not contains_blocked_term(indicator.message, block_list), indicator.message
            assert "cases" in indicator.message  # the sample size, also when hidden
        for flag in profile.flags:
            assert not contains_blocked_term(flag.message, block_list)


# --- provenance (hard rule 2): rulings and judge flags are checked against the source text -------


def tampered(records, case_id):
    """The same records, with one case's summary text changed after its rulings were coded."""
    changed = []
    for record in records:
        if record.case_id == case_id:
            document = record.documents[0]
            text = document.text.replace("Ruling 1:", "Ruling X:")
            record = record.model_copy(update={"documents": (document.model_copy(update={"text": text}),)})
        changed.append(record)
    return changed


def test_a_ruling_whose_quote_is_not_in_its_document_never_counts():
    records = tampered(history({"Ilena Varda": (7, 0), "Petra Hollin": (1, 5)}), "h01")
    report = analyze_judges(records, {}, (), CONFIG)
    _, indicator = detention(report, "Ilena Varda")
    assert (indicator.judge.k, indicator.judge.n) == (6, 6)
    assert any("h01" in note.text and "detention_ordered" in note.text for note in report.notes)


def test_a_pattern_whose_evidence_fails_the_check_is_dropped_and_not_shown():
    records = history({"Ilena Varda": (7, 0), "Petra Hollin": (1, 5)})
    report = judges.run(records, {}, (), CONFIG)
    dropped: list[str] = []
    checked = check_judges(report, resolver_for(tampered(records, "h01")), dropped)
    profile, indicator = detention(checked, "Ilena Varda")
    assert not profile.flags and len(dropped) == 1 and "h01/summary.txt" in dropped[0]
    assert not indicator.shown and not indicator.pattern and indicator.judge is None
    assert "not found in its source" in indicator.message


def test_analyze_judges_reports_no_drops_on_the_demo_history(demo):
    records, _ = demo
    report = analyze_judges(records, {}, load_alias_decisions(), CONFIG)
    assert report.dropped_flags == 0 and not report.dropped_reasons and not report.notes
    assert sum(len(p.flags) for p in report.profiles) == 1


def test_alias_decisions_must_be_utf8_and_valid(tmp_path):
    from ratio.extraction.loader import LoaderError

    path = tmp_path / "alias_decisions.yaml"
    path.write_bytes('synthetic: true\ndecisions:\n  - {case_id: c1, name: "I. Varda", same_as: "Iléna Várda"}\n'.encode("cp1252"))
    with pytest.raises(LoaderError, match="UTF-8"):
        load_alias_decisions(path)
    path.write_text("synthetic: true\ndecisions:\n  - {case_id: c1}\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="invalid"):
        load_alias_decisions(path)
    assert load_alias_decisions(tmp_path / "absent.yaml") == ()


def test_each_decisions_file_carries_its_own_kind_of_data(tmp_path):
    path = tmp_path / "alias_decisions.yaml"
    path.write_text("synthetic: false\ndecisions:\n  - {case_id: pub-02, name: M. Tull, same_as: Maren Tull, synthetic: true}\n", encoding="utf-8")
    [decision] = load_alias_decisions(path)
    assert decision.synthetic is False  # the file decides, never the entry
    [demo] = [d for d in load_alias_decisions()] or [None]
    assert demo is None  # the demo file records no decision
