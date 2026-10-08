"""Evaluate Ratio on the synthetic demo case.

    python -m eval.run_eval          replay the committed model cache (no Ollama needed)
    python -m eval.run_eval --live   call the local Ollama model on cache misses

Reports extraction quality against the hand-checked timeline, the expected outputs found and
missed (expected_flags.json), must-not-flag violations, the judge profile built from the synthetic
history of the demo court, and provenance before and after enforcement for every flag, the judge
patterns included. Exits 1 if any flag lacked an exact source span, shown or dropped (hard rule 2).

Results are in-sample: one synthetic case written by the team, so they show the pipeline
works end to end, not how it generalises. Writes eval/out/report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from ratio import netguard
from ratio.config import default_config
from ratio.evaluation import FlagRecall, event_violations, extraction_metrics, flag_recall
from ratio.expected import ExpectedFlags, GoldTimeline
from ratio.extraction.build import load_case
from ratio.history import load_alias_decisions, load_history
from ratio.paths import DEMO_CASE_DIR, GOLD_DIR, REPO_ROOT
from ratio.pipeline import analysis_context, analyze, analyze_judges, demo_llm, ingest, make_llm
from ratio.provenance import resolver_for, span_is_valid
from ratio.results import CaseAnalysis, JudgeReport
from ratio.schema import CaseRecord, Flag

OUT_DIR = REPO_ROOT / "eval" / "out"


def _load_gold() -> tuple[GoldTimeline, ExpectedFlags]:
    timeline = GoldTimeline.model_validate(json.loads((GOLD_DIR / "gold_timeline.json").read_text(encoding="utf-8")))
    expected = ExpectedFlags.model_validate(json.loads((GOLD_DIR / "expected_flags.json").read_text(encoding="utf-8")))
    return timeline, expected


def _provenance(shown: tuple[Flag, ...], dropped: int, records: list[CaseRecord]) -> dict[str, float | int]:
    """Share of flags whose spans resolved before enforcement, and of shown flags that resolve now."""
    resolve = resolver_for(records)
    exact = sum(all(span_is_valid(span, resolve) for span in flag.spans) for flag in shown)
    produced = len(shown) + dropped
    return {
        "flags_produced": produced,
        "flags_dropped": dropped,
        "flags_shown": len(shown),
        "before_enforcement": len(shown) / produced if produced else 1.0,
        "shown_with_exact_spans": exact / len(shown) if shown else 1.0,
    }


def provenance(record: CaseRecord, analysis: CaseAnalysis) -> dict[str, float | int]:
    return _provenance(analysis.all_flags(), analysis.dropped_flags, [record])


def judge_provenance(records: list[CaseRecord], report: JudgeReport) -> dict[str, float | int]:
    """The judge patterns rest on coded rulings in many cases, so their spans resolve across cases."""
    shown = tuple(flag for profile in report.profiles for flag in profile.flags)
    return _provenance(shown, report.dropped_flags, records)


def _print_judges(report: JudgeReport, case_id: str, prov: dict[str, float | int]) -> None:
    waiting = len(report.manual_confirmations)
    names = "1 name" if waiting == 1 else f"{waiting} names"
    print(f"Judge profiles (synthetic history of the demo court, in-sample): {len(report.profiles)} judges, {names} awaiting confirmation")
    for profile in (p for p in report.profiles if case_id in p.case_ids):
        print(f"  {profile.display_name}, {profile.court}: {len(profile.case_ids)} cases, {profile.k_compared} indicators compared")
        for indicator in profile.indicators:
            if not indicator.shown:
                print(f"    {indicator.label}: {indicator.message}")
                continue
            state = "pattern that warrants review" if indicator.pattern else "not distinguishable"
            judge, baseline = indicator.judge, indicator.baseline
            print(f"    {indicator.label}: {judge.k}/{judge.n} against {baseline.k}/{baseline.n} in the baseline, {state}")
    print(f"  judge patterns: {prov['flags_shown']} shown, {prov['flags_dropped']} dropped; shown with exact spans {prov['shown_with_exact_spans']:.0%}")
    for note in report.notes:
        print(f"  note: {note.text}")


def _print_flags(recall: FlagRecall, prov: dict[str, float | int]) -> None:
    print(f"Flags: {prov['flags_shown']} shown, {prov['flags_dropped']} dropped by provenance enforcement")
    print(f"  provenance before enforcement {prov['before_enforcement']:.0%}; shown flags with exact spans {prov['shown_with_exact_spans']:.0%}")
    print(f"  expected outputs found {len(recall.found)}/{len(recall.found) + len(recall.missed)} = {recall.recall:.0%}")
    for item in recall.missed:
        print(f"  missed expected output: {item}")
    for item in recall.violations:
        print(f"  must-not-flag violation: {item}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="call the local Ollama model on cache misses")
    args = parser.parse_args(argv)
    netguard.install()

    config = default_config()
    llm = make_llm(config, mode="live") if args.live else demo_llm(config)
    record, report = ingest(load_case(DEMO_CASE_DIR), llm, config)
    analysis = analyze(record, analysis_context(config, llm))
    timeline, expected = _load_gold()
    metrics = extraction_metrics(record, timeline)
    violations = event_violations(record, expected)
    recall = flag_recall(record, analysis, expected)
    prov = provenance(record, analysis)
    history = [*load_history(), record]
    judges = analyze_judges(history, {record.case_id: analysis}, load_alias_decisions(), config)
    judge_prov = judge_provenance(history, judges)

    print("Ratio evaluation: SYNTHETIC demo case (in-sample, one team-written case)")
    replayed = llm.stats.misses == 0
    source = "replayed from cache" if llm.replay_only else f"live, {llm.stats.hits} answers from cache, {llm.stats.misses} new"
    print(f"Model {llm.model} ({source})")
    print(f"Extraction: {report.events_kept} events, {report.arguments_kept} arguments, {len(report.dropped)} dropped")
    print(f"  event recall   {metrics.matched}/{metrics.gold_events} = {metrics.event_recall:.0%}")
    print(f"  date accuracy  {metrics.correct_dates}/{metrics.matched} of the events found = {metrics.date_accuracy:.0%}")
    for item in metrics.missed:
        print(f"  missed: {item}")
    for item in metrics.wrong_dates:
        print(f"  wrong date: {item}")
    for item in metrics.conflicting_types:
        print(f"  conflicting dates for: {item}")
    for item in violations:
        print(f"  must-not violation: {item}")
    _print_flags(recall, prov)
    _print_judges(judges, record.case_id, judge_prov)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "synthetic": True,
        "in_sample": True,
        "model": llm.model,
        "replayed": replayed,
        "cache_hits": llm.stats.hits,
        "cache_misses": llm.stats.misses,
        "extraction": {**asdict(metrics), "event_recall": metrics.event_recall, "date_accuracy": metrics.date_accuracy},
        "dropped": list(report.dropped),
        "event_violations": list(violations),
        "flags": {**asdict(recall), "recall": recall.recall, "dropped_reasons": list(analysis.dropped_reasons)},
        "provenance": prov,
        "judges": {
            "profiles": len(judges.profiles),
            "awaiting_confirmation": [c.case_id for c in judges.manual_confirmations],
            "patterns": [flag.message for profile in judges.profiles for flag in profile.flags],
            "notes": [note.text for note in judges.notes],
            "provenance": judge_prov,
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sourced = all(p["shown_with_exact_spans"] == 1.0 and p["flags_dropped"] == 0 for p in (prov, judge_prov))
    return 0 if sourced else 1


if __name__ == "__main__":
    sys.exit(main())
