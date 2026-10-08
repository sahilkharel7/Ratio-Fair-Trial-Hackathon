"""Evaluate Ratio on the synthetic demo case.

    python -m eval.run_eval          replay the committed model cache (no Ollama needed)
    python -m eval.run_eval --live   call the local Ollama model on cache misses

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
from ratio.evaluation import event_violations, extraction_metrics
from ratio.expected import ExpectedFlags, GoldTimeline
from ratio.extraction.build import load_case
from ratio.paths import DEMO_CASE_DIR, GOLD_DIR, REPO_ROOT
from ratio.pipeline import demo_llm, ingest, make_llm

OUT_DIR = REPO_ROOT / "eval" / "out"


def _load_gold() -> tuple[GoldTimeline, ExpectedFlags]:
    timeline = GoldTimeline.model_validate(json.loads((GOLD_DIR / "gold_timeline.json").read_text(encoding="utf-8")))
    expected = ExpectedFlags.model_validate(json.loads((GOLD_DIR / "expected_flags.json").read_text(encoding="utf-8")))
    return timeline, expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="call the local Ollama model on cache misses")
    args = parser.parse_args(argv)
    netguard.install()

    config = default_config()
    llm = make_llm(config, mode="live") if args.live else demo_llm(config)
    record, report = ingest(load_case(DEMO_CASE_DIR), llm, config)
    timeline, expected = _load_gold()
    metrics = extraction_metrics(record, timeline)
    violations = event_violations(record, expected)

    print("Ratio evaluation: SYNTHETIC demo case (in-sample, one team-written case)")
    print(f"Model {llm.model} ({'replayed from cache' if llm.replay_only else 'live'}); cache misses {llm.stats.misses}")
    print(f"Extraction: {report.events_kept} events, {report.arguments_kept} arguments, {len(report.dropped)} dropped")
    print(f"  event recall   {metrics.matched}/{metrics.gold_events} = {metrics.event_recall:.0%}")
    print(f"  date accuracy  {metrics.correct_dates}/{metrics.gold_events} = {metrics.date_accuracy:.0%}")
    for item in metrics.missed:
        print(f"  missed: {item}")
    for item in metrics.wrong_dates:
        print(f"  wrong date: {item}")
    for item in metrics.conflicting_types:
        print(f"  conflicting dates for: {item}")
    for item in violations:
        print(f"  must-not violation: {item}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "synthetic": True,
        "in_sample": True,
        "model": llm.model,
        "replayed": llm.replay_only,
        "extraction": {**asdict(metrics), "event_recall": metrics.event_recall, "date_accuracy": metrics.date_accuracy},
        "dropped": list(report.dropped),
        "event_violations": list(violations),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
