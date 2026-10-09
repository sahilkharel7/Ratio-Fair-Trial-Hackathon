"""Ratio command line. Everything runs locally; the only network peer is the local Ollama server.

    python -m ratio ingest [CASE_DIR] [--live]   build a case record and save it to the store
    python -m ratio report [CASE_DIR] [-o PATH]  write the case's findings as a Markdown report draft
    python -m ratio feedback [--json PATH]       what reviewers' decisions say about each module
    python -m ratio build-demo-cache [--fresh]   record the demo's model calls with Ollama
    python -m ratio check-ollama                 check the local Ollama server and model
    python -m ratio preflight                    check this computer is ready for the offline demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ratio import netguard, preflight
from ratio.config import ConfigError, default_config
from ratio.extraction.build import load_case
from ratio.extraction.loader import LoaderError
from ratio.feedback import case_findings, summarise
from ratio.history import load_all_alias_decisions
from ratio.history import load_alias_decisions, load_history
from ratio.llm import LLMError, OllamaClient
from ratio.paths import DEMO_CASE_DIR
from ratio.pipeline import analysis_context, analyze, analyze_judges, build_demo_cache, demo_llm, ingest, make_llm
from ratio.report import draft_report
from ratio.store import CaseStore


def _progress(label: str, done: int, total: int) -> None:
    print(f"  [{done}/{total}] {label}", flush=True)


def _llm_for(case_dir: Path, live: bool, config):
    if live:
        return make_llm(config, mode="live")
    if case_dir.resolve() == DEMO_CASE_DIR.resolve():
        return demo_llm(config)
    return make_llm(config, mode="replay")


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = default_config()
    case_dir = Path(args.case_dir)
    base = load_case(case_dir)
    llm = _llm_for(case_dir, args.live, config)
    mode = "replay" if llm.replay_only else "live"
    print(f"Ingesting {base.case_id} ({len(base.documents)} documents, model {llm.model}, {mode})")
    record, report = ingest(base, llm, config, progress=_progress)
    CaseStore().save_case(record)
    print(
        f"Done: {report.events_kept} events, {report.arguments_kept} arguments, {len(record.observations)} observations; "
        f"{len(report.dropped)} model items dropped, {report.reviews} marked for review; "
        f"cache hits {llm.stats.hits}, misses {llm.stats.misses}"
    )
    for reason in report.dropped:
        print(f"  dropped: {reason}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    config = default_config()
    case_dir = Path(args.case_dir)
    llm = _llm_for(case_dir, args.live, config)
    record, _ = ingest(load_case(case_dir), llm, config)
    analysis = analyze(record, analysis_context(config, llm))
    history = [r for r in load_history() if r.case_id != record.case_id]
    judges = analyze_judges([*history, record], {record.case_id: analysis}, load_alias_decisions(), config)
    store = CaseStore()  # decisions reviewers recorded on this case in the app
    report = draft_report(
        record, analysis, judges, config,
        history=history, reviews=store.reviews(record.case_id), missed=store.missed_issues(record.case_id),
    )  # fmt: skip
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report draft for {record.case_id} written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(report)
    return 0


def _cmd_feedback(args: argparse.Namespace) -> int:
    """Reviewers' decisions over every stored case with an analysis, per module."""
    config, store = default_config(), CaseStore()
    records = store.all_records()
    analyses = {r.case_id: a for r in records if (a := store.load_analysis(r.case_id)) is not None}
    judges = analyze_judges(records, analyses, load_all_alias_decisions(), config) if analyses else None
    findings = {case_id: case_findings(analysis, judges) for case_id, analysis in analyses.items()}
    reviews, missed = store.reviews(), store.missed_issues()
    summary = summarise(findings, reviews, missed)
    labels = config.messages
    print(f"Reviewers' decisions: {summary.reviewed} of {summary.findings} findings reviewed, over {summary.cases} analysed cases")
    print("  (measured on the cases reviewed so far: a prompt for improving Ratio, not a general accuracy figure)")
    for row in summary.modules:
        share = "n/a" if row.kept_share is None else f"{row.kept_share:.0%}"
        print(
            f"  {labels.label('module_' + row.module):<16} {row.reviewed:>3}/{row.findings:<3} reviewed; "
            f"{row.accepted} accepted, {row.edited} reworded, {row.rejected} rejected (kept {share}); {row.missed} missed issues"
        )
    for review in summary.rejections:
        print(f"  rejected ({review.case_id}, {review.flag.standard_label}): {review.note}")
    if summary.stale:
        print(f"  {len(summary.stale)} decisions concern findings the current analyses no longer produce")
    if args.json:
        export = {
            "summary": {**summary.model_dump(exclude={"stale", "rejections"}), "stale": [r.flag_id for r in summary.stale]},
            "reviews": [r.model_dump(mode="json") for r in reviews],
            "missed_issues": [m.model_dump(mode="json") for m in missed],
        }
        Path(args.json).write_text(json.dumps(export, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Every decision and missed issue written to {args.json}")
    return 0


def _cmd_build_demo_cache(args: argparse.Namespace) -> int:
    print("Running the demo case's model calls with the local Ollama model (cached replies are reused unless --fresh)...")
    manifest = build_demo_cache(default_config(), fresh=args.fresh, progress=_progress)
    print(f"Demo cache: {manifest.entries} replies from {manifest.model} (Ollama {manifest.ollama_version})")
    return 0


def _cmd_check_ollama(_: argparse.Namespace) -> int:
    client = OllamaClient(default_config().settings.llm)
    print(f"Ollama {client.version()} on loopback; model {client.model}")
    client.check_model()
    print("Model is pulled and local.")
    return 0


def _cmd_preflight(_: argparse.Namespace) -> int:
    checks = preflight.run_all(default_config())
    print(preflight.format_report(checks))
    return preflight.exit_code(checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ratio", description="Ratio (offline fair-trial monitoring analysis)")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="build a case record and save it to the store")
    ingest_parser.add_argument("case_dir", nargs="?", default=str(DEMO_CASE_DIR))
    ingest_parser.add_argument("--live", action="store_true", help="call the local Ollama model on cache misses")
    report_parser = commands.add_parser("report", help="write the case's findings as a Markdown report draft")
    report_parser.add_argument("case_dir", nargs="?", default=str(DEMO_CASE_DIR))
    report_parser.add_argument("-o", "--output", help="file to write (default: standard output)")
    report_parser.add_argument("--live", action="store_true", help="call the local Ollama model on cache misses")
    feedback_parser = commands.add_parser("feedback", help="what reviewers' decisions say about each module")
    feedback_parser.add_argument("--json", help="also write every decision and missed issue to this file")
    rebuild = commands.add_parser("build-demo-cache", help="record the demo's model calls with the local model")
    rebuild.add_argument("--fresh", action="store_true", help="ask every call again instead of reusing recorded replies")
    commands.add_parser("check-ollama", help="check the local Ollama server and model")
    commands.add_parser("preflight", help="check this computer is ready for the offline demo")
    args = parser.parse_args(argv)

    netguard.install()
    handlers = {"ingest": _cmd_ingest, "report": _cmd_report, "feedback": _cmd_feedback, "build-demo-cache": _cmd_build_demo_cache, "check-ollama": _cmd_check_ollama, "preflight": _cmd_preflight}
    try:
        return handlers[args.command](args)
    except (LLMError, LoaderError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
