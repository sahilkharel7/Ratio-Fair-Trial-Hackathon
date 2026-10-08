"""Ratio command line. Everything runs locally; the only network peer is the local Ollama server.

    python -m ratio ingest [CASE_DIR] [--live]   build a case record and save it to the store
    python -m ratio build-demo-cache             re-run the demo's model calls with Ollama
    python -m ratio check-ollama                 check the local Ollama server and model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ratio import netguard
from ratio.config import ConfigError, default_config
from ratio.extraction.build import load_case
from ratio.extraction.loader import LoaderError
from ratio.llm import LLMError, OllamaClient
from ratio.paths import DEMO_CASE_DIR
from ratio.pipeline import build_demo_cache, demo_llm, ingest, make_llm
from ratio.store import CaseStore


def _progress(label: str, done: int, total: int) -> None:
    print(f"  [{done}/{total}] {label}", flush=True)


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = default_config()
    case_dir = Path(args.case_dir)
    base = load_case(case_dir)
    is_demo = case_dir.resolve() == DEMO_CASE_DIR.resolve()
    if args.live:
        llm = make_llm(config, mode="live")
    elif is_demo:
        llm = demo_llm(config)
    else:
        llm = make_llm(config, mode="replay")
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


def _cmd_build_demo_cache(_: argparse.Namespace) -> int:
    print("Running every model call for the demo case with the local Ollama model...")
    manifest = build_demo_cache(default_config(), progress=_progress)
    print(f"Demo cache: {manifest.entries} replies from {manifest.model} (Ollama {manifest.ollama_version})")
    return 0


def _cmd_check_ollama(_: argparse.Namespace) -> int:
    client = OllamaClient(default_config().settings.llm)
    print(f"Ollama {client.version()} on loopback; model {client.model}")
    client.check_model()
    print("Model is pulled and local.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ratio", description="Ratio (offline fair-trial monitoring analysis)")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="build a case record and save it to the store")
    ingest_parser.add_argument("case_dir", nargs="?", default=str(DEMO_CASE_DIR))
    ingest_parser.add_argument("--live", action="store_true", help="call the local Ollama model on cache misses")
    commands.add_parser("build-demo-cache", help="re-run the demo's model calls with the local model")
    commands.add_parser("check-ollama", help="check the local Ollama server and model")
    args = parser.parse_args(argv)

    netguard.install()
    handlers = {"ingest": _cmd_ingest, "build-demo-cache": _cmd_build_demo_cache, "check-ollama": _cmd_check_ollama}
    try:
        return handlers[args.command](args)
    except (LLMError, LoaderError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
