"""Orchestration shared by the CLI, the app and the eval.

The demo replays every model call from the committed cache (data/demo/cache/llm), so it runs
in seconds with Ollama stopped. Live mode calls the local Ollama server on cache misses and
writes new replies to the runtime cache (data/cache/llm, never committed).
"""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ratio.config import RatioConfig
from ratio.context import AnalysisContext, Embedder
from ratio.embeddings import MiniLMEmbedder
from ratio.extraction.build import load_case
from ratio.extraction.extract import ExtractionReport, extract_record
from ratio.llm import CachedLLM, OllamaClient, ResponseCache, model_from_env
from ratio.modules import absence, clock, reuse
from ratio.paths import DEMO_CACHE_DIR, DEMO_CASE_DIR, RUNTIME_CACHE_DIR
from ratio.provenance import DocResolver, enforce, filter_follow_up, resolver_for
from ratio.results import CaseAnalysis, ReuseResult
from ratio.schema import Argument, CaseRecord, Event, Flag, Frozen

LLMMode = Literal["live", "replay"]
ProgressCallback = Callable[[str, int, int], None]

DEMO_LLM_CACHE = DEMO_CACHE_DIR / "llm"
DEMO_CACHE_MANIFEST = DEMO_CACHE_DIR / "manifest.json"


class DemoCacheManifest(Frozen):
    synthetic: bool
    model: str
    ollama_version: str
    built_at: str
    entries: int
    documents: dict[str, str]


def make_llm(
    config: RatioConfig,
    *,
    mode: LLMMode,
    write_dir=None,
    read_dirs=(DEMO_LLM_CACHE,),
    model: str | None = None,
) -> CachedLLM:
    settings = config.settings.llm
    name = model or model_from_env(settings)
    cache = ResponseCache(write_dir or RUNTIME_CACHE_DIR, read_dirs=read_dirs)
    client = OllamaClient(settings, model=name) if mode == "live" else None
    return CachedLLM(client, cache, settings=settings, model=name)


def read_demo_manifest() -> DemoCacheManifest | None:
    if not DEMO_CACHE_MANIFEST.is_file():
        return None
    return DemoCacheManifest.model_validate(json.loads(DEMO_CACHE_MANIFEST.read_text(encoding="utf-8")))


def demo_llm(config: RatioConfig) -> CachedLLM:
    """Replay-only access to the committed demo cache, with the model it was built with."""
    manifest = read_demo_manifest()
    model = manifest.model if manifest else model_from_env(config.settings.llm)
    return make_llm(config, mode="replay", write_dir=DEMO_LLM_CACHE, read_dirs=(), model=model)


def ingest(
    base: CaseRecord, llm: CachedLLM, config: RatioConfig, *, progress: ProgressCallback | None = None
) -> tuple[CaseRecord, ExtractionReport]:
    """Add model-extracted events and arguments to a deterministic base record."""
    return extract_record(base, llm, config.settings.extraction, progress=progress)


def analysis_context(config: RatioConfig, llm: CachedLLM, embedder: Embedder | None = None) -> AnalysisContext:
    return AnalysisContext(llm=llm, embedder=embedder or MiniLMEmbedder(), config=config)


def _keep_valid(flags: tuple[Flag, ...], resolve: DocResolver, dropped: list[str]) -> tuple[Flag, ...]:
    outcome = enforce(flags, resolve)
    dropped.extend(f"{flag.module}/{flag.standard_id}: {reason}" for flag, reason in outcome.dropped)
    return outcome.kept


def _reuse_with_valid_flags(result: ReuseResult, resolve: DocResolver, dropped: list[str]) -> ReuseResult:
    """Keep only flags whose spans resolve, and only the pairs and argument checks those flags back."""
    flags = _keep_valid(result.flags, resolve, dropped)
    kept = {flag.id for flag in flags}
    return result.model_copy(
        update={
            "flags": flags,
            "pairs": tuple(pair for pair in result.pairs if pair.flag_id in kept),
            "arguments": tuple(check for check in result.arguments if check.flag_id is None or check.flag_id in kept),
        }
    )


def analyze(record: CaseRecord, ctx: AnalysisContext) -> CaseAnalysis:
    """Run the modules on one case and enforce provenance on every flag (hard rule 2)."""
    resolve = resolver_for([record])
    dropped: list[str] = []
    absence_result = absence.run(record, ctx)
    absence_result = absence_result.model_copy(
        update={
            "flags": _keep_valid(absence_result.flags, resolve, dropped),
            "follow_ups": tuple(filter_follow_up(f, resolve) for f in absence_result.follow_ups),
        }
    )
    clock_result = clock.run(record, ctx)
    clock_result = clock_result.model_copy(update={"flags": _keep_valid(clock_result.flags, resolve, dropped)})
    reuse_result = _reuse_with_valid_flags(reuse.run(record, ctx), resolve, dropped)
    replay_only = getattr(ctx.llm, "replay_only", None)
    return CaseAnalysis(
        case_id=record.case_id,
        absence=absence_result,
        clock=clock_result,
        reuse=reuse_result,
        dropped_flags=len(dropped),
        dropped_reasons=tuple(dropped),
        llm_model=ctx.llm.model,
        llm_mode=None if replay_only is None else ("replay" if replay_only else "live"),
    )


def process(
    base: CaseRecord,
    llm: CachedLLM,
    config: RatioConfig,
    *,
    embedder: Embedder | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[CaseRecord, CaseAnalysis, ExtractionReport]:
    """Extraction then analysis: what the app runs when a case is loaded."""
    record, report = ingest(base, llm, config, progress=progress)
    return record, analyze(record, analysis_context(config, llm, embedder)), report


@dataclass(frozen=True)
class LiveNote:
    """One note re-extracted by the local model, bypassing every cache."""

    doc_id: str
    model: str
    seconds: float
    events: tuple[Event, ...]
    arguments: tuple[Argument, ...]
    same_as_record: bool


def _items(record: CaseRecord, doc_id: str) -> tuple[set[tuple], set[tuple]]:
    events = {(e.type, e.span.start, e.span.end, e.parsed_date) for e in record.events if e.span.doc_id == doc_id and e.type != "hearing"}
    arguments = {(a.party, a.span.start, a.span.end) for a in record.arguments if a.span.doc_id == doc_id}
    return events, arguments


def run_note_live(record: CaseRecord, doc_id: str, config: RatioConfig, *, model: str | None = None) -> LiveNote:
    """Re-run extraction for one document with the live local model and compare it with the record."""
    document = record.document(doc_id)
    single = record.model_copy(update={"documents": (document,), "events": (), "arguments": ()})
    with tempfile.TemporaryDirectory(prefix="ratio-live-") as scratch:  # no cache to read from
        llm = make_llm(config, mode="live", write_dir=Path(scratch), read_dirs=(), model=model)
        started = time.monotonic()
        live, _ = ingest(single, llm, config)
        seconds = time.monotonic() - started
    events = tuple(e for e in live.events if e.type != "hearing")
    return LiveNote(
        doc_id=doc_id,
        model=llm.model,
        seconds=round(seconds, 1),
        events=events,
        arguments=live.arguments,
        same_as_record=_items(live, doc_id) == _items(record, doc_id),
    )


def build_demo_cache(config: RatioConfig, *, progress: ProgressCallback | None = None) -> DemoCacheManifest:
    """Run every model call of the demo case live and keep exactly those replies in the demo cache."""
    settings = config.settings.llm
    client = OllamaClient(settings)
    client.check_model()
    version = client.version()
    llm = CachedLLM(client, ResponseCache(DEMO_LLM_CACHE), settings=settings)
    base = load_case(DEMO_CASE_DIR)
    record, _ = ingest(base, llm, config, progress=progress)
    analyze(record, analysis_context(config, llm))
    for path in DEMO_LLM_CACHE.glob("*.json"):
        if path.stem not in llm.used_keys:
            path.unlink()
    manifest = DemoCacheManifest(
        synthetic=True,
        model=client.model,
        ollama_version=version,
        built_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        entries=len(llm.used_keys),
        documents={doc.id: doc.sha256 for doc in base.documents},
    )
    DEMO_CACHE_MANIFEST.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest
