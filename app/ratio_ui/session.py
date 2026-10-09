"""Cached services and the case being viewed. The SQLite store is the source of truth, so after a
browser refresh the pages show the last analysed case again."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, replace

import streamlit as st

from ratio import precedents
from ratio.config import PrecedentSettings, RatioConfig, default_config
from ratio.embeddings import MiniLMEmbedder
from ratio.extraction.build import build_base_record, load_case
from ratio.extraction.loader import CaseManifest, LoaderError
from ratio.feedback import Review, case_findings, rejected_flag_ids
from ratio.history import history_for, load_all_alias_decisions, load_history
from ratio.paths import DEMO_CASE_DIR, corpus_db_path
from ratio.pipeline import analyze_judges, demo_llm, make_llm, process
from ratio.precedent_access import open_index
from ratio.precedent_schema import CaseProfile, PrecedentIndex, PrecedentLink
from ratio.results import CaseAnalysis, JudgeReport
from ratio.schema import CaseRecord, Flag
from ratio.store import CaseStore

CASE_KEY = "ratio_case_id"
PRECEDENT_INDEX_TTL_SECONDS = 60  # OpenSearch may start or stop while the app runs
SIMILAR_WAIT_SECONDS = 2.0  # the longest the Case page waits for all of Similar cases' work; it keeps computing
SIMILAR_PAGE_WAIT_SECONDS = 30.0  # the longest the Similar cases page waits before offering to check again
SIMILAR_RETRY_SECONDS = 60.0  # a failed computation is remembered this long before a run starts it again
SIMILAR_JOBS_KEPT = 32  # computations remembered, one per case version, reviews and corpus
Progress = Callable[[str, int, int], None]
Links = tuple[PrecedentLink, ...]

_log = logging.getLogger(__name__)
_link_jobs: dict[str, _LinkJob] = {}  # shared by every page, run and session, like st.cache_resource
_link_jobs_lock = threading.Lock()


@st.cache_resource(show_spinner=False)
def config() -> RatioConfig:
    return default_config()


@st.cache_resource(show_spinner="Loading the local embedding model")
def embedder() -> MiniLMEmbedder:
    return MiniLMEmbedder()


def store() -> CaseStore:
    return CaseStore()


@dataclass(frozen=True)
class LoadedCase:
    record: CaseRecord
    analysis: CaseAnalysis | None


def current_case() -> LoadedCase | None:
    case_id = st.session_state.get(CASE_KEY) or store().last_case()
    record = store().load_case(case_id) if case_id else None
    if record is None:
        return None
    st.session_state[CASE_KEY] = case_id
    return LoadedCase(record=record, analysis=store().load_analysis(case_id))


def _save(record: CaseRecord, analysis: CaseAnalysis) -> str:
    cases = store()
    cases.save_case(record)
    cases.save_analysis(analysis)
    cases.set_last_case(record.case_id)
    st.session_state[CASE_KEY] = record.case_id
    return record.case_id


def seed_history(records: tuple[CaseRecord, ...] | None = None) -> int:
    """The synthetic judicial history behind the judge page (all of it by default): coded rulings
    only, no model involved."""
    cases = store()
    records = load_history() if records is None else records
    for record in records:
        cases.save_case(record)
    return len(records)


def load_demo(progress: Progress | None = None) -> str:
    """The demo case, every model answer replayed from the committed cache (Ollama not needed), and the
    synthetic history of its court for the judge page."""
    cfg = config()
    record, analysis, _ = process(load_case(DEMO_CASE_DIR), demo_llm(cfg), cfg, embedder=embedder(), progress=progress)
    seed_history()
    return _save(record, analysis)


def judge_report() -> JudgeReport:
    """Module 4 over every case in the store, recomputed on each run (it takes milliseconds)."""
    cases = store()
    records = cases.all_records()
    analyses = {record.case_id: analysis for record in records if (analysis := cases.load_analysis(record.case_id)) is not None}
    return analyze_judges(records, analyses, load_all_alias_decisions(), config())


def judge_report_or_none() -> JudgeReport | None:
    """The judge report, or None when the registry cannot be built (the judge page explains why)."""
    try:
        return judge_report()
    except LoaderError:
        return None


def findings(loaded: LoadedCase) -> tuple[Flag, ...]:
    """Every finding shown for the case: its own, then the patterns of its presiding judge."""
    return case_findings(loaded.analysis, judge_report_or_none()) if loaded.analysis is not None else ()


def open_case(case_id: str) -> None:
    store().set_last_case(case_id)
    st.session_state[CASE_KEY] = case_id


def analyze_upload(manifest: CaseManifest, files: dict[str, bytes], progress: Progress | None = None) -> str:
    """An uploaded case, read by the live local model (its replies are cached under data/cache). A
    synthetic case from the court of the synthetic history gets that history, as the demo does."""
    cfg = config()
    base = build_base_record(manifest, files)
    record, analysis, _ = process(base, make_llm(cfg, mode="live"), cfg, embedder=embedder(), progress=progress)
    if record.meta.synthetic:  # before saving, so a history case can never replace the upload
        seed_history(history_for(record, load_history()))
    return _save(record, analysis)


# --- Similar cases ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarCases:
    """What one background computation found for a case: its fact patterns and the public cases linked
    to them (both None when no corpus is installed), or the error that stopped it."""

    profile: CaseProfile | None = None
    links: Links | None = None
    error: Exception | None = None
    finished_at: float = 0.0  # time.monotonic()


@dataclass(frozen=True)
class _LinkJob:
    outcome: Future[SimilarCases]  # always completes with a SimilarCases, never with an exception
    waited_out: bool = False  # a Case page run gave up waiting for it: later runs do not wait again


def precedent_settings() -> PrecedentSettings:
    """Which precedent index Similar cases reads (the tests choose the corpus file here)."""
    return config().settings.precedents


@st.cache_resource(show_spinner=False, ttl=PRECEDENT_INDEX_TTL_SECONDS)
def _open_precedent_index(_settings: PrecedentSettings, key: str) -> tuple[PrecedentIndex | None, str]:
    return open_index(_settings, config().fact_patterns)


def _corpus_key(settings: PrecedentSettings) -> str:
    """The settings and the corpus file's path and version: a rebuilt or new corpus is opened again."""
    path = corpus_db_path()
    try:
        version = path.stat().st_mtime_ns
    except OSError:
        version = None
    return f"{settings.model_dump_json()}|{path}|{version}"


def _index_for(settings: PrecedentSettings) -> tuple[PrecedentIndex | None, str]:
    return _open_precedent_index(settings, _corpus_key(settings))


def precedent_index() -> tuple[PrecedentIndex | None, str]:
    """The precedent index and a one-line status, or (None, what to run) when no corpus is installed."""
    return _index_for(precedent_settings())


def clear_precedent_index() -> None:
    """Forgets the open index and every link computed from it (a computation still running finishes unseen)."""
    _open_precedent_index.clear()
    with _link_jobs_lock:
        _link_jobs.clear()


def _fact_profile(loaded: LoadedCase, reviews: Sequence[Review]) -> CaseProfile:
    """The case's fact patterns, each with the exact passages it rests on; a finding the reviewing
    lawyer rejected makes none."""
    if loaded.analysis is None:
        return CaseProfile(case_id=loaded.record.case_id)
    rejected = rejected_flag_ids(reviews)
    return precedents.profile(loaded.record, loaded.analysis, config().fact_patterns, rejected_flag_ids=rejected)


def _compute_similar(loaded: LoadedCase, reviews: Sequence[Review], settings: PrecedentSettings) -> SimilarCases:
    """All of Similar cases' work for one case, run in the background: opening the index, reading the
    case's fact patterns and linking them."""
    index, _ = _index_for(settings)
    if index is None:
        return SimilarCases(finished_at=time.monotonic())
    found = _fact_profile(loaded, reviews)
    links = precedents.link(found, index, embedder(), settings, config().fact_patterns)
    return SimilarCases(profile=found, links=links, finished_at=time.monotonic())


def _in_background(compute: Callable[[], SimilarCases]) -> Future[SimilarCases]:
    """``compute`` on a daemon thread: a stalled index never holds up a page, nor the app's exit. A
    failure is logged here, once, and kept as the outcome."""
    future: Future[SimilarCases] = Future()

    def run() -> None:
        try:
            outcome = compute()
        except Exception as exc:  # noqa: BLE001 - kept as the outcome: remembered, and shown by the Similar cases page
            _log.warning("Similar cases could not be computed", exc_info=True)
            outcome = SimilarCases(error=exc, finished_at=time.monotonic())
        future.set_result(outcome)

    threading.Thread(target=run, name="ratio-similar-cases", daemon=True).start()
    return future


def _job_key(loaded: LoadedCase, reviews: Sequence[Review], settings: PrecedentSettings) -> str:
    """The case id, a digest of the stored case, its analysis and its reviews, and the corpus: cheap to
    compute on every run (well under a millisecond for the demo)."""
    digest = hashlib.sha256(loaded.record.model_dump_json().encode("utf-8"))
    digest.update(loaded.analysis.model_dump_json().encode("utf-8") if loaded.analysis is not None else b"")
    for review in reviews:
        digest.update(review.model_dump_json().encode("utf-8"))
    return f"{loaded.record.case_id}|{digest.hexdigest()}|{_corpus_key(settings)}"


def _retry_due(job: _LinkJob) -> bool:
    """A failed computation is remembered SIMILAR_RETRY_SECONDS, so a broken index is not retried, nor
    logged, on every run."""
    if not job.outcome.done():
        return False
    outcome = job.outcome.result()
    return outcome.error is not None and time.monotonic() - outcome.finished_at >= SIMILAR_RETRY_SECONDS


def _remember(key: str, job: _LinkJob) -> None:
    _link_jobs.pop(key, None)
    _link_jobs[key] = job
    while len(_link_jobs) > SIMILAR_JOBS_KEPT:
        _link_jobs.pop(next(iter(_link_jobs)))  # the oldest


def _similar_job(loaded: LoadedCase) -> tuple[str, _LinkJob]:
    """The case's Similar cases computation, started once per case version, reviews and corpus, and
    shared by every page and run. Only the key is computed here; the work runs in the background."""
    settings = precedent_settings()
    reviews = store().reviews(loaded.record.case_id)
    key = _job_key(loaded, reviews, settings)
    with _link_jobs_lock:
        job = _link_jobs.get(key)
        if job is None or _retry_due(job):
            job = _LinkJob(_in_background(lambda: _compute_similar(loaded, reviews, settings)))
            _remember(key, job)
        return key, job


def _wait(key: str, job: _LinkJob, timeout: float) -> SimilarCases | None:
    """The outcome within ``timeout`` seconds, else None; a wait that runs out is remembered."""
    try:
        return job.outcome.result(timeout=timeout)
    except TimeoutError:
        with _link_jobs_lock:
            if _link_jobs.get(key) is job:
                _link_jobs[key] = replace(job, waited_out=True)
        return None


def similar_cases(loaded: LoadedCase, timeout: float) -> SimilarCases | None:
    """The case's fact patterns and the public cases sharing enough of them, best first, waiting at most
    ``timeout`` seconds: None while they are still computing. Raises the error that stopped the
    computation (EmbeddingModelMissing when the local embedding model has not been downloaded)."""
    key, job = _similar_job(loaded)
    found = _wait(key, job, timeout)
    if found is not None and found.error is not None:
        raise found.error
    return found


def similar_links_ready(loaded: LoadedCase) -> Links | None:
    """The links if they are ready within SIMILAR_WAIT_SECONDS, else None (also without a corpus). For the
    Case page, which never fails for Similar cases and waits for a computation only once: after a wait
    ran out, later runs return at once until it finishes. The Similar cases page shows any error."""
    try:
        key, job = _similar_job(loaded)
    except Exception:  # noqa: BLE001 - an optional line: the demo's main page must never break on it
        _log.warning("Similar cases left out of the Case page", exc_info=True)
        return None
    if job.waited_out and not job.outcome.done():
        return None
    found = _wait(key, job, SIMILAR_WAIT_SECONDS)
    return None if found is None or found.error is not None else found.links
