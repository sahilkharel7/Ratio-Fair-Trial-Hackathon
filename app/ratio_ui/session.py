"""Cached services and the case being viewed. The SQLite store is the source of truth, so after a
browser refresh the pages show the last analysed case again."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

from ratio.config import RatioConfig, default_config
from ratio.embeddings import MiniLMEmbedder
from ratio.extraction.build import build_base_record, load_case
from ratio.extraction.loader import CaseManifest
from ratio.history import history_for, load_all_alias_decisions, load_history
from ratio.paths import DEMO_CASE_DIR
from ratio.pipeline import analyze_judges, demo_llm, make_llm, process
from ratio.results import CaseAnalysis, JudgeReport
from ratio.schema import CaseRecord
from ratio.store import CaseStore

CASE_KEY = "ratio_case_id"
Progress = Callable[[str, int, int], None]


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
