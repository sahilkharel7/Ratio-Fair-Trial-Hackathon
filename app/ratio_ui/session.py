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
from ratio.paths import DEMO_CASE_DIR
from ratio.pipeline import demo_llm, make_llm, process
from ratio.results import CaseAnalysis
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


def load_demo(progress: Progress | None = None) -> str:
    """The demo case, every model answer replayed from the committed cache (Ollama not needed)."""
    cfg = config()
    record, analysis, _ = process(load_case(DEMO_CASE_DIR), demo_llm(cfg), cfg, embedder=embedder(), progress=progress)
    return _save(record, analysis)


def analyze_upload(manifest: CaseManifest, files: dict[str, bytes], progress: Progress | None = None) -> str:
    """An uploaded case, read by the live local model (its replies are cached under data/cache)."""
    cfg = config()
    base = build_base_record(manifest, files)
    record, analysis, _ = process(base, make_llm(cfg, mode="live"), cfg, embedder=embedder(), progress=progress)
    return _save(record, analysis)
