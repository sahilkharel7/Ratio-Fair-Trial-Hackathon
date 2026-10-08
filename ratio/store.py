"""SQLite persistence for case records and analyses.

Records are stored as validated JSON. Each call opens a short-lived connection, because
Streamlit runs app code on several threads, and every query is parameterised.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ratio.paths import db_path
from ratio.results import CaseAnalysis
from ratio.schema import CaseRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    synthetic INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
    case_id TEXT PRIMARY KEY,
    analysis_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class CaseSummary:
    case_id: str
    title: str
    synthetic: bool
    updated_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CaseStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        try:
            with conn:  # commit on success, roll back on error
                yield conn
        finally:
            conn.close()

    def save_case(self, record: CaseRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cases (case_id, title, synthetic, record_json, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(case_id) DO UPDATE SET title = excluded.title, synthetic = excluded.synthetic, "
                "record_json = excluded.record_json, updated_at = excluded.updated_at",
                (record.case_id, record.meta.title, int(record.meta.synthetic), record.model_dump_json(), _now()),
            )
            conn.execute("DELETE FROM analyses WHERE case_id = ?", (record.case_id,))  # no stale analysis

    def load_case(self, case_id: str) -> CaseRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT record_json FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return CaseRecord.model_validate_json(row[0]) if row else None

    def list_cases(self) -> tuple[CaseSummary, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT case_id, title, synthetic, updated_at FROM cases ORDER BY title, case_id"
            ).fetchall()
        return tuple(CaseSummary(case_id=r[0], title=r[1], synthetic=bool(r[2]), updated_at=r[3]) for r in rows)

    def all_records(self) -> tuple[CaseRecord, ...]:
        with self._connect() as conn:
            rows = conn.execute("SELECT record_json FROM cases ORDER BY case_id").fetchall()
        return tuple(CaseRecord.model_validate_json(row[0]) for row in rows)

    def save_analysis(self, analysis: CaseAnalysis) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO analyses (case_id, analysis_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(case_id) DO UPDATE SET analysis_json = excluded.analysis_json, "
                "updated_at = excluded.updated_at",
                (analysis.case_id, analysis.model_dump_json(), _now()),
            )

    def load_analysis(self, case_id: str) -> CaseAnalysis | None:
        with self._connect() as conn:
            row = conn.execute("SELECT analysis_json FROM analyses WHERE case_id = ?", (case_id,)).fetchone()
        return CaseAnalysis.model_validate_json(row[0]) if row else None

    def set_last_case(self, case_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO kv (key, value) VALUES ('last_case', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (case_id,),
            )

    def last_case(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key = 'last_case'").fetchone()
        return row[0] if row else None
