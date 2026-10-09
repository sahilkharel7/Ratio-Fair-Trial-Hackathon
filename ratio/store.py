"""SQLite persistence for case records, analyses and reviewers' decisions.

Records are stored as validated JSON. Each call opens a short-lived connection, because
Streamlit runs app code on several threads, and every query is parameterised. Reviewers' decisions
are append-only and survive a re-analysis of their case (ratio/feedback.py).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ratio.feedback import MissedIssue, Review
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
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    flag_id TEXT NOT NULL,
    review_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS reviews_by_case ON reviews (case_id, id);
CREATE TABLE IF NOT EXISTS missed_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    issue_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    withdrawn INTEGER NOT NULL DEFAULT 0
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

    # --- reviewers' decisions (append-only) ---------------------------------------------------

    def add_review(self, review: Review) -> Review:
        stored = review.model_copy(update={"created_at": _now()})
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO reviews (case_id, flag_id, review_json, created_at) VALUES (?, ?, ?, ?)",
                (stored.case_id, stored.flag_id, stored.model_dump_json(), stored.created_at),
            )
        return stored

    def reviews(self, case_id: str | None = None) -> tuple[Review, ...]:
        """Every decision, in the order it was made; for one case, or for all."""
        with self._connect() as conn:
            if case_id is None:
                rows = conn.execute("SELECT review_json FROM reviews ORDER BY id").fetchall()
            else:
                rows = conn.execute("SELECT review_json FROM reviews WHERE case_id = ? ORDER BY id", (case_id,)).fetchall()
        return tuple(Review.model_validate_json(row[0]) for row in rows)

    def add_missed_issue(self, issue: MissedIssue) -> MissedIssue:
        created = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO missed_issues (case_id, issue_json, created_at) VALUES (?, ?, ?)",
                (issue.case_id, "{}", created),
            )
            stored = issue.model_copy(update={"id": cursor.lastrowid, "created_at": created})
            conn.execute("UPDATE missed_issues SET issue_json = ? WHERE id = ?", (stored.model_dump_json(), stored.id))
        return stored

    def missed_issues(self, case_id: str | None = None) -> tuple[MissedIssue, ...]:
        """The missed issues not withdrawn, in the order they were recorded."""
        query = "SELECT issue_json FROM missed_issues WHERE withdrawn = 0"
        with self._connect() as conn:
            if case_id is None:
                rows = conn.execute(f"{query} ORDER BY id").fetchall()
            else:
                rows = conn.execute(f"{query} AND case_id = ? ORDER BY id", (case_id,)).fetchall()
        return tuple(MissedIssue.model_validate_json(row[0]) for row in rows)

    def withdraw_missed_issue(self, issue_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE missed_issues SET withdrawn = 1 WHERE id = ?", (issue_id,))
