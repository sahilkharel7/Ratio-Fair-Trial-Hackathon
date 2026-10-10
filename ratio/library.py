"""Court-scoped matter catalogue and staged document intake in Ratio's existing SQLite file.

The original CaseRecord/CaseManifest remain the interchange contract. A bulk PDF
classifier can stage files, then submit a verified CaseManifest with the selected
file ids; it does not need to change analysis modules or the UI. History/precedent
reference records are not silently promoted into the reviewer's matter collection.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ratio.extraction.build import build_base_record
from ratio.extraction.loader import (
    CaseManifest,
    LoaderError,
    MAX_DOCUMENT_BYTES,
    _check_relative,
    decode_document,
)
from ratio.focus import FocusRecord, screen, penalty_summary
from ratio.schema import CaseRecord, Frozen
from ratio.store import CaseStore

DDL = """
CREATE TABLE IF NOT EXISTS matter_catalogue (
 case_id TEXT PRIMARY KEY, title TEXT NOT NULL, court TEXT NOT NULL,
 defendant TEXT NOT NULL, charge TEXT NOT NULL, focus_json TEXT NOT NULL,
 updated_at TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'ready');
CREATE INDEX IF NOT EXISTS matters_by_court ON matter_catalogue(court, case_id);
CREATE TABLE IF NOT EXISTS intake_files (
 id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, name TEXT NOT NULL, sha256 TEXT NOT NULL,
 content BLOB NOT NULL, text TEXT, problem TEXT, provenance TEXT NOT NULL,
 source_note TEXT NOT NULL, assigned_case_id TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS focus_reviews (
 id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, prompt_id TEXT NOT NULL,
 decision_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS matter_files (
 case_id TEXT NOT NULL, path TEXT NOT NULL, intake_id TEXT NOT NULL,
 PRIMARY KEY(case_id,path));
CREATE TABLE IF NOT EXISTS case_notes (
 id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL,
 note TEXT NOT NULL, created_at TEXT NOT NULL);
"""


class FocusDecision(Frozen):
    case_id: str
    prompt_id: str
    decision: Literal["supported", "not_supported", "follow_up", "reopened"]
    reason: str
    reviewer: str = ""
    created_at: str = ""
    source_snapshot: dict


class LibraryStore:
    def __init__(self, store: CaseStore | None = None):
        self.cases = store or CaseStore()
        with self.connect() as conn:
            conn.executescript(DDL)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.cases.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def register(
        self, record: CaseRecord, *, defendant: str | None = None
    ) -> FocusRecord:
        focus = screen(record, defendant=defendant)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO matter_catalogue(case_id,title,court,defendant,charge,focus_json,updated_at) VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(case_id) DO UPDATE SET title=excluded.title,court=excluded.court,defendant=excluded.defendant,charge=excluded.charge,focus_json=excluded.focus_json,updated_at=excluded.updated_at",
                (
                    record.case_id,
                    record.meta.title,
                    record.meta.court,
                    focus.defendant,
                    focus.charge,
                    focus.model_dump_json(),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return focus

    def sync_analysed_matters(self):
        # Existing imports from earlier Ratio versions become discoverable. The
        # synthetic judge reference history has no analysis and stays out.
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.record_json FROM cases c JOIN analyses a ON a.case_id=c.case_id LEFT JOIN matter_catalogue m ON m.case_id=c.case_id WHERE m.case_id IS NULL"
            ).fetchall()
        for row in rows:
            self.register(CaseRecord.model_validate_json(row[0]))

    def list(self, court: str | None = None):
        self.sync_analysed_matters()
        sql = "SELECT m.*,c.synthetic,c.record_json,a.analysis_json FROM matter_catalogue m JOIN cases c ON c.case_id=m.case_id LEFT JOIN analyses a ON a.case_id=m.case_id"
        with self.connect() as conn:
            rows = conn.execute(
                sql
                + (" WHERE m.court=?" if court else "")
                + " ORDER BY m.updated_at DESC,m.case_id",
                (court,) if court else (),
            ).fetchall()
        results = []
        for row in rows:
            focus = FocusRecord.model_validate_json(row["focus_json"])
            record = CaseRecord.model_validate_json(row["record_json"])
            decisions = self.decisions(record.case_id)
            latest = {}
            for d in decisions:
                if d["decision"] == "reopened":
                    latest.pop(d["prompt_id"], None)
                else:
                    latest[d["prompt_id"]] = d
            results.append(
                {
                    "case_id": row["case_id"],
                    "title": row["title"],
                    "court": row["court"],
                    "defendant": row["defendant"],
                    "charge": row["charge"],
                    "synthetic": bool(row["synthetic"]),
                    "document_count": len(record.documents),
                    "has_analysis": row["analysis_json"] is not None,
                    "focus": focus.model_dump(mode="json"),
                    "penalty_summaries": {
                        stage: penalty_summary(focus, stage)
                        for stage in ("requested", "imposed", "statutory")
                    },
                    "decisions": decisions,
                    "pending_prompts": sum(
                        p.id not in latest or latest[p.id]["decision"] == "follow_up"
                        for p in focus.prompts
                    ),
                    "updated_at": row["updated_at"],
                }
            )
        return results

    def focus(self, case_id: str) -> FocusRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT focus_json FROM matter_catalogue WHERE case_id=?", (case_id,)
            ).fetchone()
        if row:
            return FocusRecord.model_validate_json(row[0])
        record = self.cases.load_case(case_id)
        return self.register(record) if record else None

    def stage(self, files: dict[str, bytes], *, provenance: str, source_note: str = ""):
        if provenance not in {"public", "synthetic"}:
            raise LoaderError("Declare public or synthetic source material.")
        if provenance == "public" and not source_note.strip():
            raise LoaderError("Public documents need their publication source.")
        if (
            not files
            or len(files) > 100
            or sum(len(data) for data in files.values()) > 50_000_000
        ):
            raise LoaderError("Upload 1–100 files, at most 50 MB in total.")
        batch = uuid.uuid4().hex
        rows = []
        for name, content in files.items():
            try:
                _check_relative(name)
            except ValueError as exc:
                raise LoaderError(str(exc)) from exc
            if Path(name).suffix.lower() not in {".pdf", ".txt", ".md"}:
                raise LoaderError(f"{name}: PDF, TXT or MD required.")
            if len(content) > MAX_DOCUMENT_BYTES:
                raise LoaderError(f"{name}: file exceeds the 5 MB input limit.")
            text, problem = None, None
            try:
                text = decode_document(name, content)
                if not text.strip():
                    problem = (
                        "No text extracted. Add an OCR text version before analysis."
                    )
            except LoaderError as exc:
                problem = str(exc)
            rows.append(
                (
                    uuid.uuid4().hex,
                    batch,
                    name,
                    hashlib.sha256(content).hexdigest(),
                    content,
                    text,
                    problem,
                    provenance,
                    source_note,
                    datetime.now(UTC).isoformat(),
                )
            )
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO intake_files(id,batch_id,name,sha256,content,text,problem,provenance,source_note,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return batch

    def pending_files(self):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id,batch_id,name,sha256,problem,provenance,source_note,LENGTH(content) AS bytes FROM intake_files WHERE assigned_case_id IS NULL ORDER BY created_at,name"
            ).fetchall()
        return [dict(row) for row in rows]

    def import_manifest(
        self,
        manifest: CaseManifest,
        file_ids: dict[str, str],
        *,
        defendant: str | None = None,
    ):
        if self.cases.load_case(manifest.case_id) is not None:
            raise LoaderError(
                "That case reference is already stored. Open the existing case or use a distinct reference."
            )
        if set(file_ids) != {doc.path for doc in manifest.documents}:
            raise LoaderError("Assign exactly the documents declared for this case.")
        if len(set(file_ids.values())) != len(file_ids):
            raise LoaderError("Each selected file must be assigned once.")
        files = {}
        with self.connect() as conn:
            for path, file_id in file_ids.items():
                row = conn.execute(
                    "SELECT * FROM intake_files WHERE id=? AND assigned_case_id IS NULL",
                    (file_id,),
                ).fetchone()
                if row is None:
                    raise LoaderError(
                        "A selected file is unavailable or already assigned to a case."
                    )
                if row["problem"]:
                    raise LoaderError(f"{row['name']}: {row['problem']}")
                if row["provenance"] != manifest.data_provenance:
                    raise LoaderError(
                        "Do not combine public and synthetic files in one case."
                    )
                files[path] = bytes(row["content"])
        record = build_base_record(manifest, files)
        # The original loader validates document declarations and synthetic markers.
        focus = screen(record, defendant=defendant)
        now = datetime.now(UTC).isoformat()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO cases(case_id,title,synthetic,record_json,updated_at) VALUES(?,?,?,?,?)",
                (
                    record.case_id,
                    record.meta.title,
                    int(record.meta.synthetic),
                    record.model_dump_json(),
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO matter_catalogue(case_id,title,court,defendant,charge,focus_json,updated_at) VALUES(?,?,?,?,?,?,?)",
                (
                    record.case_id,
                    record.meta.title,
                    record.meta.court,
                    focus.defendant,
                    focus.charge,
                    focus.model_dump_json(),
                    now,
                ),
            )
            for file_id in file_ids.values():
                changed = conn.execute(
                    "UPDATE intake_files SET assigned_case_id=? WHERE id=? AND assigned_case_id IS NULL",
                    (record.case_id, file_id),
                )
                if changed.rowcount != 1:
                    raise LoaderError(
                        "A selected file was assigned by another import; this import was rolled back."
                    )
            conn.executemany(
                "INSERT INTO matter_files(case_id,path,intake_id) VALUES(?,?,?)",
                [(record.case_id, path, id) for path, id in file_ids.items()],
            )
            conn.execute(
                "INSERT INTO kv(key,value) VALUES('last_case',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (record.case_id,),
            )
        return record

    def save_decision(self, decision: dict):
        stored = FocusDecision.model_validate(decision)
        if stored.decision != "reopened" and not stored.reason.strip():
            raise ValueError("Record a reason for the assessment.")
        focus = self.focus(stored.case_id)
        prompt = (
            next((p for p in focus.prompts if p.id == stored.prompt_id), None)
            if focus
            else None
        )
        if prompt is None or prompt.model_dump(mode="json") != stored.source_snapshot:
            raise ValueError(
                "The assessment must refer to a current, unchanged source prompt."
            )
        stored = stored.model_copy(update={"created_at": datetime.now(UTC).isoformat()})
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO focus_reviews(case_id,prompt_id,decision_json,created_at) VALUES(?,?,?,?)",
                (
                    stored.case_id,
                    stored.prompt_id,
                    stored.model_dump_json(),
                    stored.created_at,
                ),
            )
        return stored.model_dump(mode="json")

    def decisions(self, case_id):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT decision_json FROM focus_reviews WHERE case_id=? ORDER BY id",
                (case_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def add_note(self, case_id: str, note: str):
        if self.cases.load_case(case_id) is None:
            raise ValueError("Case not found.")
        if not note.strip() or len(note) > 5000:
            raise ValueError("Enter a working note of 1–5000 characters.")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO case_notes(case_id,note,created_at) VALUES(?,?,?)",
                (case_id, note.strip(), datetime.now(UTC).isoformat()),
            )

    def notes(self, case_id):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT note,created_at FROM case_notes WHERE case_id=? ORDER BY id",
                (case_id,),
            ).fetchall()
        return [dict(row) for row in rows]
