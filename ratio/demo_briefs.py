"""Offline coordination of an explicitly enabled public-document Groq demo worker.

The normal application network guard remains installed. A fixed worker receives
only the selected public/synthetic case sources through stdin, never the database
or notes. Keys and HTTP clients stay in that worker.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from pydantic import Field

from ratio.case_briefs import BriefDraft, BriefSection
from ratio.paths import REPO_ROOT

VERSION = "groq-case-brief-v3"
MODEL = "openai/gpt-oss-120b"


class CaseBriefSection(BriefSection):
    doc_id: str = Field(min_length=1, max_length=200)
    source_start: int | None = Field(default=None, ge=0)


class CaseBriefDraft(BriefDraft):
    sections: list[CaseBriefSection] = Field(min_length=1, max_length=5)


def source_pinpoint(title, text, start):
    label = title if len(title) <= 60 else title[:57] + "..."
    return f"{label} · stored text line {text.count(chr(10), 0, start) + 1}"


def verified_sections(record, draft):
    docs = {doc.id: doc for doc in record.documents}
    sections = []
    for section in draft.sections:
        doc = docs.get(section.doc_id)
        start = (
            section.source_start
            if section.source_start is not None
            else (doc.text.find(section.quote) if doc else -1)
        )
        if (
            doc is None
            or start < 0
            or doc.text[start : start + len(section.quote)] != section.quote
        ):
            raise ValueError(
                "Draft citation is absent from the saved source; no draft saved."
            )
        pinpoint = source_pinpoint(doc.title, doc.text, start)
        sections.append(
            {
                **section.model_dump(),
                "pinpoint": pinpoint,
                "span": {
                    "doc_id": doc.id,
                    "start": start,
                    "end": start + len(section.quote),
                    "text": section.quote,
                },
            }
        )
    return sections


class DemoBriefService:
    def __init__(self, library, *, runner=None):
        self.library, self._runner = library, runner or self._run_worker
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="public-demo-brief"
        )
        self._capacity, self._lock = threading.BoundedSemaphore(3), threading.Lock()
        self._model = os.environ.get("GROQ_MODEL", MODEL)
        if self._model not in {MODEL, "openai/gpt-oss-20b"}:
            raise ValueError("Unsupported Groq demo model.")
        with library.connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS case_ai_briefs (
                case_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL,
                method TEXT NOT NULL, sections_json TEXT, pdf BLOB, error TEXT NOT NULL,
                updated_at TEXT NOT NULL)""")
            conn.execute(
                "UPDATE case_ai_briefs SET status='failed',error='The previous demo run stopped. Retry the draft.' WHERE status IN ('queued','running')"
            )

    def info(self):
        return {
            "enabled": True,
            "provider": "Groq",
            "model": self._model,
            "scope": "Only confirmed public records and synthetic examples; selected source excerpts are sent to Groq.",
        }

    def _record(self, case_id):
        record = self.library.cases.load_case(case_id)
        if record is None:
            raise KeyError("Case not found.")
        if record.meta.data_provenance not in {"public", "synthetic"}:
            raise ValueError("Private records cannot be sent to Groq.")
        if not record.documents or len(record.documents) > 20:
            raise ValueError("A demo brief supports 1–20 source documents per case.")
        return record

    def _fingerprint(self, record):
        return hashlib.sha256(
            (VERSION + self._model + record.model_dump_json()).encode()
        ).hexdigest()

    def get(self, case_id):
        record = self._record(case_id)
        with self.library.connect() as conn:
            row = conn.execute(
                "SELECT * FROM case_ai_briefs WHERE case_id=?", (case_id,)
            ).fetchone()
        if row is None or row["fingerprint"] != self._fingerprint(record):
            return {"status": "not_generated", **self.info()}
        sections = []
        status, error = row["status"], row["error"]
        if row["status"] == "ready":
            try:
                manifest = json.loads(row["sections_json"])
                if (
                    hashlib.sha256(bytes(row["pdf"])).hexdigest()
                    != manifest["pdf_sha256"]
                ):
                    raise ValueError("Saved PDF checksum changed.")
                sections = verified_sections(
                    record,
                    CaseBriefDraft.model_validate({"sections": manifest["sections"]}),
                )
            except (ValueError, KeyError, TypeError):
                status, error = (
                    "failed",
                    "The saved draft changed or is damaged. Regenerate it from the original sources.",
                )
        return {
            "status": status,
            "method": row["method"],
            "sections": sections,
            "error": error,
            "updated_at": row["updated_at"],
            **self.info(),
        }

    def start(self, case_id, *, force=False):
        with self._lock:
            current = self.get(case_id)
            if current["status"] in {"queued", "running"} or (
                current["status"] == "ready" and not force
            ):
                return current
            record = self._record(case_id)
            if not self._capacity.acquire(blocking=False):
                raise ValueError(
                    "Three drafts are already queued. Let one finish before starting another."
                )
            try:
                with self.library.connect() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO case_ai_briefs VALUES(?,?,?,?,?,?,?,?)",
                        (
                            case_id,
                            self._fingerprint(record),
                            "queued",
                            "Groq draft from selected source excerpts · legal review required",
                            None,
                            None,
                            "",
                            datetime.now(UTC).isoformat(),
                        ),
                    )
                self._pool.submit(self._generate, record)
            except Exception:
                self._capacity.release()
                raise
        return self.get(case_id)

    def _run_worker(self, payload):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/groq_case_worker.py")],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=200,
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode or len(result.stdout) > 4_000_000:
            raise ValueError(
                "The demo worker could not complete the draft. Check Groq configuration and retry."
            )
        response = json.loads(result.stdout)
        if response.get("error"):
            raise ValueError(response["error"])
        return response

    def _generate(self, record):
        try:
            with self.library.connect() as conn:
                conn.execute(
                    "UPDATE case_ai_briefs SET status='running' WHERE case_id=?",
                    (record.case_id,),
                )
            payload = {
                "provenance": record.meta.data_provenance,
                "title": record.meta.title,
                "court": record.meta.court,
                "source_note": record.meta.source_note or "Synthetic demo",
                "model": self._model,
                "sources": [
                    {"id": d.id, "title": d.title, "type": d.type, "text": d.text}
                    for d in record.documents
                ],
            }
            response = self._runner(payload)
            draft = CaseBriefDraft.model_validate({"sections": response["sections"]})
            sections = verified_sections(record, draft)
            raw = base64.b64decode(response["pdf"], validate=True)
            from pypdf import PdfReader

            if (
                len(raw) > 2_000_000
                or not raw.startswith(b"%PDF-")
                or len(PdfReader(io.BytesIO(raw)).pages) != 1
            ):
                raise ValueError("Generated PDF did not pass the one-page check.")
            if self._fingerprint(self._record(record.case_id)) != self._fingerprint(
                record
            ):
                raise ValueError(
                    "The case changed during drafting; generate from the latest record."
                )
            with self.library.connect() as conn:
                conn.execute(
                    "UPDATE case_ai_briefs SET status='ready',sections_json=?,pdf=?,error='',updated_at=? WHERE case_id=?",
                    (
                        json.dumps(
                            {
                                "sections": sections,
                                "pdf_sha256": hashlib.sha256(raw).hexdigest(),
                            }
                        ),
                        raw,
                        datetime.now(UTC).isoformat(),
                        record.case_id,
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - background jobs must record failure and release queue capacity
            # Worker errors are curated messages; never expose raw subprocess stderr or source text.
            error = (
                str(exc)
                if isinstance(exc, ValueError) and len(str(exc)) < 300
                else "Draft generation failed. Your original documents are unchanged; retry later."
            )
            with self.library.connect() as conn:
                conn.execute(
                    "UPDATE case_ai_briefs SET status='failed',error=?,updated_at=? WHERE case_id=?",
                    (error, datetime.now(UTC).isoformat(), record.case_id),
                )
        finally:
            self._capacity.release()

    def pdf(self, case_id):
        if self.get(case_id)["status"] != "ready":
            return None
        with self.library.connect() as conn:
            row = conn.execute(
                "SELECT pdf,sections_json FROM case_ai_briefs WHERE case_id=?",
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        raw = bytes(row["pdf"])
        return (
            raw
            if hashlib.sha256(raw).hexdigest()
            == json.loads(row["sections_json"])["pdf_sha256"]
            else None
        )

    def close(self):
        self._pool.shutdown(wait=True)
