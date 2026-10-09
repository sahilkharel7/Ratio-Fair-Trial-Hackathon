"""The builder's work database, data/corpus/build.db (git-ignored): what was fetched, the normalised
documents, the model's raw answers (the extraction cache), the verified facets and their vectors.
`build_db.build_db` turns it into data/corpus/precedents.db, the file the app reads.

Every step reads the previous step's table and is idempotent, so a build can stop and resume.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ratio.paths import CORPUS_DIR
from ratio.precedent_schema import PrecedentDoc, PrecedentFacet

BUILD_DB = CORPUS_DIR / "build.db"
RAW_DIR = CORPUS_DIR / "raw"

_DDL = (
    """CREATE TABLE IF NOT EXISTS raw (
        url TEXT PRIMARY KEY, source_id TEXT NOT NULL, sha256 TEXT NOT NULL, content_type TEXT NOT NULL,
        path TEXT NOT NULL, fetched_at TEXT NOT NULL, title TEXT, symbol TEXT, state TEXT, year INTEGER)""",
    "CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, raw_url TEXT NOT NULL, doc_json TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS extractions (
        cache_key TEXT PRIMARY KEY, precedent_id TEXT NOT NULL, model TEXT NOT NULL, answer_json TEXT NOT NULL,
        created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS verified (
        precedent_id TEXT PRIMARY KEY, facets_json TEXT NOT NULL, dropped INTEGER NOT NULL, cache_key TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS vectors (
        precedent_id TEXT NOT NULL, facet_id TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
        vec BLOB NOT NULL, PRIMARY KEY (precedent_id, facet_id, start, end))""",
)


@dataclass(frozen=True)
class RawItem:
    url: str
    source_id: str
    sha256: str
    content_type: str
    path: Path
    fetched_at: str
    title: str | None = None
    symbol: str | None = None
    state: str | None = None
    year: int | None = None


class BuildStore:
    def __init__(self, path: Path = BUILD_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            for statement in _DDL:
                db.execute(statement)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    # --- fetch ----------------------------------------------------------------------------------

    def record_raw(self, item: RawItem) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO raw VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (item.url, item.source_id, item.sha256, item.content_type, str(item.path), item.fetched_at,
                 item.title, item.symbol, item.state, item.year),
            )  # fmt: skip

    def raw_items(self, source_id: str | None = None) -> list[RawItem]:
        query, args = "SELECT * FROM raw", ()
        if source_id is not None:
            query, args = query + " WHERE source_id = ?", (source_id,)
        with self._connect() as db:
            rows = db.execute(query + " ORDER BY url", args).fetchall()
        return [RawItem(r[0], r[1], r[2], r[3], Path(r[4]), r[5], r[6], r[7], r[8], r[9]) for r in rows]

    def has_raw(self, url: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM raw WHERE url = ?", (url,)).fetchone() is not None

    # --- normalize ------------------------------------------------------------------------------

    def put_document(self, doc: PrecedentDoc, raw_url: str) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", (doc.id, raw_url, doc.model_dump_json()))

    def documents(self) -> list[PrecedentDoc]:
        with self._connect() as db:
            rows = db.execute("SELECT doc_json FROM documents ORDER BY id").fetchall()
        return [PrecedentDoc.model_validate_json(row[0]) for row in rows]

    def document(self, precedent_id: str) -> PrecedentDoc | None:
        with self._connect() as db:
            row = db.execute("SELECT doc_json FROM documents WHERE id = ?", (precedent_id,)).fetchone()
        return PrecedentDoc.model_validate_json(row[0]) if row else None

    # --- extract (the model's raw answers; the cache) -------------------------------------------

    def get_extraction(self, cache_key: str) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT answer_json FROM extractions WHERE cache_key = ?", (cache_key,)).fetchone()
        return row[0] if row else None

    def put_extraction(self, cache_key: str, precedent_id: str, model: str, answer_json: str, created_at: str) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO extractions VALUES (?, ?, ?, ?, ?)", (cache_key, precedent_id, model, answer_json, created_at))

    # --- verify ---------------------------------------------------------------------------------

    def put_verified(self, precedent_id: str, facets: tuple[PrecedentFacet, ...], dropped: int, cache_key: str) -> None:
        payload = json.dumps([facet.model_dump(mode="json") for facet in facets])
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO verified VALUES (?, ?, ?, ?)", (precedent_id, payload, dropped, cache_key))

    def verified(self) -> dict[str, tuple[tuple[PrecedentFacet, ...], int]]:
        with self._connect() as db:
            rows = db.execute("SELECT precedent_id, facets_json, dropped FROM verified ORDER BY precedent_id").fetchall()
        return {pid: (tuple(PrecedentFacet.model_validate(f) for f in json.loads(payload)), dropped) for pid, payload, dropped in rows}

    # --- embed ----------------------------------------------------------------------------------

    def put_vector(self, precedent_id: str, facet_id: str, start: int, end: int, vec: np.ndarray) -> None:
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?)", (precedent_id, facet_id, start, end, blob))

    def vectors(self) -> dict[tuple[str, str, int, int], np.ndarray]:
        with self._connect() as db:
            rows = db.execute("SELECT precedent_id, facet_id, start, end, vec FROM vectors").fetchall()
        return {(r[0], r[1], r[2], r[3]): np.frombuffer(r[4], dtype=np.float32) for r in rows}
