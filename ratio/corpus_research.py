"""Read the installed corpus for local wording research, without a model or network."""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

from ratio.case_briefs import editorial_card, load_brief
from ratio.precedent_store import CorpusUnavailable, SqlitePrecedentIndex

DIRECTORIES = (
    {
        "title": "OHCHR JURIS",
        "url": "https://juris.ohchr.org/",
        "scope": "UN treaty-body decisions. Select CCPR and Article 14(2); retain inadmissibility decisions separately.",
    },
    {
        "title": "HUDOC",
        "url": "https://hudoc.echr.coe.int/",
        "scope": "European Court judgments and decisions. Filter Article 6-2, English, and document type. This is a separate regional standard.",
    },
    {
        "title": "OHCHR collection of full court judgments",
        "url": "https://cambodia.ohchr.org/en/rule-of-law/echr-decisions?items_per_page=60",
        "scope": "Full ECHR judgments reproduced by OHCHR Cambodia. Check document language and date; issuing court remains the ECHR.",
    },
    {
        "title": "ECHR presumption of innocence case guide",
        "url": "https://ks.echr.coe.int/documents/d/echr-ks/presumption-of-innocence",
        "scope": "Court-produced reading guide with links to leading cases; verify each judgment and applicable forum.",
    },
)


class CorpusResearch:
    """Refresh on an atomic corpus rebuild; refuse missing or altered texts."""

    def __init__(self, path: Path | None = None):
        from ratio.paths import corpus_db_path

        self.path = Path(path) if path else corpus_db_path()
        self._stamp = None
        self._index = None
        self._embedder = None
        self._model_lock = threading.Lock()

    def index(self):
        try:
            stat = self.path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            if self._index is None or stamp != self._stamp:
                self._index = SqlitePrecedentIndex(self.path)
                self._stamp = stamp
            return self._index
        except (OSError, CorpusUnavailable):
            self._index = None
            return None

    def catalogue(self):
        index = self.index()
        if index is None:
            return {
                "available": False,
                "documents": [],
                "directories": DIRECTORIES,
                "message": "Build or copy the local precedent corpus to search downloaded cases.",
            }
        docs = [d for d in index.documents() if index.text(d.doc_id) is not None]
        return {
            "available": True,
            "built_at": index.meta().built_at,
            "documents": [
                {
                    **self.descriptor(d),
                    "one_pager": self.brief(d.id) is not None,
                    **editorial_card(d),
                }
                for d in sorted(
                    docs, key=lambda d: (d.kind != "court_judgment", d.title)
                )
            ],
            "directories": DIRECTORIES,
            "message": "Downloaded reference material. Wording matches are reading leads, not findings or coded outcomes.",
        }

    @staticmethod
    def descriptor(doc):
        return {
            key: getattr(doc, key)
            for key in (
                "id",
                "title",
                "body",
                "kind",
                "state",
                "year",
                "url",
                "private",
                "attribution",
            )
        }

    def document(self, document_id):
        index = self.index()
        if index is None:
            return None
        doc = next((d for d in index.documents() if d.id == document_id), None)
        if doc is None or index.text(doc.doc_id) is None:
            return None
        return {
            **self.descriptor(doc),
            "text": doc.text,
            "text_sha256": doc.text_sha256,
            "local_pdf": self._pdf_path(doc) is not None,
            "brief": (brief[0] if (brief := self.brief(doc.id)) else None),
        }

    def brief(self, document_id):
        index = self.index()
        if index is None:
            return None
        doc = next((d for d in index.documents() if d.id == document_id), None)
        if doc is None or index.text(doc.doc_id) is None:
            return None
        return load_brief(doc, self.path.parent / "briefs")

    def _pdf_path(self, doc):
        if doc.private:
            return None
        raw_root = (self.path.parent / "raw").resolve()
        for path in raw_root.glob(f"*/{doc.raw_sha256}.pdf"):
            if path.resolve().is_relative_to(raw_root) and path.is_file():
                return path
        return None

    def original_pdf(self, document_id):
        index = self.index()
        if index is None:
            return None
        doc = next((d for d in index.documents() if d.id == document_id), None)
        if doc is None or index.text(doc.doc_id) is None:
            return None
        path = self._pdf_path(doc)
        try:
            if path is None or path.stat().st_size > 30_000_000:
                return None
            raw = path.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(raw).hexdigest() != doc.raw_sha256:
            return None
        return doc, raw

    def search(self, query, *, kind="", state="", limit=20, mode="phrase"):
        if not isinstance(query, str) or len(query) > 300:
            raise ValueError("Use a search of at most 300 characters.")
        query = query.strip()
        if mode not in {"phrase", "meaning"}:
            raise ValueError("Choose phrase or meaning search.")
        index = self.index()
        if index is None or len(query) < 3:
            return {
                "hits": [],
                "total_documents": 0,
                "method": "Exact phrase, ignoring whitespace and letter case",
            }
        if mode == "meaning":
            return self._meaning(index, query, kind=kind, state=state, limit=limit)
        # PDF line breaks must not prevent an exact phrase match. Original offsets
        # and original characters are retained in every displayed passage.
        pattern = re.compile(
            r"\s+".join(re.escape(w) for w in query.split()), re.IGNORECASE
        )
        hits = []
        matched_documents = 0
        for doc in sorted(index.documents(), key=lambda d: (d.title, d.id)):
            if (kind and doc.kind != kind) or (state and doc.state != state):
                continue
            text = index.text(doc.doc_id)
            if text is None:
                continue
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            matched_documents += 1
            for match in matches[:3]:
                start, end = (
                    max(0, match.start() - 200),
                    min(len(text), match.end() + 350),
                )
                hits.append(
                    {
                        "document": self.descriptor(doc),
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                        "match_start": match.start(),
                        "match_end": match.end(),
                        "line": text.count("\n", 0, match.start()) + 1,
                    }
                )
        return {
            "hits": hits[: max(1, min(limit, 50))],
            "total_documents": matched_documents,
            "method": "Exact phrase, ignoring whitespace and letter case",
        }

    def _meaning(self, index, query, *, kind, state, limit):
        from ratio.embeddings import EmbeddingModelMissing, MiniLMEmbedder

        docs = {
            d.id: d
            for d in index.documents()
            if (not kind or d.kind == kind) and (not state or d.state == state)
        }
        try:
            with self._model_lock:
                if self._embedder is None:
                    self._embedder = MiniLMEmbedder()
                vector = self._embedder.encode([query])[0]
        except EmbeddingModelMissing as exc:
            raise ValueError(
                "Meaning search needs the local MiniLM model. Use phrase search or run scripts/fetch_models.py."
            ) from exc
        found = index.search_passages(
            vector, words=query, k=max(1, min(limit, 50)), precedent_ids=docs
        )
        hits = []
        for hit in found:
            doc, span = docs[hit.precedent_id], hit.quote.span
            hits.append(
                {
                    "document": self.descriptor(doc),
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                    "match_start": span.start,
                    "match_end": span.end,
                    "line": doc.text.count("\n", 0, span.start) + 1,
                }
            )
        return {
            "hits": hits,
            "total_documents": len({h["document"]["id"] for h in hits}),
            "method": "Local meaning and wording ranking; document count covers shown results only",
        }
