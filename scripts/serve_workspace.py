"""Serve the built React case workspace with a persistent, loopback-only SQLite API.

Build with npm --prefix web run build, then run this script. --demo adds the three
synthetic matters without overwriting existing cases. The default server remains
offline. --groq-demo explicitly enables a separate online worker for selected
public/synthetic source excerpts. Original runtime files never enter the public exporter.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError

from ratio import netguard
from ratio.corpus_research import CorpusResearch
from ratio.extraction.loader import CaseManifest, LoaderError
from ratio.library import LibraryStore
from ratio.outcomes import registry
from ratio.store import CaseStore
from ratio.workspace import case_payload, focus_report, seed_collection

MAX_REQUEST = 70_000_000


def handler_for(library: LibraryStore, dist: Path, research=None, briefs=None):
    research = research or CorpusResearch()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(dist), **kwargs)

        def log_message(self, *args):
            pass  # do not log uploaded case titles, request bodies or source text

        def end_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
            )
            super().end_headers()

        def reply(self, payload, status=200):
            raw = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def allowed(self):
            host = self.headers.get("Host", "")
            if host not in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }:
                self.reply(
                    {
                        "error": "Use the loopback address printed by the workspace server."
                    },
                    403,
                )
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }:
                self.reply(
                    {"error": "Cross-origin workspace access is not allowed."}, 403
                )
                return False
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.reply(
                    {"error": "Cross-site workspace access is not allowed."}, 403
                )
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            path = unquote(urlsplit(self.path).path)
            params = parse_qs(urlsplit(self.path).query)
            if path == "/api/precedents":
                return self.reply(research.catalogue())
            if path == "/api/precedents/search":
                try:
                    return self.reply(
                        research.search(
                            params.get("q", [""])[0],
                            kind=params.get("kind", [""])[0],
                            state=params.get("state", [""])[0],
                            mode=params.get("mode", ["phrase"])[0],
                        )
                    )
                except ValueError as exc:
                    return self.reply({"error": str(exc)}, 400)
            if path.startswith("/api/precedents/"):
                if path.endswith("/brief.pdf"):
                    brief = research.brief(
                        path.removeprefix("/api/precedents/").removesuffix("/brief.pdf")
                    )
                    if brief is None:
                        return self.reply(
                            {
                                "error": "Verified one-pager is unavailable; read the original source."
                            },
                            404,
                        )
                    raw = brief[1]
                    self.send_response(200)
                    self.send_header("Content-Type", "application/pdf")
                    self.send_header(
                        "Content-Disposition",
                        'attachment; filename="case-reading-sheet.pdf"',
                    )
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if path.endswith("/original"):
                    original = research.original_pdf(
                        path.removeprefix("/api/precedents/").removesuffix("/original")
                    )
                    if original is None:
                        return self.reply(
                            {
                                "error": "Verified original PDF is unavailable on this computer."
                            },
                            404,
                        )
                    doc, raw = original
                    self.send_response(200)
                    self.send_header("Content-Type", "application/pdf")
                    self.send_header(
                        "Content-Disposition", f'attachment; filename="{doc.id}.pdf"'
                    )
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                document = research.document(path.removeprefix("/api/precedents/"))
                return (
                    self.reply(document)
                    if document
                    else self.reply({"error": "Reference document not found."}, 404)
                )
            if path == "/api/library":
                return self.reply(
                    {
                        "mode": "local_sqlite",
                        "cases": library.list(),
                        "pending_files": library.pending_files(),
                        "outcomes": registry(),
                        "ai_briefs": briefs.info() if briefs else {"enabled": False},
                        "limits": {
                            "max_files": 100,
                            "max_file_bytes": 5_000_000,
                            "max_batch_bytes": 50_000_000,
                        },
                    }
                )
            if path.startswith("/api/cases/"):
                remainder = path.removeprefix("/api/cases/")
                parts = remainder.split("/")
                case_id = parts[0]
                if len(parts) == 2 and parts[1] in {"brief", "brief.pdf"}:
                    if briefs is None:
                        return self.reply(
                            {
                                "error": "Start the local server with --groq-demo to enable public-case drafts."
                            },
                            400,
                        )
                    try:
                        if parts[1] == "brief":
                            return self.reply(briefs.get(case_id))
                        raw = briefs.pdf(case_id)
                    except (KeyError, ValueError):
                        return self.reply(
                            {"error": "A verified case brief is unavailable."}, 404
                        )
                    if raw is None:
                        return self.reply(
                            {"error": "Generate the case brief before downloading."},
                            404,
                        )
                    self.send_response(200)
                    self.send_header("Content-Type", "application/pdf")
                    self.send_header(
                        "Content-Disposition",
                        'attachment; filename="case-brief-draft.pdf"',
                    )
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if len(parts) == 1:
                    payload = case_payload(library, case_id)
                    return (
                        self.reply(payload)
                        if payload
                        else self.reply({"error": "Case not found."}, 404)
                    )
                if parts[1] == "report":
                    try:
                        raw = focus_report(library, case_id).encode()
                    except KeyError:
                        return self.reply({"error": "Case not found."}, 404)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/markdown; charset=utf-8")
                    self.send_header(
                        "Content-Disposition",
                        f"attachment; filename*=UTF-8''{quote(case_id + '-case-review.md')}",
                    )
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if len(parts) == 3 and parts[1] == "files":
                    with library.connect() as conn:
                        row = conn.execute(
                            "SELECT f.content,f.name FROM matter_files m JOIN intake_files f ON f.id=m.intake_id WHERE m.case_id=? AND m.intake_id=?",
                            (case_id, parts[2]),
                        ).fetchone()
                    if row is None:
                        return self.reply({"error": "Source file not found."}, 404)
                    raw = bytes(row["content"])
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        mimetypes.guess_type(row["name"])[0]
                        or "application/octet-stream",
                    )
                    self.send_header(
                        "Content-Disposition",
                        f"attachment; filename*=UTF-8''{quote(Path(row['name']).name)}",
                    )
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
            if path.startswith("/api/"):
                return self.reply({"error": "API endpoint not found."}, 404)
            # Only the compiled application is served. SQLite, source inputs and the
            # repository itself are never inside this static root.
            relative = Path(path.lstrip("/"))
            candidate = (dist / relative).resolve()
            if not candidate.is_relative_to(dist.resolve()):
                return self.reply({"error": "Invalid path."}, 400)
            if not candidate.is_file():
                self.path = "/index.html"
            super().do_GET()

        def do_POST(self):
            if not self.allowed():
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.reply({"error": "Send a JSON request."}, 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST:
                    return self.reply(
                        {"error": "Request exceeds the upload limit."}, 413
                    )
                body = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path.startswith("/api/cases/") and path.endswith("/brief"):
                    if briefs is None:
                        return self.reply(
                            {
                                "error": "Public-case AI drafts are not enabled in this server."
                            },
                            400,
                        )
                    case_id = unquote(
                        path.removeprefix("/api/cases/").removesuffix("/brief")
                    )
                    return self.reply(
                        briefs.start(case_id, force=body.get("force") is True), 202
                    )
                if path.startswith("/api/precedents/") and path.endswith("/stage"):
                    original = research.original_pdf(
                        unquote(
                            path.removeprefix("/api/precedents/").removesuffix("/stage")
                        )
                    )
                    if original is None:
                        return self.reply(
                            {
                                "error": "Verified public PDF is unavailable on this computer."
                            },
                            404,
                        )
                    doc, raw = original
                    batch = library.stage(
                        {doc.id + ".pdf": raw}, provenance="public", source_note=doc.url
                    )
                    return self.reply(
                        {"batch_id": batch, "pending_files": library.pending_files()},
                        201,
                    )
                if path == "/api/intake":
                    files = body["files"]
                    if len(files) != len({f["name"] for f in files}):
                        raise LoaderError(
                            "Duplicate filenames: upload separately or give files distinct names."
                        )
                    raw = {
                        f["name"]: base64.b64decode(f["content"], validate=True)
                        for f in files
                    }
                    batch = library.stage(
                        raw,
                        provenance=body["provenance"],
                        source_note=body.get("source_note", ""),
                    )
                    return self.reply(
                        {"batch_id": batch, "pending_files": library.pending_files()},
                        201,
                    )
                if path == "/api/cases":
                    manifest = CaseManifest.model_validate(body["manifest"])
                    record = library.import_manifest(
                        manifest, body["file_ids"], defendant=body.get("defendant")
                    )
                    return self.reply(case_payload(library, record.case_id), 201)
                if path == "/api/focus-reviews":
                    return self.reply(library.save_decision(body), 201)
                if path == "/api/case-notes":
                    library.add_note(body["case_id"], body["note"])
                    return self.reply({"notes": library.notes(body["case_id"])}, 201)
                if path == "/api/demo":
                    seed_collection(library)
                    return self.reply({"cases": library.list()})
                return self.reply({"error": "API endpoint not found."}, 404)
            except ValidationError as exc:
                return self.reply(
                    {
                        "error": "; ".join(
                            f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                            for e in exc.errors(include_input=False)
                        )
                    },
                    400,
                )
            except (ValueError, LoaderError, KeyError, TypeError) as exc:
                return self.reply({"error": str(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8503)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--groq-demo",
        action="store_true",
        help="Explicitly enable Groq drafts for public/synthetic sources through a separate online worker",
    )
    args = parser.parse_args()
    dist = ROOT / "web" / "dist"
    if not (dist / "index.html").is_file():
        parser.error("Build the interface first: npm --prefix web run build")
    netguard.install()
    library = LibraryStore(CaseStore(args.db))
    briefs = None
    if args.groq_demo:
        from ratio.demo_briefs import DemoBriefService

        briefs = DemoBriefService(library)
    if args.demo:
        seed_collection(library)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler_for(library, dist, briefs=briefs)
    )
    print(
        f"Ratio case workspace: http://127.0.0.1:{args.port} (persistent local SQLite)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if briefs:
            briefs.close()


if __name__ == "__main__":
    main()
