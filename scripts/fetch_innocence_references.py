#!/usr/bin/env python3
"""A small explicit official source list, deduplicated into the existing corpus."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
from pypdf import PdfReader

from corpus_builder.normalize import clean_text
from corpus_builder.sources import REFUSED_HOSTS
from corpus_builder.store import BuildStore, RawItem
from ratio.precedent_schema import PrecedentDoc

SOURCES = (
    ("ccpr-1870-2009", "Sobhraj v. Nepal", "CCPR/C/99/D/1870/2009", "Nepal", 2010),
    ("ccpr-1520-2006", "Mwamba v. Zambia", "CCPR/C/98/D/1520/2006", "Zambia", 2010),
    (
        "ccpr-770-1997",
        "Gridin v. Russian Federation",
        "CCPR/C/69/D/770/1997",
        "Russian Federation",
        2000,
    ),
    (
        "ccpr-964-2001",
        "Saidov v. Tajikistan",
        "CCPR/C/81/D/964/2001",
        "Tajikistan",
        2004,
    ),
    (
        "ccpr-2526-2015",
        "Mirzakhodzhaev v. Kyrgyzstan",
        "CCPR/C/130/D/2526/2015",
        "Kyrgyzstan",
        2020,
    ),
    (
        "ccpr-2490-2014",
        "Hincapié Dávila v. Colombia",
        "CCPR/C/122/D/2490/2014/REV.1",
        "Colombia",
        2018,
    ),
    ("ccpr-3124-2018", "G.B. v. Latvia", "CCPR/C/133/D/3124/2018", "Latvia", 2021),
)


def main():
    store = BuildStore()
    seen = {d.raw_sha256 for d in store.documents()}
    report = []
    with httpx.Client(
        timeout=25,
        follow_redirects=True,
        headers={"User-Agent": "Ratio public legal reference downloader/0.1"},
    ) as client:
        for id, title, symbol, state, year in SOURCES:
            if store.document(id):
                report.append({"id": id, "status": "existing"})
                continue
            url = (
                "https://documents.un.org/api/symbol/access?l=en&s="
                + quote(symbol, safe="")
                + "&t=pdf"
            )
            try:
                if urlsplit(url).hostname in REFUSED_HOSTS:
                    report.append(
                        {
                            "id": id,
                            "status": "metadata only; automatic access disallowed; use configured local copies",
                        }
                    )
                    continue
                response = client.get(url)
                response.raise_for_status()
                raw = response.content
                if not raw.startswith(b"%PDF-") or len(raw) > 15_000_000:
                    raise ValueError("Not a bounded PDF")
                reader = PdfReader(io.BytesIO(raw))
                pages = [p.extract_text() or "" for p in reader.pages]
                text = clean_text("\n\n".join(pages))
                first = "".join(pages[:3]).replace(" ", "").replace("\n", "")
                if (
                    symbol.replace(" ", "") not in first
                    or "Human Rights Committee" not in text
                ):
                    raise ValueError("Identity/language validation failed")
                sha = hashlib.sha256(raw).hexdigest()
                if sha in seen:
                    report.append({"id": id, "status": "duplicate original"})
                    continue
                path = ROOT / "data/corpus/raw/ccpr_innocence" / f"{sha}.pdf"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                now = datetime.now(UTC).isoformat()
                store.record_raw(
                    RawItem(
                        url,
                        "ccpr_innocence",
                        sha,
                        "application/pdf",
                        path,
                        now,
                        title,
                        symbol,
                        state,
                        year,
                    )
                )
                doc = PrecedentDoc(
                    id=id,
                    kind="ccpr_views",
                    symbol=symbol,
                    title=title,
                    body="UN Human Rights Committee",
                    state=state,
                    year=year,
                    url=url,
                    retrieved_at=now,
                    raw_sha256=sha,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    language="en",
                    attribution="Official United Nations full decision, identified through OHCHR JURIS / UN document symbol; preserve attribution.",
                    text=text,
                )
                store.put_document(doc, url)
                seen.add(sha)
                report.append(
                    {
                        "id": id,
                        "status": "downloaded",
                        "pages": len(pages),
                        "url": url,
                        "sha256": sha,
                    }
                )
                print("Saved", id, len(pages), "pages", flush=True)
            except (httpx.HTTPError, ValueError) as exc:
                report.append(
                    {"id": id, "status": "unavailable", "reason": str(exc)[:150]}
                )
                print("Unavailable", id, type(exc).__name__, flush=True)
            time.sleep(2)
    out = ROOT / "data/real-demo/reference-download-audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
