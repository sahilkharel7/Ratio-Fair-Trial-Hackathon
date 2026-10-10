#!/usr/bin/env python3
"""Download three original court judgments held out from the installed case and reference libraries."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pypdf import PdfReader

from corpus_builder.fetch import Fetcher
from corpus_builder.sources import Seed, Source
from corpus_builder.store import BuildStore

CASES = (
    {
        "file": "01-Butkevicius-v-Lithuania-2002-original.pdf",
        "title": "Butkevičius v. Lithuania",
        "application": "48297/99",
        "date": "26 March 2002",
        "state": "Lithuania",
        "year": 2002,
        "url": "https://cambodia.ohchr.org/sites/default/files/echrsource/Butkevi%C4%8Dius%20v.%20Lithuania%20%5B26%20Mar%202003%5D%20%5BEN%5D.pdf",
    },
    {
        "file": "02-Nestak-v-Slovakia-2007-original.pdf",
        "title": "Nešťák v. Slovakia",
        "application": "65559/01",
        "date": "27 February 2007",
        "state": "Slovakia",
        "year": 2007,
        "url": "https://cambodia.ohchr.org/sites/default/files/echrsource/Ne%C5%A1%C5%A5%C3%A1k%20v.%20Slovakia%20%5B27%20Feb%202007%5D%20%5BEN%5D.pdf",
    },
    {
        "file": "03-Minelli-v-Switzerland-1983-original.pdf",
        "title": "Minelli v. Switzerland",
        "application": "8660/79",
        "date": "25 March 1983",
        "state": "Switzerland",
        "year": 1983,
        "url": "https://cambodia.ohchr.org/sites/default/files/echrsource/Minelli%20v.%20Switzerland%20%5B25%20Mar%201983%5D%20%5BEN%5D.pdf",
    },
)


def main():
    out = ROOT / "output/demo-upload-pack"
    out.mkdir(parents=True, exist_ok=True)
    source = Source(
        id="held_out_court_judgments",
        kind="court_judgment",
        name="Demo holdout: original ECHR judgments",
        body="European Court of Human Rights",
        attribution="Original English court judgments reproduced by OHCHR Cambodia.",
        terms_url="https://www.ohchr.org/en/copyright",
        terms_checked="2026-10-10",
        terms_summary="Public UN-hosted reproductions; preserve original text and attribution.",
        allowed_prefixes=("https://cambodia.ohchr.org/",),
        min_interval_s=3,
        seeds=tuple(
            Seed(url=c["url"], title=c["title"], state=c["state"], year=c["year"])
            for c in CASES
        ),
    )
    manifest = []
    with tempfile.TemporaryDirectory(prefix="ratio-demo-pack-") as temp:
        fetcher = Fetcher(BuildStore(Path(temp) / "downloads.db"))
        for case in CASES:
            path = out / case["file"]
            raw = (
                path.read_bytes()
                if path.exists()
                else fetcher.get(source, case["url"]).data
            )
            if not raw.startswith(b"%PDF-"):
                raise ValueError("Source returned something other than a PDF.")
            pdf = PdfReader(io.BytesIO(raw))
            first = " ".join(p.extract_text() or "" for p in pdf.pages[:4])
            if case["application"] not in first or len(pdf.pages) < 5:
                raise ValueError(
                    "Downloaded PDF does not identify the expected court case."
                )
            path.write_bytes(raw)
            manifest.append(
                {
                    **case,
                    "pages": len(pdf.pages),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "installed": False,
                }
            )
            print(
                f"Saved {case['file']}: {len(pdf.pages)} pages; application {case['application']}"
            )
    (out / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    guide = ROOT / "docs/DEMO-UPLOAD-PACK.md"
    if guide.exists():
        (out / "START-HERE.md").write_bytes(guide.read_bytes())
    archive = out / "Ratio-real-case-demo-pack.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.suffix in {".pdf", ".md", ".json"}:
                z.write(p, p.name)
    print(
        "Held-out originals remain outside both SQLite case intake and the installed reference corpus."
    )


if __name__ == "__main__":
    main()
