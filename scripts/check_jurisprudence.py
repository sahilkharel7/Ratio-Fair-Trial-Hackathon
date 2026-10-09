"""Online check of the jurisprudence corpus (ratio/config/jurisprudence.yaml) against its sources.

Ratio itself never downloads anything. Run this while online after any change to the corpus:

    python scripts/check_jurisprudence.py

It downloads each official General Comment PDF listed under `sources`, extracts its text, and checks
that every `quote` (each segment between "…" marks, in order) and every `cited_as` occurs in its
source. Text is compared with all whitespace removed, because PDF extraction breaks words and lines
at random ("accu sed"). Exits 1 if anything is not found.

This script deliberately does not import the `ratio` package, which installs the network guard.
"""

from __future__ import annotations

import io
import re
import sys
import urllib.request
from pathlib import Path

import yaml
from pypdf import PdfReader

CORPUS = Path(__file__).resolve().parent.parent / "ratio" / "config" / "jurisprudence.yaml"
STEELMAN = CORPUS.with_name("steelman.yaml")  # the grounds the State's reply may rest on quote the same documents
ALLOWED_HOSTS = ("https://documents.un.org/",)


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text)


def source_text(url: str) -> str:
    if not url.startswith(ALLOWED_HOSTS):
        raise SystemExit(f"refusing to download {url}: not an official UN document address")
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - https only, allowlisted host
        data = response.read()
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)


def found_in_order(segments: list[str], text: str) -> bool:
    position = 0
    for segment in segments:
        position = text.find(segment, position)
        if position < 0:
            return False
        position += len(segment)
    return True


def main() -> int:
    corpus = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))
    texts = {key: squeeze(source_text(source["url"])) for key, source in corpus["sources"].items()}
    missing = []
    for entry in corpus["entries"]:
        text = texts[entry["source"]]
        if entry["kind"] == "general_comment":
            segments = [squeeze(part) for part in entry["quote"].split("…") if part.strip()]
            ok = found_in_order(segments, text)
        else:
            ok = squeeze(entry["cited_as"]) in text and entry["communication"] in entry["cited_as"]
        print(f"  {'ok     ' if ok else 'MISSING'} {entry['id']} ({corpus['sources'][entry['source']]['symbol']}, {entry['pinpoint']})")
        if not ok:
            missing.append(entry["id"])
    print(f"{len(corpus['entries']) - len(missing)} of {len(corpus['entries'])} entries found in their sources")
    grounds = [g for listed in yaml.safe_load(STEELMAN.read_text(encoding="utf-8"))["grounds"].values() for g in listed if g.get("quote")]
    for ground in grounds:
        ok = found_in_order([squeeze(part) for part in ground["quote"].split("…") if part.strip()], texts[ground["source"]])
        print(f"  {'ok     ' if ok else 'MISSING'} steelman ground {ground['id']} ({corpus['sources'][ground['source']]['symbol']}, {ground['pinpoint']})")
        if not ok:
            missing.append(ground["id"])
    print(f"{sum(g['id'] not in missing for g in grounds)} of {len(grounds)} steelman ground quotes found in their sources")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
