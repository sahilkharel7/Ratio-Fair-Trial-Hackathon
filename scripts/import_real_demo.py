#!/usr/bin/env python3
"""Idempotent real-case intake using supplied sources, never supplied model answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pypdf import PdfReader

from ratio.extraction.loader import CaseManifest, safe_yaml
from ratio.library import LibraryStore


def canonical(text):
    return re.sub(r"\W", "", unicodedata.normalize("NFKC", text).lower())


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def import_maseko(root, library):
    folder = root / "maseko-case-for-ratio"
    metadata = safe_yaml((folder / "case.yaml").read_text(), "case.yaml")
    case_id = metadata["case_id"]
    if library.cases.load_case(case_id):
        print("Existing case preserved:", case_id)
        return case_id
    report_path = root / "maseko-downloads/icj_report.pdf"
    report = canonical(
        "\n".join(page.extract_text() or "" for page in PdfReader(report_path).pages)
    )
    capture = max(
        strings(json.loads((root / "maseko-downloads/judgment_exa.json").read_text())),
        key=len,
    )
    judgment = canonical(capture)
    files = {}
    audit = {
        "case_id": case_id,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "documents": [],
        "excluded_inputs": [
            "icj_answer_key.json",
            "record.json",
            "analysis.json",
            "report conclusions",
        ],
    }
    for declared in metadata["documents"]:
        path = declared["path"]
        raw = (folder / path).read_text()
        parts = raw.split("\n\n")
        if declared["type"] == "monitoring_note":
            kept = [parts[0]]
            verified = omitted = 0
            for part in parts[1:]:
                if not part.strip():
                    continue
                if canonical(part) in report:
                    kept.append(part)
                    verified += 1
                else:
                    omitted += 1
            raw = "\n\n".join(kept) + "\n"
            audit["documents"].append(
                {
                    "path": path,
                    "verified_paragraphs": verified,
                    "omitted_unverified_paragraphs": omitted,
                }
            )
        else:
            failures = [
                part[:80]
                for part in parts
                if len(part) > 30 and canonical(part) not in judgment
            ]
            if failures:
                raise ValueError(
                    f"Cannot verify {path} against the supplied judgment capture."
                )
            audit["documents"].append(
                {
                    "path": path,
                    "verified_paragraphs": sum(len(p) > 30 for p in parts),
                    "omitted_unverified_paragraphs": 0,
                }
            )
        files[path] = raw.encode()
    metadata["source_note"] += (
        " Import verification: judgment and indictment checked against the supplied EswatiniLII capture; ICJ note paragraphs checked against the original PDF ignoring layout/punctuation. Unmatched report paragraphs were omitted. Source fidelity audit stored locally. These are published-source excerpts, not independent hearing transcripts. Case processing remains confidential/local."
    )
    manifest = CaseManifest.model_validate(metadata)
    batch = library.stage(files, provenance="public", source_note=manifest.source_note)
    ids = {
        f["name"]: f["id"] for f in library.pending_files() if f["batch_id"] == batch
    }
    record = library.import_manifest(
        manifest, ids, defendant="Thulani Maseko and Bheki Makhubu"
    )
    out = ROOT / "data/real-demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / (case_id + "-source-audit.json")).write_text(json.dumps(audit, indent=2))
    print(
        "Imported and registered:",
        record.case_id,
        "documents:",
        len(record.documents),
        "observations:",
        len(record.observations),
    )
    print(
        "Verified report paragraphs:",
        sum(
            d["verified_paragraphs"]
            for d in audit["documents"]
            if d["path"].startswith("notes/")
        ),
    )
    print(
        "Unmatched report paragraphs omitted:",
        sum(d["omitted_unverified_paragraphs"] for d in audit["documents"]),
    )
    return record.case_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder", type=Path, default=Path.home() / "Downloads/Ratio-real-cases"
    )
    args = parser.parse_args()
    import_maseko(args.folder, LibraryStore())


if __name__ == "__main__":
    main()
