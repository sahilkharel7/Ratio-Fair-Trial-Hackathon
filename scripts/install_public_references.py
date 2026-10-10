#!/usr/bin/env python3
"""Verify and install the versioned public-only reference snapshot, preserving existing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported reference snapshot version.")
    for relative, expected in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or digest(path) != expected:
            raise ValueError(f"Snapshot checksum/path validation failed: {relative}")
    ids = set()
    for doc in manifest["documents"]:
        id = doc["id"]
        if (
            doc["private"]
            or not doc["url"].startswith("https://")
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,80}", id)
        ):
            raise ValueError("Snapshot contains a private/unidentified reference.")
        if id in ids:
            raise ValueError("Duplicate reference identifier.")
        ids.add(id)
        original = bundle / "originals" / (id + ".pdf")
        text = json.loads((bundle / "texts" / (id + ".json")).read_text())
        aid = json.loads((bundle / "reading-aids" / (id + ".json")).read_text())
        if digest(original) != doc[
            "raw_sha256"
        ] or not original.read_bytes().startswith(b"%PDF-"):
            raise ValueError(f"Original PDF changed: {id}")
        if (
            text["private"]
            or hashlib.sha256(text["text"].encode()).hexdigest() != doc["text_sha256"]
        ):
            raise ValueError(f"Public source text changed: {id}")
        if (
            aid["document_id"] != id
            or aid["text_sha256"] != doc["text_sha256"]
            or digest(bundle / "reading-aids" / (id + ".pdf")) != aid["pdf_sha256"]
        ):
            raise ValueError(f"Reading aid changed: {id}")
        if any(section["quote"] not in text["text"] for section in aid["sections"]):
            raise ValueError(f"Reading-aid quote missing from source: {id}")
    return manifest


def install(bundle, destination):
    manifest = validate(bundle)
    index = destination / "precedents.db"
    source_index = bundle / "precedents.db"
    if index.exists() and digest(index) != digest(source_index):
        raise ValueError(
            "A different reference index already exists; it was preserved. Choose --destination for a separate installation."
        )
    planned = [(source_index, index)]
    for doc in manifest["documents"]:
        id = doc["id"]
        planned.append(
            (
                bundle / "originals" / (id + ".pdf"),
                destination / "raw" / "public_snapshot" / (doc["raw_sha256"] + ".pdf"),
            )
        )
        for suffix in (".pdf", ".json"):
            planned.append(
                (
                    bundle / "reading-aids" / (id + suffix),
                    destination / "briefs" / (id + suffix),
                )
            )
    # Check every conflict before making changes. Custom/private runtime data is never cleared.
    for source, target in planned:
        if target.exists() and digest(target) != digest(source):
            raise ValueError(
                f"Existing reference asset was preserved: {target}. Use a separate destination."
            )
    for source, target in planned:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return len(manifest["documents"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/corpus")
    args = parser.parse_args()
    try:
        count = install(ROOT / "public-references", args.destination)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Reference snapshot not installed: {exc}\n")
    print(
        f"Verified and installed {count} public originals and reading aids in {args.destination}"
    )


if __name__ == "__main__":
    main()
