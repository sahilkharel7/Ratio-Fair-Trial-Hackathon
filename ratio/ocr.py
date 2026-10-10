"""On-device OCR for scanned PDF pages, with macOS's Vision framework through a small compiled helper.

Nothing leaves the computer: the helper reads a temporary copy of the PDF and prints the recognised text.
It is compiled once from ratio/vision_ocr.swift into the git-ignored models/ocr/ folder. Where Swift or
Vision is unavailable, ocr_pages raises OCRUnavailable and the scanned page stays unread, as before.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ratio.paths import MODELS_DIR, PACKAGE_DIR

SOURCE = PACKAGE_DIR / "vision_ocr.swift"
TIMEOUT_SECONDS = 300
MAX_OUTPUT_BYTES = 20_000_000


class OCRUnavailable(RuntimeError):
    """On-device OCR cannot run on this computer."""


def enabled() -> bool:
    return os.environ.get("RATIO_OCR", "1") != "0"


def _helper() -> Path:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]
    binary = MODELS_DIR / "ocr" / f"ratio-ocr-{digest}"
    if binary.is_file():
        return binary
    compiler = shutil.which("swiftc")
    if compiler is None:
        raise OCRUnavailable("On-device OCR needs macOS with the Swift command line tools (xcode-select --install).")
    binary.parent.mkdir(parents=True, exist_ok=True)
    partial = binary.with_suffix(".partial")
    result = subprocess.run(
        [compiler, "-O", str(SOURCE), "-o", str(partial)], capture_output=True, text=True, timeout=TIMEOUT_SECONDS
    )
    if result.returncode != 0:
        raise OCRUnavailable(f"The OCR helper did not compile: {result.stderr.strip()[:300]}")
    partial.replace(binary)
    return binary


def ocr_pages(data: bytes) -> list[str]:
    """The recognised text of every page of a PDF, in page order."""
    if not enabled():
        raise OCRUnavailable("On-device OCR is turned off (RATIO_OCR=0).")
    return list(_ocr_cached(data))


@functools.lru_cache(maxsize=8)  # staging and case creation read the same scan: OCR it once (inputs are at most 5 MB)
def _ocr_cached(data: bytes) -> tuple[str, ...]:
    try:
        helper = _helper()
        with tempfile.TemporaryDirectory(prefix="ratio-ocr-") as folder:
            path = Path(folder) / "scan.pdf"
            path.write_bytes(data)
            result = subprocess.run([str(helper), str(path)], capture_output=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise OCRUnavailable("On-device OCR took too long for this document.") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise OCRUnavailable("On-device OCR could not run on this computer.") from exc
    if result.returncode != 0 or len(result.stdout) > MAX_OUTPUT_BYTES:
        raise OCRUnavailable("On-device OCR could not read this PDF.")
    try:
        pages = json.loads(result.stdout).get("pages")
    except (ValueError, AttributeError) as exc:
        raise OCRUnavailable("On-device OCR returned an unexpected reply.") from exc
    if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
        raise OCRUnavailable("On-device OCR returned an unexpected reply.")
    return tuple(pages)
