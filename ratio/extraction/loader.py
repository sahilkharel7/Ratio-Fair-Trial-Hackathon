"""Read a case folder (or uploaded files) into Documents, case metadata and coded rulings.

Inputs are untrusted: only .txt, .md and .pdf are accepted, sizes are capped, text must be
UTF-8, and manifest paths may not leave the case folder. Line endings are normalised to LF so
character offsets are stable on every OS. Synthetic documents must carry the SYNTHETIC marker
on their first line; the loader removes that line and records the flag instead.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import unicodedata
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import yaml
from pydantic import Field, ValidationError, model_validator

from ratio.extraction.align import find_all
from ratio.extraction.dates import has_explicit_year, parse_date_text
from ratio.extraction.segment import header_value
from ratio.schema import CaseMeta, DataProvenance, DocType, Document, Frozen, Ruling, SourceSpan, stable_id

SYNTHETIC_MARKER = "SYNTHETIC:"
MAX_DOCUMENT_BYTES = 5_000_000
MAX_MANIFEST_BYTES = 200_000
SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".pdf"})
MANIFEST_NAME = "case.yaml"


class LoaderError(ValueError):
    """A case folder or uploaded file cannot be used; the message says why."""


def _check_relative(path: str) -> str:
    pure = PurePosixPath(path)
    if not path.strip() or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"path {path!r} must stay inside the case folder")
    return path


class ManifestDocument(Frozen):
    path: str
    type: DocType
    title: str

    @model_validator(mode="after")
    def _safe_path(self) -> ManifestDocument:
        _check_relative(self.path)
        if PurePosixPath(self.path).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"{self.path}: only .txt, .md and .pdf documents are supported")
        return self


class CaseManifest(Frozen):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._'-]{0,63}$")
    title: str = Field(min_length=1)
    court: str = Field(min_length=1)
    charge_type: str = Field(min_length=1)
    data_provenance: DataProvenance
    synthetic: bool
    source_note: str | None = None
    documents: tuple[ManifestDocument, ...] = Field(min_length=1)
    rulings: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> CaseManifest:
        if self.synthetic != (self.data_provenance == "synthetic"):
            raise ValueError("'synthetic' must match data_provenance")
        paths = [doc.path for doc in self.documents]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate document paths")
        if sum(doc.type == "judgment" for doc in self.documents) > 1:
            raise ValueError("list one judgment per case; upload an appeal judgment as a separate case")
        if self.rulings is not None:
            _check_relative(self.rulings)
        return self


class _RulingEntry(Frozen):
    code: str
    doc: str
    quote: str = Field(min_length=3)
    judge: str = Field(min_length=1)
    date: dt.date | None = None
    value: float | None = None


class _RulingsFile(Frozen):
    synthetic: bool
    rulings: tuple[_RulingEntry, ...] = ()


def safe_yaml(text: str, name: str) -> object:
    """yaml.safe_load without anchors/aliases (a tiny alias bomb can otherwise exhaust memory)."""
    try:
        for token in yaml.scan(text):
            if isinstance(token, (yaml.AnchorToken, yaml.AliasToken)):
                raise LoaderError(f"{name}: YAML anchors and aliases are not allowed")
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise LoaderError(f"{name} is not valid YAML: {exc}") from exc


def parse_manifest(text: str) -> CaseManifest:
    if len(text.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise LoaderError(f"{MANIFEST_NAME} is too large")
    try:
        return CaseManifest.model_validate(safe_yaml(text, MANIFEST_NAME))
    except ValidationError as exc:
        raise LoaderError(f"{MANIFEST_NAME} is invalid: {exc.error_count()} problem(s): {exc.errors()[0]['msg']}") from exc


def decode_document(name: str, data: bytes) -> str:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise LoaderError(f"{name}: file type {suffix or '(none)'} is not supported (use .txt, .md or .pdf)")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise LoaderError(f"{name}: larger than {MAX_DOCUMENT_BYTES // 1_000_000} MB")
    if suffix == ".pdf":
        text = _pdf_text(name, data)
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise LoaderError(f"{name}: not valid UTF-8 text") from exc
    text = text.encode("utf-8", errors="replace").decode("utf-8")  # lone surrogates from PDFs
    text = unicodedata.normalize("NFC", text)  # one form for accents, so copies match character for character
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _pdf_text(name: str, data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        encrypted = reader.is_encrypted
        pages = [] if encrypted else [(page.extract_text() or "").strip() for page in reader.pages]
    except (PdfReadError, ValueError, OSError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise LoaderError(f"{name}: unreadable PDF ({type(exc).__name__})") from exc
    if encrypted:
        raise LoaderError(f"{name}: encrypted PDFs are not supported")
    return join_pdf_pages(pages)


def join_pdf_pages(pages: list[str]) -> str:
    """A page that ends mid-sentence continues on the next one; otherwise a new paragraph starts."""
    text = ""
    for page in (page for page in pages if page):
        text += ("" if not text else "\n\n" if text[-1] in ".!?:" else "\n") + page
    return text


def strip_synthetic_marker(text: str) -> tuple[str, bool]:
    first, _, rest = text.partition("\n")
    return (rest, True) if first.startswith(SYNTHETIC_MARKER) else (text, False)


def _span(doc_id: str, text: str, start: int, end: int) -> SourceSpan:
    return SourceSpan(doc_id=doc_id, start=start, end=end, text=text[start:end])


def _hearing_date(doc_id: str, doc_type: str, text: str) -> tuple[dt.date | None, SourceSpan | None]:
    """The 'Hearing date:' header of a monitoring note. A header that is present must give a full date."""
    if doc_type != "monitoring_note":
        return None, None
    located = header_value(text, "Hearing date")
    if located is None:
        return None, None
    value = text[located[0] : located[1]]
    parsed = parse_date_text(value)
    if parsed is None or parsed.precision not in ("date", "datetime") or not has_explicit_year(value):
        raise LoaderError(f"{doc_id}: 'Hearing date: {value}' must give day, month and year (e.g. 2 June 2025)")
    return parsed.value.date(), _span(doc_id, text, *located)


def build_documents(manifest: CaseManifest, files: Mapping[str, bytes]) -> tuple[Document, ...]:
    documents = []
    for entry in manifest.documents:
        if entry.path not in files:
            raise LoaderError(f"{entry.path}: listed in {MANIFEST_NAME} but not provided")
        text, marked = strip_synthetic_marker(decode_document(entry.path, files[entry.path]))
        if manifest.synthetic and not marked:
            raise LoaderError(f"{entry.path}: every synthetic document must start with the '{SYNTHETIC_MARKER}' marker line")
        if not manifest.synthetic and marked:
            raise LoaderError(f"{entry.path}: marked SYNTHETIC but the case is declared {manifest.data_provenance}")
        if not text.strip():
            raise LoaderError(f"{entry.path}: document is empty")
        doc_id = f"{manifest.case_id}/{entry.path}"
        date, date_span = _hearing_date(doc_id, entry.type, text)
        documents.append(
            Document(
                id=doc_id,
                case_id=manifest.case_id,
                path=entry.path,
                type=entry.type,
                title=entry.title,
                text=text,
                synthetic=manifest.synthetic,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                date=date,
                date_span=date_span,
            )
        )
    return tuple(documents)


def build_meta(manifest: CaseManifest, documents: tuple[Document, ...]) -> CaseMeta:
    """The presiding judge comes from a 'Presiding Judge:' header line, judgments first."""
    judge, judge_span = None, None
    ordered = sorted(documents, key=lambda doc: doc.type != "judgment")
    for doc in ordered:
        located = header_value(doc.text, "Presiding Judge")
        if located is not None:
            judge, judge_span = doc.text[located[0] : located[1]], _span(doc.id, doc.text, *located)
            break
    return CaseMeta(
        case_id=manifest.case_id,
        title=manifest.title,
        court=manifest.court,
        charge_type=manifest.charge_type,
        presiding_judge=judge,
        presiding_judge_span=judge_span,
        data_provenance=manifest.data_provenance,
        source_note=manifest.source_note,
        synthetic=manifest.synthetic,
    )


def build_rulings(manifest: CaseManifest, documents: tuple[Document, ...], data: bytes | None) -> tuple[Ruling, ...]:
    """Hand-coded rulings: every quote must occur exactly once in its document."""
    if data is None:
        return ()
    try:
        parsed = _RulingsFile.model_validate(safe_yaml(data.decode("utf-8-sig"), str(manifest.rulings)))
    except (UnicodeDecodeError, ValidationError) as exc:
        raise LoaderError(f"{manifest.rulings} is invalid: {exc}") from exc
    if parsed.synthetic != manifest.synthetic:
        raise LoaderError(f"{manifest.rulings}: 'synthetic' does not match {MANIFEST_NAME}")
    by_path = {doc.path: doc for doc in documents}
    rulings = []
    for number, entry in enumerate(parsed.rulings, start=1):
        doc = by_path.get(entry.doc)
        if doc is None:
            raise LoaderError(f"{manifest.rulings} #{number}: unknown document {entry.doc!r}")
        positions = find_all(doc.text, entry.quote)
        if len(positions) != 1:
            problem = "not found" if not positions else f"found {len(positions)} times; make it unique"
            raise LoaderError(f"{manifest.rulings} #{number}: quote {problem} in {entry.doc}: {entry.quote!r}")
        start = positions[0]
        rulings.append(
            Ruling(
                id=stable_id(doc.id, start, entry.code, "ruling"),
                case_id=manifest.case_id,
                code=entry.code,
                judge_name=entry.judge,
                date=entry.date,
                value=entry.value,
                span=_span(doc.id, doc.text, start, start + len(entry.quote)),
            )
        )
    return tuple(rulings)


def read_uploaded_files(uploaded: Mapping[str, bytes]) -> tuple[CaseManifest, dict[str, bytes]]:
    """The manifest and listed files of an uploaded folder, read in memory (nothing is written to disk).

    Keys are the browser's relative paths ("venn-case/notes/hearing_1.txt"); the folder that
    holds case.yaml is the case root, and only files the manifest lists are kept.
    """
    names = {PurePosixPath(name.removeprefix("./")).as_posix(): name for name in uploaded}
    manifests = sorted((p for p in names if PurePosixPath(p).name == MANIFEST_NAME), key=lambda p: (p.count("/"), p))
    if not manifests:
        raise LoaderError(f"the uploaded folder has no {MANIFEST_NAME}")
    if len(manifests) > 1 and manifests[0].count("/") == manifests[1].count("/"):
        raise LoaderError(f"the uploaded folder has several {MANIFEST_NAME} files at the same level")
    root = PurePosixPath(manifests[0]).parent
    data = uploaded[names[manifests[0]]]
    if len(data) > MAX_MANIFEST_BYTES:
        raise LoaderError(f"{MANIFEST_NAME} is too large")
    try:
        manifest = parse_manifest(data.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise LoaderError(f"{MANIFEST_NAME} is not valid UTF-8") from exc
    listed = [doc.path for doc in manifest.documents] + ([manifest.rulings] if manifest.rulings else [])
    files: dict[str, bytes] = {}
    for name in listed:
        key = (root / name).as_posix() if str(root) != "." else name
        if key not in names:
            raise LoaderError(f"{name}: listed in {MANIFEST_NAME} but not in the uploaded folder")
        files[name] = uploaded[names[key]]
    return manifest, files


def read_case_folder(folder: Path) -> tuple[CaseManifest, dict[str, bytes]]:
    root = Path(folder).resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise LoaderError(f"{folder}: no {MANIFEST_NAME} found")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise LoaderError(f"{MANIFEST_NAME} is too large")
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise LoaderError(f"{MANIFEST_NAME} is not valid UTF-8") from exc
    manifest = parse_manifest(manifest_text)
    names = [doc.path for doc in manifest.documents] + ([manifest.rulings] if manifest.rulings else [])
    files: dict[str, bytes] = {}
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise LoaderError(f"{name}: path leaves the case folder")
        if not path.is_file():
            raise LoaderError(f"{name}: file not found in {folder}")
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise LoaderError(f"{name}: larger than {MAX_DOCUMENT_BYTES // 1_000_000} MB")
        files[name] = path.read_bytes()
    return manifest, files
