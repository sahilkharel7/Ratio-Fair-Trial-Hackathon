"""Turns each fetched file into a PrecedentDoc: its text, cleaned just enough that a quote copied from
it is found again character for character, plus its id, provenance and hashes.

The text is never reflowed or dehyphenated ("witnes-\\nses" stays as it is), because the model quotes
this stored text and every quote is checked against it. Only English documents are kept.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from unidecode import unidecode

from corpus_builder.fetch import HTML, MAX_BYTES, PDF, Issue
from corpus_builder.sources import Source
from corpus_builder.store import BuildStore, RawItem
from ratio.precedent_schema import PrecedentDoc, PrecedentKind
from ratio.schema import Frozen

TEXT = "text/plain"  # a .txt or .md file uploaded into a collection
MIN_TEXT_CHARS = 500
MIN_ASCII_LETTER_SHARE = 0.5  # below this, the text is mostly not ASCII letters: not English
MIN_ENGLISH_WORD_SHARE = 0.05  # English prose has far more of these words (about 20%)
ENGLISH_WORDS = frozenset({"the", "and", "of", "to", "that", "with", "which", "was", "were", "for", "this", "has", "have", "been"})
SYMBOL_SEARCH_CHARS = 3000  # a document's own symbol is on its first page; later ones cite other cases
MAX_ID_CHARS = 80
MAX_TITLE_CHARS = 200
UPLOADS_PATH = "/wp-content/uploads/"  # where TrialWatch pages link their report PDFs

_CCPR_SYMBOL = re.compile(r"CCPR/C/[^/\s]+/D/(\d+(?:-\d+)?)/(\d{4})")
_WGAD_SYMBOL = re.compile(r"A/HRC/WGAD/(\d{4})/(\d+)")
_SYMBOLS: dict[str, re.Pattern[str]] = {"ccpr_views": _CCPR_SYMBOL, "wgad_opinion": _WGAD_SYMBOL}
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # control characters other than tab and line feed
_TRAILING_SPACE = re.compile(r"[^\S\n]+$", re.MULTILINE)
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")

_DROPPED = frozenset({"script", "style", "nav", "header", "footer", "form", "noscript", "template", "svg", "title"})
_BLOCKS = frozenset({
    "address", "article", "aside", "blockquote", "dd", "details", "div", "dl", "dt", "figcaption", "figure",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "main", "ol", "p", "pre", "section", "summary", "table", "td",
    "th", "tr", "ul",
})  # fmt: skip
_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})
_TEXT_REGIONS = ("article", "entry", "main", "page")  # in order of preference
_TITLE_REGIONS = ("h1", "title")
_PARAGRAPH = "\n\n"


class NormalizeReport(Frozen):
    documents: tuple[str, ...] = ()  # precedent ids written
    skipped: tuple[Issue, ...] = ()
    discovered_pdfs: tuple[str, ...] = ()  # report PDFs linked from TrialWatch pages, not fetched yet


class Skip(Exception):
    """Why a fetched file gives no document."""


@dataclass(frozen=True)
class HtmlPage:
    text: str
    title: str | None
    links: tuple[str, ...]  # every <a href>, as written, in document order


# --- the step -------------------------------------------------------------------------------------


def normalize_all(store: BuildStore, sources: Sequence[Source]) -> NormalizeReport:
    """Normalise every fetched file of a known source and store the documents; report the skips."""
    by_id = {source.id: source for source in sources}
    items = store.raw_items()
    documents: list[str] = []
    skipped: list[Issue] = []
    links: set[str] = set()
    owners: dict[str, str] = {}  # precedent id -> the URL it came from
    for item in items:
        try:
            doc, pdfs = normalize_item(item, _source_of(item, by_id))
            if doc.id in owners:
                raise Skip(f"same id {doc.id} as {owners[doc.id]}")
        except Skip as exc:
            skipped.append(Issue(url=item.url, reason=str(exc)))
            continue
        store.put_document(doc, item.url)
        owners[doc.id] = item.url
        documents.append(doc.id)
        links.update(pdfs)
    fetched = {item.url for item in items}
    return NormalizeReport(documents=tuple(documents), skipped=tuple(skipped), discovered_pdfs=tuple(sorted(links - fetched)))


def normalize_item(item: RawItem, source: Source) -> tuple[PrecedentDoc, tuple[str, ...]]:
    """The document from one fetched file, and the report PDFs its page links to; raises Skip."""
    data = _read_raw(item)
    page_title, pdfs = None, ()
    if item.content_type == PDF:
        raw_text = pdf_text(data)
    elif item.content_type == HTML:
        page = html_page(data.decode("utf-8-sig", errors="replace"))
        raw_text, page_title = page.text, page.title
        pdfs = report_pdfs(page.links, item.url, source) if source.kind == "trialwatch_report" else ()
    elif item.content_type == TEXT:
        raw_text = data.decode("utf-8-sig", errors="replace")
    else:
        raise Skip(f"unsupported content type {item.content_type}")
    text = clean_text(raw_text)
    problem = language_problem(text)
    if problem:
        raise Skip(problem)
    return _document(item, source, text, data, page_title), pdfs


def _document(item: RawItem, source: Source, text: str, data: bytes, page_title: str | None) -> PrecedentDoc:
    symbol = item.symbol or find_symbol(source.kind, text)
    precedent_id = make_id(source.kind, item.url, symbol)
    if precedent_id is None:
        raise Skip(f"no id: a {source.kind} needs its symbol (set it on the seed)" if source.kind in _SYMBOLS else "no id: no symbol and an empty URL slug")
    try:
        return PrecedentDoc(
            id=precedent_id, kind=source.kind, symbol=symbol, title=_title(item, page_title, symbol, text), body=source.body,
            state=item.state, year=item.year or _symbol_year(source.kind, symbol), url=item.url, retrieved_at=item.fetched_at,
            raw_sha256=hashlib.sha256(data).hexdigest(), text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            language="en", attribution=source.attribution, text=text, private=bool(getattr(source, "private", False)),
        )  # fmt: skip
    except ValidationError as exc:
        raise Skip(f"invalid document: {exc.errors()[0]['msg']}") from None


def _source_of(item: RawItem, sources: dict[str, Source]) -> Source:
    if item.source_id not in sources:
        raise Skip(f"source {item.source_id!r} is not in sources.yaml")
    return sources[item.source_id]


def _read_raw(item: RawItem) -> bytes:
    try:
        if item.path.stat().st_size > MAX_BYTES:
            raise Skip(f"raw file larger than {MAX_BYTES} bytes")
        data = item.path.read_bytes()
    except OSError as exc:
        raise Skip(f"cannot read raw file {item.path}: {exc.strerror or exc}") from None
    if hashlib.sha256(data).hexdigest() != item.sha256:
        raise Skip("raw file changed since it was fetched (sha256 differs)")
    return data


# --- text -----------------------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Remove soft hyphens, NUL and other control characters, normalise line endings and Unicode form
    (NFC), strip trailing spaces, and let no more than one blank line stand between paragraphs."""
    text = text.encode("utf-8", errors="replace").decode("utf-8")  # lone surrogates from PDFs
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = _CONTROL.sub("", text)
    text = _TRAILING_SPACE.sub("", text)
    return _EXTRA_BLANK_LINES.sub("\n\n", text).strip()


def pdf_text(data: bytes) -> str:
    """pypdf's text of each page, pages joined by a blank line."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise Skip("encrypted PDF")
        return _PARAGRAPH.join(page.extract_text() or "" for page in reader.pages)
    except (PdfReadError, ValueError, OSError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise Skip(f"unreadable PDF ({type(exc).__name__})") from None


def language_problem(text: str) -> str | None:
    """Why this text is not kept as an English document, or None."""
    if len(text) < MIN_TEXT_CHARS:
        return f"text too short ({len(text)} characters, at least {MIN_TEXT_CHARS} needed)"
    letters = [char for char in text if char.isalpha()]
    if sum(char.isascii() for char in letters) < MIN_ASCII_LETTER_SHARE * len(letters):
        return "not English: mostly not ASCII letters"
    words = re.findall(r"[a-z]+", text.lower())
    if sum(word in ENGLISH_WORDS for word in words) < MIN_ENGLISH_WORD_SHARE * len(words):
        return "not English: few common English words"
    return None


# --- ids and metadata -----------------------------------------------------------------------------


def find_symbol(kind: PrecedentKind, text: str) -> str | None:
    """The UN symbol near the start of a Views or opinion, when the seed did not give one."""
    pattern = _SYMBOLS.get(kind)
    match = pattern.search(text[:SYMBOL_SEARCH_CHARS]) if pattern else None
    return match.group(0) if match else None


def make_id(kind: PrecedentKind, url: str, symbol: str | None) -> str | None:
    """ccpr-<communication>-<year>, wgad-<year>-<number>, or tw-<slug>: a TrialWatch seed's symbol (the
    slug of its case page) when it has one, else the URL's file name."""
    if kind == "ccpr_views":
        match = _CCPR_SYMBOL.search(symbol or "")
        return f"ccpr-{match[1]}-{match[2]}" if match else None
    if kind == "wgad_opinion":
        match = _WGAD_SYMBOL.search(symbol or "")
        return f"wgad-{match[1]}-{int(match[2])}" if match else None
    slug = slugify(symbol or "") or url_slug(url)
    if kind == "court_judgment":
        return f"court-{slug}"[:MAX_ID_CHARS].rstrip("-") if slug else None
    prefix = "col" if kind == "collection_document" else "tw"  # col-<collection>-<file> for an uploaded document
    return f"{prefix}-{slug}"[:MAX_ID_CHARS].rstrip("-") if slug else None


def url_slug(url: str) -> str:
    """The last path segment, without .pdf or .html, as a slug (see slugify)."""
    segments = [segment for segment in urlsplit(url).path.split("/") if segment]
    if not segments:
        return ""
    return slugify(re.sub(r"\.(?:pdf|html?)$", "", unquote(segments[-1]), flags=re.IGNORECASE))


def slugify(name: str) -> str:
    """Lower-case ASCII letters and digits, in words joined by single hyphens ("" when none)."""
    return re.sub(r"[^a-z0-9]+", "-", unidecode(name).lower()).strip("-")


def _symbol_year(kind: PrecedentKind, symbol: str | None) -> int | None:
    """An opinion's symbol carries its year; a Views symbol carries the communication's, so not that."""
    match = _WGAD_SYMBOL.search(symbol or "") if kind == "wgad_opinion" else None
    return int(match[1]) if match else None


def _title(item: RawItem, page_title: str | None, symbol: str | None, text: str) -> str:
    for candidate in (item.title, page_title, symbol):
        if candidate and candidate.strip():
            return candidate.strip()[:MAX_TITLE_CHARS]
    return next(line.strip() for line in text.splitlines() if line.strip())[:MAX_TITLE_CHARS]


# --- HTML -----------------------------------------------------------------------------------------


def html_page(markup: str) -> HtmlPage:
    """The main article text of a page: <article>, else .entry-content, else <main>, else the whole page,
    always without scripts, styles, navigation, headers, footers and forms. Parsed only."""
    parser = _PageParser()
    parser.feed(markup)
    parser.close()
    texts = {region: _region_text(parser.parts[region]) for region in _TEXT_REGIONS}
    text = next((texts[region] for region in _TEXT_REGIONS if texts[region]), "")
    title = _one_line(parser.parts["h1"]) or _one_line(parser.parts["title"]) or None
    return HtmlPage(text=text, title=title, links=tuple(parser.links))


def report_pdfs(links: Sequence[str], page_url: str, source: Source) -> tuple[str, ...]:
    """PDFs under /wp-content/uploads/ that a page links to, on the source's allowlist."""
    found = []
    for href in links:
        parts = urlsplit(urljoin(page_url, href.strip()))
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
        if UPLOADS_PATH in parts.path and parts.path.lower().endswith(".pdf") and source.allows(url):
            found.append(url)
    return tuple(dict.fromkeys(found))


def _region_text(parts: list[str]) -> str:
    text = re.sub(r" *\n *", "\n", "".join(parts))
    text = re.sub(r" {2,}", " ", text)
    return _EXTRA_BLANK_LINES.sub("\n\n", text).strip()


def _one_line(parts: list[str]) -> str:
    return " ".join("".join(parts).split())


class _PageParser(HTMLParser):
    """Collects the text of each candidate region of a page as it streams past."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: dict[str, list[str]] = {name: [] for name in (*_TEXT_REGIONS, *_TITLE_REGIONS)}
        self.links: list[str] = []
        self.stack: list[str] = []  # the open elements
        self.opened: dict[str, int] = {"page": -1}  # region -> depth of the element that opened it
        self.closed: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"] or "")
        if tag in _VOID:
            if tag in ("br", "hr"):
                self._emit("\n" if tag == "br" else _PARAGRAPH)
            return
        self.stack.append(tag)
        classes = (attributes.get("class") or "").split()
        starts = {"article": tag == "article", "entry": "entry-content" in classes, "main": tag == "main", "h1": tag == "h1", "title": tag == "title"}
        for region, starting in starts.items():
            if starting and region not in self.opened:
                self.opened[region] = len(self.stack) - 1
        if tag in _BLOCKS:
            self._emit(_PARAGRAPH)

    def handle_endtag(self, tag: str) -> None:
        if tag not in self.stack:
            return
        while self.stack:
            open_tag = self.stack[-1]
            if open_tag in _BLOCKS:
                self._emit(_PARAGRAPH)
            self.stack.pop()
            depth = len(self.stack)
            self.closed.update(region for region, start in self.opened.items() if start == depth)
            if open_tag == tag:
                return

    def handle_data(self, data: str) -> None:
        self._emit(re.sub(r"\s+", " ", data))

    def _emit(self, chunk: str) -> None:
        dropped = any(tag in _DROPPED for tag in self.stack)
        for region in self.opened.keys() - self.closed:
            if region in _TITLE_REGIONS or not dropped:  # a title counts even inside a dropped <header>
                self.parts[region].append(chunk)
