"""Fetches the public documents of each source in sources.yaml into data/corpus/raw/, politely.

- Only URLs on a source's allowed prefixes are requested, and a redirect that leaves them is refused.
- robots.txt is read with our user agent and obeyed, for every redirect target too. A missing one
  (4xx) allows everything; a host whose robots.txt cannot be read (5xx, network error) is skipped.
- One request per host every max(min_interval_s, Crawl-delay) seconds; 429, 5xx and timeouts are
  retried three times, each wait twice the one before.
- Only the seeds, and the sitemap pages that match url_pattern, are fetched: links in pages never are.
- Downloads are untrusted data: capped at 30 MB, saved under their sha256, parsed later, never executed.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import http.client
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

from corpus_builder.sources import LOCAL_DIR, LocalFile, Seed, Source
from corpus_builder.store import RAW_DIR, BuildStore, RawItem
from ratio.schema import Frozen

UA = "RatioCorpusBuilder/0.1 (+https://github.com/sahilkharel7/Ratio-Fair-Trial-Hackathon)"
MAX_BYTES = 30 * 1024 * 1024  # a larger download is refused
RETRIES = 3  # after the first try, on 429, 5xx and timeouts
TIMEOUT_S = 60.0
MAX_SITEMAPS = 50  # sitemap files read per source, nested ones included
PDF, HTML = "application/pdf", "text/html"  # RawItem.content_type
EXTENSIONS = {PDF: "pdf", HTML: "html"}
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_LOCAL_TYPES = {".pdf": PDF, ".html": HTML, ".htm": HTML}
_ACCEPT = "application/pdf, text/html, application/xml;q=0.9, text/plain;q=0.5"

log = logging.getLogger(__name__)


class FetchRefused(Exception):
    """Policy says no: off the allowlist, disallowed by robots.txt, redirected away, too large or not a document."""


class FetchFailed(Exception):
    """The server or the network did not deliver."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class Issue(Frozen):
    url: str
    reason: str


class FetchReport(Frozen):
    source_id: str
    fetched: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()  # already in the work database
    local: tuple[str, ...] = ()  # local files ingested
    discovered: tuple[str, ...] = ()  # sitemap pages matching url_pattern
    refused: tuple[Issue, ...] = ()
    failed: tuple[Issue, ...] = ()


@dataclass(frozen=True)
class Response:
    url: str  # after redirects
    data: bytes
    content_type: str  # from the header, lower case, without parameters


class Opener(Protocol):
    def open(self, request: urllib.request.Request, timeout: float = ...) -> Any: ...


Check = Callable[[str], None]  # raises FetchRefused when a URL may not be fetched


class AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to an address its check allows: the Fetcher's checks the allowlist (so
    never from https to http) and the robots.txt of the target's host."""

    max_redirections = 5

    def __init__(self, check: Check) -> None:
        super().__init__()
        self._check = check

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> urllib.request.Request | None:
        try:
            self._check(newurl)
        except FetchRefused as exc:
            fp.close()
            raise FetchRefused(f"redirect from {req.full_url} to {newurl}: {exc}") from None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_opener(check: Check, *handlers: urllib.request.BaseHandler) -> urllib.request.OpenerDirector:
    """A urllib opener whose redirects must pass `check`; `handlers` replace urllib's defaults (tests
    pass a fake transport)."""
    return urllib.request.build_opener(AllowlistRedirectHandler(check), *handlers)


def require_allowed(source: Source, url: str) -> None:
    if not source.allows(url):
        raise FetchRefused(f"{url} is outside the allowed prefixes of {source.id}")


@dataclass
class _Tally:
    """What one fetch_source run did, collected as it goes."""

    source_id: str
    fetched: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    local: list[str] = field(default_factory=list)
    discovered: list[str] = field(default_factory=list)
    refused: list[Issue] = field(default_factory=list)
    failed: list[Issue] = field(default_factory=list)

    def full(self, limit: int | None) -> bool:
        return limit is not None and len(self.fetched) >= limit

    def report(self) -> FetchReport:
        return FetchReport(
            source_id=self.source_id, fetched=tuple(self.fetched), skipped=tuple(self.skipped), local=tuple(self.local),
            discovered=tuple(self.discovered), refused=tuple(self.refused), failed=tuple(self.failed),
        )  # fmt: skip


class Fetcher:
    """Fetches sources into a BuildStore. Raw files go next to the work database (RAW_DIR for the
    default one) unless raw_dir is given; local files are read from local_dir/<source id>/. Without
    an `opener`, it builds a urllib opener per source whose redirects are checked like requests;
    `handlers` go into that opener (tests pass a fake transport)."""

    def __init__(
        self,
        store: BuildStore,
        *,
        opener: Opener | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        user_agent: str = UA,
        raw_dir: Path | None = None,
        local_dir: Path | None = None,
        handlers: Sequence[urllib.request.BaseHandler] = (),
    ) -> None:
        self._store = store
        self._opener = opener
        self._handlers = tuple(handlers)
        self._clock, self._sleep, self._user_agent = clock, sleep, user_agent
        self._raw_dir = Path(raw_dir) if raw_dir else store.path.parent / RAW_DIR.name
        self._local_dir = Path(local_dir) if local_dir else store.path.parent / LOCAL_DIR.name
        self._robots: dict[str, RobotFileParser | str] = {}  # site -> its rules, or why it is skipped
        self._last: dict[str, float] = {}  # host -> when its last request ended

    def fetch_source(self, source: Source, *, limit: int | None = None) -> FetchReport:
        """Ingest the local files, fetch the seeds, then the sitemap pages; at most `limit` new downloads."""
        opener = self._opener_for(source)
        tally = _Tally(source.id)
        self._ingest_local(source, tally)
        self._fetch_seeds(source, opener, source.seeds, tally, limit)
        if source.sitemap is not None and not tally.full(limit):
            seeded = {seed.url for seed in source.seeds}
            pages = [Seed(url=url) for url in self._discover(source, opener, tally) if url not in seeded]
            self._fetch_seeds(source, opener, pages, tally, limit)
        return tally.report()

    def get(self, source: Source, url: str, *, opener: Opener | None = None) -> Response:
        """One polite GET of an allowed URL; raises FetchRefused or FetchFailed."""
        require_allowed(source, url)
        opener = opener or self._opener_for(source)
        rules = self._permit(source, url)
        response = self._request(opener, url, self._interval(source, rules))
        if not source.allows(response.url):
            raise FetchRefused(f"{url} redirected to {response.url}, outside the allowed prefixes")
        return response

    def _opener_for(self, source: Source) -> Opener:
        """The injected opener, or one whose every redirect is checked like a new request."""
        if self._opener is not None:
            return self._opener
        return build_opener(functools.partial(self._follow, source), *self._handlers)

    def _robots_opener(self, source: Source) -> Opener:
        """robots.txt is always fetchable: its redirects need only stay on the allowlist (RFC 9309)."""
        if self._opener is not None:
            return self._opener
        return build_opener(functools.partial(require_allowed, source), *self._handlers)

    def _permit(self, source: Source, url: str) -> RobotFileParser:
        """The robots.txt rules of an allowed URL's host, when they allow it; else FetchRefused."""
        require_allowed(source, url)
        rules = self._rules(source, url)
        if not rules.can_fetch(self._user_agent, url):
            raise FetchRefused(f"robots.txt disallows {url}")
        return rules

    def _follow(self, source: Source, url: str) -> None:
        """A redirect target is checked like a new request: the allowlist and its host's robots.txt,
        then it waits for that host's turn."""
        rules = self._permit(source, url)
        host = urlsplit(url).netloc.lower()
        self._wait_turn(host, self._interval(source, rules))
        self._last[host] = self._clock()

    def _interval(self, source: Source, rules: RobotFileParser) -> float:
        return max(source.min_interval_s, float(rules.crawl_delay(self._user_agent) or 0))

    # --- documents --------------------------------------------------------------------------------

    def _fetch_seeds(self, source: Source, opener: Opener, seeds: Sequence[Seed], tally: _Tally, limit: int | None) -> None:
        for seed in seeds:
            if tally.full(limit):
                return
            if self._store.has_raw(seed.url):
                tally.skipped.append(seed.url)
                continue
            try:
                response = self.get(source, seed.url, opener=opener)
                self._save(source, seed, response.data, document_type(response.data, response.content_type, response.url))
            except FetchRefused as exc:
                tally.refused.append(Issue(url=seed.url, reason=str(exc)))
            except FetchFailed as exc:
                tally.failed.append(Issue(url=seed.url, reason=str(exc)))
            else:
                log.info("%s: fetched %s", source.id, seed.url)
                tally.fetched.append(seed.url)

    def _ingest_local(self, source: Source, tally: _Tally) -> None:
        folder = self._local_dir / source.id
        for local in source.local_files:
            if self._store.has_raw(local.url):
                tally.skipped.append(local.url)
                continue
            try:
                data = _read_local(folder, local)
                self._save(source, local, data, document_type(data, _LOCAL_TYPES[Path(local.path).suffix.lower()], local.path))
            except FetchRefused as exc:
                tally.refused.append(Issue(url=local.url, reason=str(exc)))
            except OSError as exc:
                tally.failed.append(Issue(url=local.url, reason=f"cannot read local file {folder / local.path}: {exc.strerror or exc}"))
            else:
                tally.local.append(local.url)

    def _save(self, source: Source, seed: Seed, data: bytes, content_type: str) -> RawItem:
        sha = hashlib.sha256(data).hexdigest()
        path = self._raw_dir / source.id / f"{sha}.{EXTENSIONS[content_type]}"
        _write_once(path, data)
        item = RawItem(
            url=seed.url, source_id=source.id, sha256=sha, content_type=content_type, path=path, fetched_at=_utc_now(),
            title=seed.title, symbol=seed.symbol, state=seed.state, year=seed.year,
        )  # fmt: skip
        self._store.record_raw(item)
        return item

    # --- sitemap ----------------------------------------------------------------------------------

    def _discover(self, source: Source, opener: Opener, tally: _Tally) -> tuple[str, ...]:
        """Pages listed in the sitemap (and in nested sitemaps on the allowlist) that match url_pattern."""
        queue, seen, pages = [source.sitemap], set(), []
        while queue and len(seen) < MAX_SITEMAPS:
            url = queue.pop(0)
            if url is None or url in seen:
                continue
            seen.add(url)
            try:
                kind, locations = parse_sitemap(self.get(source, url, opener=opener).data)
            except FetchRefused as exc:
                tally.refused.append(Issue(url=url, reason=str(exc)))
                continue
            except FetchFailed as exc:
                tally.failed.append(Issue(url=url, reason=str(exc)))
                continue
            if kind == "sitemapindex":
                queue.extend(location for location in locations if source.allows(location))
            else:
                pages.extend(location for location in locations if source.wants(location))
        found = tuple(dict.fromkeys(pages))
        tally.discovered.extend(found)
        return found

    # --- robots.txt, rate limit, retries ----------------------------------------------------------

    def _rules(self, source: Source, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        site = f"{parts.scheme}://{parts.netloc}"
        if site not in self._robots:
            self._robots[site] = self._read_robots(source, self._robots_opener(source), site)
        rules = self._robots[site]
        if isinstance(rules, str):
            raise FetchRefused(rules)
        return rules

    def _read_robots(self, source: Source, opener: Opener, site: str) -> RobotFileParser | str:
        rules = RobotFileParser()  # parsed from what we fetched: .read() would use urllib's default agent
        try:
            response = self._request(opener, f"{site}/robots.txt", source.min_interval_s)
        except FetchFailed as exc:
            if exc.status is not None and 400 <= exc.status < 500 and exc.status != 429:
                rules.parse([])  # no robots.txt: everything is allowed
                return rules
            return f"robots.txt of {site} could not be read ({exc}): host skipped"
        except FetchRefused as exc:
            return f"robots.txt of {site} could not be read ({exc}): host skipped"
        rules.parse(response.data.decode("utf-8", errors="replace").splitlines())
        return rules

    def _request(self, opener: Opener, url: str, interval: float) -> Response:
        host = urlsplit(url).netloc.lower()
        for attempt in range(RETRIES + 1):
            self._wait_turn(host, interval * 2**attempt)
            try:
                return self._open(opener, url)
            except FetchFailed as exc:
                if not exc.retryable or attempt == RETRIES:
                    raise
                log.info("retrying %s after: %s", url, exc)
            finally:
                self._last[host] = self._clock()
        raise AssertionError("unreachable")

    def _wait_turn(self, host: str, interval: float) -> None:
        last = self._last.get(host)
        if last is None:
            return
        remaining = interval - (self._clock() - last)
        if remaining > 0:
            self._sleep(remaining)

    def _open(self, opener: Opener, url: str) -> Response:
        request = urllib.request.Request(url, headers={"User-Agent": self._user_agent, "Accept": _ACCEPT})
        try:
            with opener.open(request, timeout=TIMEOUT_S) as response:
                return _read_capped(response)
        except urllib.error.HTTPError as exc:
            exc.close()
            retryable = exc.code == 429 or exc.code >= 500
            raise FetchFailed(f"HTTP {exc.code} from {url}", status=exc.code, retryable=retryable) from None
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
            raise FetchFailed(f"{url}: {exc}", retryable=timed_out) from None


def fetch_all(
    sources: Sequence[Source], store: BuildStore, *, only: str | None = None, limit: int | None = None, fetcher: Fetcher | None = None
) -> list[FetchReport]:
    """Fetch every source, or only the one whose id is `only`."""
    chosen = [source for source in sources if only is None or source.id == only]
    if only is not None and not chosen:
        raise ValueError(f"unknown source {only!r}; sources.yaml has {[source.id for source in sources]}")
    fetcher = fetcher or Fetcher(store)
    return [fetcher.fetch_source(source, limit=limit) for source in chosen]


def parse_sitemap(data: bytes) -> tuple[str, tuple[str, ...]]:
    """("sitemapindex" | "urlset", the <loc> of each entry). Parsed only: nothing in it is fetched here."""
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise FetchFailed(f"sitemap is not valid XML: {exc}") from None
    kind = _local_name(root.tag)
    if kind not in ("sitemapindex", "urlset"):
        raise FetchFailed(f"not a sitemap (its root element is <{kind}>)")
    entry = "sitemap" if kind == "sitemapindex" else "url"
    locations = (
        (loc.text or "").strip()
        for element in root if _local_name(element.tag) == entry
        for loc in element if _local_name(loc.tag) == "loc"
    )  # fmt: skip
    return kind, tuple(location for location in locations if location)


def document_type(data: bytes, content_type: str, name: str) -> str:
    """PDF or HTML, from the bytes first; anything else is refused."""
    if b"%PDF-" in data[:1024]:
        return PDF
    head = data[:512].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if content_type in _HTML_TYPES or head.startswith((b"<!doctype html", b"<html")):
        return HTML
    raise FetchRefused(f"{name} is neither a PDF nor an HTML page ({content_type})")


def _read_capped(response: Any) -> Response:
    length = response.headers.get("Content-Length")
    if length and length.isdigit() and int(length) > MAX_BYTES:
        raise FetchRefused(f"{response.geturl()} is larger than {MAX_BYTES} bytes")
    data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise FetchRefused(f"{response.geturl()} is larger than {MAX_BYTES} bytes")
    return Response(url=response.geturl(), data=data, content_type=response.headers.get_content_type())


def _read_local(folder: Path, local: LocalFile) -> bytes:
    path = folder / local.path
    if not path.resolve().is_relative_to(folder.resolve()):
        raise FetchRefused(f"local file {local.path} resolves outside {folder}")
    if path.stat().st_size > MAX_BYTES:
        raise FetchRefused(f"local file {local.path} is larger than {MAX_BYTES} bytes")
    return path.read_bytes()


def _write_once(path: Path, data: bytes) -> None:
    """Content-addressed: an existing file already holds these bytes."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    os.replace(partial, path)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
