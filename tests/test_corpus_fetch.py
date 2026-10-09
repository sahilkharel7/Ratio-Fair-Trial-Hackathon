"""Corpus builder, fetch step: sources.yaml rules, the allowlist, redirects, robots.txt, the rate limit,
retries, storage, sitemap discovery, local files and the CLI. A fake opener serves canned replies and a
fake clock measures waits: these tests never touch the network."""

from __future__ import annotations

import email.message
import io
import sqlite3
import urllib.error
import urllib.request
import urllib.response
from dataclasses import dataclass
from pathlib import Path

import pytest

from corpus_builder import __main__ as cli
from corpus_builder import fetch
from corpus_builder.fetch import UA, AllowlistRedirectHandler, Fetcher, FetchRefused, fetch_all
from corpus_builder.sources import Source, load_sources
from corpus_builder.store import BuildStore, RawItem
from ratio.config import ConfigError

PDF_BYTES = b"%PDF-1.4\nSYNTHETIC test document\n%%EOF\n"
HTML_BYTES = b"<!doctype html><html><body><p>SYNTHETIC test page</p></body></html>"


# --- fakes -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reply:
    body: bytes = b""
    status: int = 200
    content_type: str = "text/html"
    final_url: str | None = None  # where the server's redirects ended


class FakeResponse:
    def __init__(self, url: str, reply: Reply) -> None:
        self._url = url
        self._body = io.BytesIO(reply.body)
        self.headers = email.message.Message()
        self.headers["Content-Type"] = reply.content_type

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeOpener:
    """Serves canned replies by URL (a list is served in turn) and records every request. A URL
    without a reply answers 404, like a site without that page."""

    def __init__(self, routes: dict[str, object], clock: FakeClock | None = None, cost_s: float = 0.0) -> None:
        self.routes = {url: list(reply) if isinstance(reply, list) else [reply] for url, reply in routes.items()}
        self.requests: list[urllib.request.Request] = []
        self._clock, self._cost_s = clock, cost_s

    def open(self, request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
        self.requests.append(request)
        if self._clock is not None:
            self._clock.now += self._cost_s  # the time a download takes
        url = request.full_url
        replies = self.routes.get(url, [Reply(status=404)])
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, BaseException):
            raise reply
        if reply.status >= 400:
            raise urllib.error.HTTPError(url, reply.status, "error", email.message.Message(), io.BytesIO(b""))
        return FakeResponse(reply.final_url or url, reply)

    @property
    def urls(self) -> list[str]:
        return [request.full_url for request in self.requests]


class FakeHTTPS(urllib.request.HTTPSHandler):
    """The transport under a real urllib opener: canned replies by URL, so urllib's own redirect
    handling (and the allowlist handler in it) runs. A reply with a Location header is a redirect."""

    def __init__(self, routes: dict[str, Reply | tuple[int, str]]) -> None:
        super().__init__()
        self.routes = routes
        self.urls: list[str] = []
        self.agents: list[str | None] = []

    def https_open(self, req: urllib.request.Request) -> urllib.response.addinfourl:
        self.urls.append(req.full_url)
        self.agents.append(req.get_header("User-agent"))
        reply = self.routes.get(req.full_url, Reply(status=404))
        headers = email.message.Message()
        if isinstance(reply, tuple):  # (status, location)
            status, body = reply[0], b""
            headers["Location"] = reply[1]
        else:
            status, body = reply.status, reply.body
            headers["Content-Type"] = reply.content_type
        response = urllib.response.addinfourl(io.BytesIO(body), headers, req.full_url, status)
        response.msg = "canned"
        return response


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


# --- helpers -----------------------------------------------------------------------------------


def make_source(**overrides: object) -> Source:
    data: dict[str, object] = {
        "id": "example",
        "kind": "trialwatch_report",
        "name": "Example reports (SYNTHETIC)",
        "body": "Example monitor",
        "attribution": "SYNTHETIC test source",
        "terms_url": "https://example.org/terms/",
        "terms_checked": "2026-10-09",
        "terms_summary": "Invented for tests.",
        "allowed_prefixes": ["https://example.org/"],
        "min_interval_s": 3,
        "seeds": [{"url": "https://example.org/reports/a.pdf", "title": "Report A", "state": "Quillmark", "year": 2024}],
    }
    data.update(overrides)
    return Source.model_validate(data)


@pytest.fixture
def store(tmp_path: Path) -> BuildStore:
    return BuildStore(tmp_path / "build.db")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_fetcher(store: BuildStore, opener: FakeOpener, clock: FakeClock, tmp_path: Path) -> Fetcher:
    return Fetcher(store, opener=opener, clock=clock, sleep=clock.sleep, raw_dir=tmp_path / "raw", local_dir=tmp_path / "local")


def pdf_reply(**kwargs: object) -> Reply:
    return Reply(body=PDF_BYTES, content_type="application/pdf", **kwargs)


# --- sources.yaml rules (the repository's own seeds: tests/test_corpus_sources.py) --------------


def test_a_source_with_a_sitemap_wants_only_pages_matching_its_url_pattern():
    source = make_source(sitemap="https://example.org/sitemap.xml", url_pattern=r"^https://example\.org/reports/[^/]+/$")
    assert source.wants("https://example.org/reports/some-trial/")
    assert not source.wants("https://example.org/reports/some-trial/annex/")
    assert not make_source().wants("https://example.org/reports/some-trial/")  # no sitemap, no discovered pages


@pytest.mark.parametrize("host", ["documents.un.org", "docs.un.org"])
def test_un_document_hosts_are_refused_as_prefixes(host: str):
    with pytest.raises(ValueError, match="robots.txt"):
        make_source(allowed_prefixes=[f"https://{host}/"], seeds=[])


@pytest.mark.parametrize("prefix", ["http://example.org/", "https://example.org", "https://Example.org/"])
def test_a_prefix_must_be_https_and_end_at_a_path(prefix: str):
    with pytest.raises(ValueError, match="prefix"):
        make_source(allowed_prefixes=[prefix], seeds=[])


def test_a_seed_off_the_allowlist_is_refused():
    with pytest.raises(ValueError, match="outside the allowed prefixes"):
        make_source(seeds=[{"url": "https://example.org.evil.test/a.pdf"}])


def test_a_sitemap_off_the_allowlist_is_refused():
    with pytest.raises(ValueError, match="outside the allowed prefixes"):
        make_source(sitemap="https://elsewhere.test/sitemap.xml", url_pattern="^https://example\\.org/r/$")


def test_a_sitemap_needs_a_valid_url_pattern():
    with pytest.raises(ValueError, match="url_pattern"):
        make_source(sitemap="https://example.org/sitemap.xml")
    with pytest.raises(ValueError, match="not a valid pattern"):
        make_source(sitemap="https://example.org/sitemap.xml", url_pattern="(")


def test_the_minimum_interval_is_three_seconds():
    with pytest.raises(ValueError):
        make_source(min_interval_s=1)


@pytest.mark.parametrize("value", ["9 October 2026", "2026-13-01", "20261009", ""])
def test_terms_checked_must_be_an_iso_date(value: str):
    with pytest.raises(ValueError, match="ISO date"):
        make_source(terms_checked=value)


@pytest.mark.parametrize("path", ["../escape.pdf", "/etc/passwd.pdf", "notes.txt", "a\\b.pdf"])
def test_a_local_file_stays_in_its_folder_and_is_a_document(path: str):
    with pytest.raises(ValueError):
        make_source(local_files=[{"path": path, "url": "https://example.org/a.pdf"}])


def test_load_sources_reads_an_unquoted_yaml_date_and_reports_errors(tmp_path: Path):
    good = tmp_path / "sources.yaml"
    good.write_text(
        "version: 1\nsources:\n  - id: ex\n    kind: wgad_opinion\n    name: N\n    body: B\n    attribution: A\n"
        "    terms_url: https://example.org/terms/\n    terms_checked: 2026-10-09\n    terms_summary: S\n"
        "    allowed_prefixes: [https://example.org/]\n    min_interval_s: 3\n",
        encoding="utf-8",
    )
    (source,) = load_sources(good)
    assert source.terms_checked == "2026-10-09" and source.seeds == ()
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nsources:\n  - id: ex\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="bad.yaml"):
        load_sources(bad)
    with pytest.raises(ConfigError, match="cannot read"):
        load_sources(tmp_path / "missing.yaml")


def test_source_ids_are_unique(tmp_path: Path):
    path = tmp_path / "sources.yaml"
    entry = (
        "  - {id: ex, kind: ccpr_views, name: N, body: B, attribution: A, terms_url: 'https://example.org/t/',"
        " terms_checked: '2026-10-09', terms_summary: S, allowed_prefixes: ['https://example.org/'], min_interval_s: 3}\n"
    )
    path.write_text("version: 1\nsources:\n" + entry + entry, encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate source id"):
        load_sources(path)


# --- allowlist and redirects -------------------------------------------------------------------


def test_a_url_outside_the_allowed_prefixes_is_refused_without_a_request(store, clock, tmp_path):
    opener = FakeOpener({})
    fetcher = make_fetcher(store, opener, clock, tmp_path)
    with pytest.raises(FetchRefused, match="outside the allowed prefixes"):
        fetcher.get(make_source(), "https://elsewhere.test/a.pdf")
    assert opener.requests == []


def test_a_redirect_leaving_the_allowlist_is_refused_and_nothing_is_saved(store, clock, tmp_path):
    seed_url = "https://example.org/reports/a.pdf"
    opener = FakeOpener({seed_url: pdf_reply(final_url="https://elsewhere.test/a.pdf")})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source())
    assert report.fetched == () and [issue.url for issue in report.refused] == [seed_url]
    assert "redirected" in report.refused[0].reason
    assert store.raw_items() == [] and not (tmp_path / "raw").exists()


def test_the_redirect_handler_refuses_a_target_its_check_refuses():
    source = make_source()

    def check(url: str) -> None:
        if not source.allows(url):
            raise FetchRefused(f"{url} is outside the allowed prefixes")

    handler = AllowlistRedirectHandler(check)
    request = urllib.request.Request("https://example.org/a")
    headers = email.message.Message()
    with pytest.raises(FetchRefused, match="redirect from https://example.org/a to https://elsewhere.test/a"):
        handler.redirect_request(request, io.BytesIO(b""), 302, "Found", headers, "https://elsewhere.test/a")
    with pytest.raises(FetchRefused):
        handler.redirect_request(request, io.BytesIO(b""), 301, "Moved", headers, "http://example.org/a")
    followed = handler.redirect_request(request, io.BytesIO(b""), 302, "Found", headers, "https://example.org/b")
    assert followed.full_url == "https://example.org/b"


def test_the_default_opener_uses_the_allowlist_redirect_handler():
    opener = fetch.build_opener(lambda url: None)
    handlers = [h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler)]
    assert len(handlers) == 1 and isinstance(handlers[0], AllowlistRedirectHandler)


def real_opener_fetcher(store: BuildStore, transport: FakeHTTPS, clock: FakeClock, tmp_path: Path) -> Fetcher:
    """A Fetcher that builds its own urllib opener (with the redirect handler) over a fake transport."""
    return Fetcher(store, handlers=(transport,), clock=clock, sleep=clock.sleep, raw_dir=tmp_path / "raw", local_dir=tmp_path / "local")


ROBOTS_PRIVATE = Reply(body=b"User-agent: *\nDisallow: /private/\n", content_type="text/plain")


def test_a_redirect_to_a_path_robots_txt_disallows_is_refused(store, clock, tmp_path):
    transport = FakeHTTPS({
        "https://example.org/robots.txt": ROBOTS_PRIVATE,
        "https://example.org/reports/a.pdf": (302, "https://example.org/private/a.pdf"),
        "https://example.org/private/a.pdf": pdf_reply(),
    })
    report = real_opener_fetcher(store, transport, clock, tmp_path).fetch_source(make_source())
    assert report.fetched == () and [issue.url for issue in report.refused] == ["https://example.org/reports/a.pdf"]
    assert "robots.txt disallows https://example.org/private/a.pdf" in report.refused[0].reason
    assert "redirect from https://example.org/reports/a.pdf" in report.refused[0].reason
    assert "https://example.org/private/a.pdf" not in transport.urls
    assert store.raw_items() == []


def test_a_redirect_to_another_allowed_host_obeys_that_hosts_robots_txt(store, clock, tmp_path):
    source = make_source(allowed_prefixes=["https://example.org/", "https://cdn.example.org/"])
    transport = FakeHTTPS({
        "https://example.org/reports/a.pdf": (302, "https://cdn.example.org/private/a.pdf"),
        "https://cdn.example.org/robots.txt": ROBOTS_PRIVATE,
    })
    report = real_opener_fetcher(store, transport, clock, tmp_path).fetch_source(source)
    assert report.fetched == () and "robots.txt disallows" in report.refused[0].reason
    assert transport.urls == ["https://example.org/robots.txt", "https://example.org/reports/a.pdf", "https://cdn.example.org/robots.txt"]
    assert all(agent == UA for agent in transport.agents)


def test_a_redirect_to_an_allowed_host_without_robots_txt_is_followed_politely(store, clock, tmp_path):
    source = make_source(allowed_prefixes=["https://example.org/", "https://cdn.example.org/"], min_interval_s=10)
    transport = FakeHTTPS({
        "https://example.org/reports/a.pdf": (302, "https://cdn.example.org/files/a.pdf"),
        "https://cdn.example.org/files/a.pdf": pdf_reply(),
    })
    report = real_opener_fetcher(store, transport, clock, tmp_path).fetch_source(source)
    assert report.fetched == ("https://example.org/reports/a.pdf",)
    assert transport.urls[-2:] == ["https://cdn.example.org/robots.txt", "https://cdn.example.org/files/a.pdf"]
    assert clock.sleeps == [10, 10]  # after example.org's robots.txt, and between cdn's robots.txt and the file
    (item,) = store.raw_items()
    assert item.url == "https://example.org/reports/a.pdf" and item.path.read_bytes() == PDF_BYTES


def test_a_redirect_off_the_allowlist_is_refused_by_the_real_opener(store, clock, tmp_path):
    transport = FakeHTTPS({"https://example.org/reports/a.pdf": (302, "https://elsewhere.test/a.pdf")})
    report = real_opener_fetcher(store, transport, clock, tmp_path).fetch_source(make_source())
    assert report.fetched == () and "outside the allowed prefixes" in report.refused[0].reason
    assert not any(url.startswith("https://elsewhere.test/") for url in transport.urls)


def test_a_redirected_robots_txt_is_followed_on_the_allowlist(store, clock, tmp_path):
    transport = FakeHTTPS({
        "https://example.org/robots.txt": (301, "https://example.org/static/robots.txt"),
        "https://example.org/static/robots.txt": Reply(body=b"User-agent: *\nDisallow: /reports/\n", content_type="text/plain"),
    })
    report = real_opener_fetcher(store, transport, clock, tmp_path).fetch_source(make_source())
    assert report.fetched == () and "robots.txt disallows https://example.org/reports/a.pdf" in report.refused[0].reason
    assert "https://example.org/reports/a.pdf" not in transport.urls


# --- robots.txt ---------------------------------------------------------------------------------


def test_robots_txt_is_fetched_once_per_host_with_our_user_agent(store, clock, tmp_path):
    source = make_source(seeds=[{"url": "https://example.org/a.pdf"}, {"url": "https://example.org/b.pdf"}])
    opener = FakeOpener({
        "https://example.org/robots.txt": Reply(body=b"User-agent: *\nAllow: /\n", content_type="text/plain"),
        "https://example.org/a.pdf": pdf_reply(),
        "https://example.org/b.pdf": pdf_reply(),
    })
    make_fetcher(store, opener, clock, tmp_path).fetch_source(source)
    assert opener.urls == ["https://example.org/robots.txt", "https://example.org/a.pdf", "https://example.org/b.pdf"]
    assert all(request.get_header("User-agent") == UA for request in opener.requests)


def test_a_path_robots_txt_disallows_is_refused(store, clock, tmp_path):
    source = make_source(seeds=[{"url": "https://example.org/private/a.pdf"}, {"url": "https://example.org/public/b.pdf"}])
    opener = FakeOpener({
        "https://example.org/robots.txt": Reply(body=b"User-agent: *\nDisallow: /private/\n", content_type="text/plain"),
        "https://example.org/public/b.pdf": pdf_reply(),
    })
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(source)
    assert report.fetched == ("https://example.org/public/b.pdf",)
    assert [issue.url for issue in report.refused] == ["https://example.org/private/a.pdf"]
    assert "robots.txt" in report.refused[0].reason
    assert "https://example.org/private/a.pdf" not in opener.urls


def test_robots_rules_for_our_user_agent_apply(store, clock, tmp_path):
    robots = b"User-agent: RatioCorpusBuilder\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
    opener = FakeOpener({"https://example.org/robots.txt": Reply(body=robots, content_type="text/plain")})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source())
    assert report.fetched == () and len(report.refused) == 1
    assert opener.urls == ["https://example.org/robots.txt"]


def test_a_missing_robots_txt_allows_everything(store, clock, tmp_path):
    opener = FakeOpener({"https://example.org/robots.txt": Reply(status=404), "https://example.org/reports/a.pdf": pdf_reply()})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source())
    assert report.fetched == ("https://example.org/reports/a.pdf",)


@pytest.mark.parametrize("failure", [Reply(status=500), urllib.error.URLError(ConnectionRefusedError("refused"))])
def test_a_host_whose_robots_txt_cannot_be_read_is_skipped(store, clock, tmp_path, failure):
    opener = FakeOpener({"https://example.org/robots.txt": failure, "https://example.org/reports/a.pdf": pdf_reply()})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source())
    assert report.fetched == () and "robots.txt" in report.refused[0].reason
    assert set(opener.urls) == {"https://example.org/robots.txt"}


# --- rate limit and retries ---------------------------------------------------------------------


def two_seed_source(**overrides: object) -> Source:
    return make_source(seeds=[{"url": "https://example.org/a.pdf"}, {"url": "https://example.org/b.pdf"}], **overrides)


def test_requests_to_a_host_are_spaced_by_the_minimum_interval(store, clock, tmp_path):
    opener = FakeOpener({"https://example.org/a.pdf": pdf_reply(), "https://example.org/b.pdf": pdf_reply()})
    make_fetcher(store, opener, clock, tmp_path).fetch_source(two_seed_source(min_interval_s=10))
    assert len(opener.requests) == 3  # robots.txt, a, b
    assert clock.sleeps == [10, 10]


def test_a_longer_crawl_delay_in_robots_txt_wins(store, clock, tmp_path):
    opener = FakeOpener({
        "https://example.org/robots.txt": Reply(body=b"User-agent: *\nCrawl-delay: 15\n", content_type="text/plain"),
        "https://example.org/a.pdf": pdf_reply(),
        "https://example.org/b.pdf": pdf_reply(),
    })
    make_fetcher(store, opener, clock, tmp_path).fetch_source(two_seed_source(min_interval_s=3))
    assert clock.sleeps == [15, 15]


def test_time_spent_downloading_counts_toward_the_interval(store, clock, tmp_path):
    opener = FakeOpener({"https://example.org/a.pdf": pdf_reply(), "https://example.org/b.pdf": pdf_reply()}, clock=clock, cost_s=2)
    make_fetcher(store, opener, clock, tmp_path).fetch_source(two_seed_source(min_interval_s=5))
    assert clock.sleeps == [5, 5]  # measured from the end of the previous request


def test_server_errors_are_retried_with_exponential_backoff(store, clock, tmp_path):
    opener = FakeOpener({"https://example.org/reports/a.pdf": [Reply(status=503), Reply(status=429), pdf_reply()]})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source(min_interval_s=3))
    assert report.fetched == ("https://example.org/reports/a.pdf",)
    assert clock.sleeps == [3, 6, 12]  # after robots.txt, then twice the wait after each failure


def test_retries_stop_after_three(store, clock, tmp_path):
    opener = FakeOpener({"https://example.org/reports/a.pdf": Reply(status=502)})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source(min_interval_s=3))
    assert opener.urls.count("https://example.org/reports/a.pdf") == 4  # the first try and three retries
    assert report.failed[0].reason.startswith("HTTP 502")


def test_timeouts_are_retried(store, clock, tmp_path):
    url = "https://example.org/reports/a.pdf"
    opener = FakeOpener({url: [urllib.error.URLError(TimeoutError("timed out")), TimeoutError("read timed out"), pdf_reply()]})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source())
    assert report.fetched == (url,) and opener.urls.count(url) == 3


def test_a_client_error_is_not_retried(store, clock, tmp_path):
    opener = FakeOpener({})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source())
    assert opener.urls.count("https://example.org/reports/a.pdf") == 1
    assert report.failed[0].reason.startswith("HTTP 404")


# --- storage -------------------------------------------------------------------------------------


def test_a_url_already_fetched_is_skipped_without_a_request(store, clock, tmp_path):
    url = "https://example.org/reports/a.pdf"
    store.record_raw(RawItem(url, "example", "0" * 64, fetch.PDF, tmp_path / "x.pdf", "2026-10-09T00:00:00+00:00"))
    opener = FakeOpener({})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source())
    assert report.skipped == (url,) and report.fetched == () and opener.requests == []


def test_a_download_is_saved_by_content_hash_with_the_seed_metadata(store, clock, tmp_path):
    url = "https://example.org/reports/a.pdf"
    opener = FakeOpener({url: pdf_reply()})
    make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source())
    (item,) = store.raw_items()
    assert item.path.parent == tmp_path / "raw" / "example" and item.path.name == f"{item.sha256}.pdf"
    assert item.path.read_bytes() == PDF_BYTES and item.content_type == fetch.PDF
    assert (item.url, item.title, item.state, item.year, item.symbol) == (url, "Report A", "Quillmark", 2024, None)


def test_an_html_page_is_saved_as_html(store, clock, tmp_path):
    url = "https://example.org/reports/a/"
    opener = FakeOpener({url: Reply(body=HTML_BYTES, content_type="text/html; charset=utf-8")})
    make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source(seeds=[{"url": url}]))
    (item,) = store.raw_items()
    assert item.content_type == fetch.HTML and item.path.suffix == ".html"


def test_a_download_larger_than_the_cap_is_refused(store, clock, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "MAX_BYTES", 10)
    opener = FakeOpener({"https://example.org/reports/a.pdf": pdf_reply()})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source())
    assert report.fetched == () and "larger than" in report.refused[0].reason
    assert store.raw_items() == []


def test_content_that_is_neither_pdf_nor_html_is_refused(store, clock, tmp_path):
    opener = FakeOpener({"https://example.org/reports/a.pdf": Reply(body=b"MZ\x90\x00", content_type="application/octet-stream")})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(make_source())
    assert report.fetched == () and "neither a PDF nor an HTML page" in report.refused[0].reason


def test_limit_caps_new_downloads(store, clock, tmp_path):
    opener = FakeOpener({"https://example.org/a.pdf": pdf_reply(), "https://example.org/b.pdf": pdf_reply()})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(two_seed_source(), limit=1)
    assert report.fetched == ("https://example.org/a.pdf",)
    assert "https://example.org/b.pdf" not in opener.urls


# --- sitemap discovery ---------------------------------------------------------------------------

SITEMAP_INDEX = b"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.org/report-sitemap.xml</loc></sitemap>
  <sitemap><loc>https://elsewhere.test/sitemap.xml</loc></sitemap>
</sitemapindex>"""

REPORT_SITEMAP = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.org/reports/first-trial/</loc></url>
  <url><loc>https://example.org/reports/second-trial/</loc></url>
  <url><loc>https://example.org/reports/second-trial/annex/</loc></url>
  <url><loc>https://example.org/news/a-story/</loc></url>
  <url><loc>https://elsewhere.test/reports/third-trial/</loc></url>
</urlset>"""

LINKING_PAGE = b"<!doctype html><html><body><p>SYNTHETIC</p><a href='https://example.org/reports/linked/'>x</a></body></html>"


def sitemap_source() -> Source:
    return make_source(
        seeds=[],
        sitemap="https://example.org/sitemap_index.xml",
        url_pattern=r"^https://example\.org/reports/[^/]+/$",
    )


def test_sitemap_discovery_keeps_only_pages_matching_the_url_pattern(store, clock, tmp_path):
    xml = "application/xml"
    opener = FakeOpener({
        "https://example.org/sitemap_index.xml": Reply(body=SITEMAP_INDEX, content_type=xml),
        "https://example.org/report-sitemap.xml": Reply(body=REPORT_SITEMAP, content_type=xml),
        "https://example.org/reports/first-trial/": Reply(body=LINKING_PAGE),
        "https://example.org/reports/second-trial/": Reply(body=HTML_BYTES),
    })
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(sitemap_source())
    pages = ("https://example.org/reports/first-trial/", "https://example.org/reports/second-trial/")
    assert report.discovered == pages and report.fetched == pages
    assert "https://elsewhere.test/sitemap.xml" not in opener.urls  # a sitemap off the allowlist is not followed
    assert "https://example.org/reports/linked/" not in opener.urls  # links in pages are never followed
    assert "https://example.org/news/a-story/" not in opener.urls


def test_sitemap_pages_already_fetched_are_skipped(store, clock, tmp_path):
    url = "https://example.org/reports/first-trial/"
    store.record_raw(RawItem(url, "example", "0" * 64, fetch.HTML, tmp_path / "x.html", "2026-10-09T00:00:00+00:00"))
    sitemap = b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>' + url.encode() + b"</loc></url></urlset>"
    opener = FakeOpener({"https://example.org/sitemap_index.xml": Reply(body=sitemap, content_type="application/xml")})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(sitemap_source())
    assert report.skipped == (url,) and url not in opener.urls


def test_a_broken_sitemap_is_reported(store, clock, tmp_path):
    opener = FakeOpener({"https://example.org/sitemap_index.xml": Reply(body=b"<not xml", content_type="application/xml")})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(sitemap_source())
    assert "not valid XML" in report.failed[0].reason


# --- local files ---------------------------------------------------------------------------------


def test_local_files_are_ingested_without_fetching(store, clock, tmp_path):
    folder = tmp_path / "local" / "example"
    folder.mkdir(parents=True)
    (folder / "views.pdf").write_bytes(PDF_BYTES)
    source = make_source(
        seeds=[],
        local_files=[{"path": "views.pdf", "url": "https://documents.example.test/views.pdf", "symbol": "CCPR/C/1/D/2/2020"}],
    )
    opener = FakeOpener({})
    report = make_fetcher(store, opener, clock, tmp_path).fetch_source(source)
    assert report.local == ("https://documents.example.test/views.pdf",) and opener.requests == []
    (item,) = store.raw_items()
    assert item.symbol == "CCPR/C/1/D/2/2020" and item.path.read_bytes() == PDF_BYTES


def test_a_missing_local_file_is_reported(store, clock, tmp_path):
    source = make_source(seeds=[], local_files=[{"path": "absent.pdf", "url": "https://example.org/absent.pdf"}])
    report = make_fetcher(store, FakeOpener({}), clock, tmp_path).fetch_source(source)
    assert report.local == () and "absent.pdf" in report.failed[0].reason


# --- fetch_all and the CLI -----------------------------------------------------------------------


def test_fetch_all_runs_only_the_chosen_source(store, clock, tmp_path):
    other = make_source(id="other", seeds=[{"url": "https://example.org/other.pdf"}])
    opener = FakeOpener({"https://example.org/reports/a.pdf": pdf_reply()})
    fetcher = make_fetcher(store, opener, clock, tmp_path)
    reports = fetch_all([make_source(), other], store, only="example", fetcher=fetcher)
    assert [report.source_id for report in reports] == ["example"]
    with pytest.raises(ValueError, match="unknown source"):
        fetch_all([make_source()], store, only="nope", fetcher=fetcher)


def test_the_cli_knows_every_step():
    parser = cli.build_parser()
    for argv in (
        ["fetch", "--source", "x", "--limit", "2"], ["normalize"], ["extract", "--model", "ollama", "--only", "x"],
        ["verify"], ["embed"], ["build-db", "--out", "x.db"], ["index-opensearch", "--url", "http://127.0.0.1:9200"],
        ["status"], ["snapshot"], ["restore"],
    ):
        assert callable(parser.parse_args(argv).handler)


def test_the_cli_status_counts_each_table_and_source(tmp_path, capsys):
    work_db = tmp_path / "build.db"
    BuildStore(work_db).record_raw(
        RawItem("https://cfj.org/reports/x/", "trialwatch_reports", "0" * 64, fetch.HTML, tmp_path / "x.html", "2026-10-09T00:00:00+00:00")
    )
    assert cli.main(["--work-db", str(work_db), "status"]) == 0
    out = capsys.readouterr().out
    assert "raw: 1" in out and "documents: 0" in out
    seeds = len(next(source for source in load_sources() if source.id == "trialwatch_reports").seeds)
    assert f"trialwatch_reports: {seeds} seeds, 0 local files, 1 fetched, 0 documents" in out


def test_the_cli_fetch_passes_the_source_and_limit(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_fetch_all(sources, store, *, only=None, limit=None):
        calls.append((only, limit, store.path))
        return [fetch.FetchReport(source_id=only, fetched=("https://cfj.org/reports/x/",))]

    monkeypatch.setattr(cli, "fetch_all", fake_fetch_all)
    work_db = tmp_path / "build.db"
    assert cli.main(["--work-db", str(work_db), "fetch", "--source", "trialwatch_reports", "--limit", "2"]) == 0
    assert calls == [("trialwatch_reports", 2, work_db)]
    assert "fetched: 1" in capsys.readouterr().out


def test_the_cli_refuses_an_unknown_source(tmp_path, capsys):
    assert cli.main(["--work-db", str(tmp_path / "build.db"), "fetch", "--source", "nope"]) == 2
    assert "unknown source" in capsys.readouterr().err


def test_the_cli_refuses_an_opensearch_url_off_this_computer(capsys):
    assert cli.main(["index-opensearch", "--url", "http://search.example.test:9200"]) == 2
    assert "127.0.0.1" in capsys.readouterr().err


def test_the_work_database_tables_the_status_reads_exist(tmp_path):
    BuildStore(tmp_path / "build.db")
    with sqlite3.connect(tmp_path / "build.db") as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert set(cli.WORK_TABLES) <= tables
