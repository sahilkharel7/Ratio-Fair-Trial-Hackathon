"""The public sources of the precedent corpus (sources.yaml): what may be fetched, from where, how
politely, and how each source is credited. Nothing outside a source's allowed prefixes is ever fetched.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from ratio.config import ConfigError
from ratio.paths import CORPUS_DIR
from ratio.precedent_schema import PrecedentKind
from ratio.schema import Frozen

SOURCES_FILE = Path(__file__).with_name("sources.yaml")
LOCAL_DIR = CORPUS_DIR / "local"  # documents downloaded by hand: data/corpus/local/<source id>/<path>
MIN_INTERVAL_S = 3.0  # the shortest wait between two requests to one host
# robots.txt on these disallows /doc, /api and /access: their documents are downloaded by hand (local_files).
REFUSED_HOSTS = frozenset({"documents.un.org", "docs.un.org"})
LOCAL_SUFFIXES = frozenset({".pdf", ".html", ".htm"})
_PREFIX = re.compile(r"^https://[a-z0-9.-]+(?::\d+)?/")  # scheme, lower-case host, and the start of a path
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Seed(Frozen):
    """One document to fetch, with what is known about it (all optional except the address)."""

    url: str = Field(pattern=r"^https://")
    symbol: str | None = None  # CCPR/C/107/D/1787/2008, A/HRC/WGAD/2023/1, ...
    title: str | None = None
    state: str | None = None
    year: int | None = Field(default=None, ge=1950, le=2100)


class LocalFile(Seed):
    """A document downloaded by hand into data/corpus/local/<source id>/. Its url says where it came
    from, for attribution; it is never fetched."""

    path: str

    @field_validator("path")
    @classmethod
    def _inside_its_folder(cls, path: str) -> str:
        pure = PurePosixPath(path)
        if not path.strip() or "\\" in path or pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"local file {path!r} must be a relative path inside data/corpus/local/<source id>/")
        if pure.suffix.lower() not in LOCAL_SUFFIXES:
            raise ValueError(f"local file {path!r} must be a PDF or an HTML page")
        return path


class Source(Frozen):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    kind: PrecedentKind
    name: str = Field(min_length=1)
    body: str = Field(min_length=1)  # who decided or assessed, e.g. "UN Human Rights Committee"
    attribution: str = Field(min_length=1)
    terms_url: str = Field(pattern=r"^https://")
    terms_checked: str  # ISO date the terms of use were last read
    terms_summary: str = Field(min_length=1)
    allowed_prefixes: tuple[str, ...] = Field(min_length=1)
    min_interval_s: float = Field(ge=MIN_INTERVAL_S)
    seeds: tuple[Seed, ...] = ()
    sitemap: str | None = None
    url_pattern: str | None = None  # which sitemap pages to fetch; no other link is ever followed
    local_files: tuple[LocalFile, ...] = ()

    @field_validator("terms_checked", mode="before")
    @classmethod
    def _iso_date(cls, value: object) -> object:
        if isinstance(value, dt.date):  # YAML reads an unquoted 2026-10-09 as a date
            return value.isoformat()
        if not isinstance(value, str) or not _ISO_DATE.match(value):
            raise ValueError(f"terms_checked must be an ISO date (YYYY-MM-DD), not {value!r}")
        try:
            dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"terms_checked must be an ISO date (YYYY-MM-DD), not {value!r}") from exc
        return value

    @field_validator("allowed_prefixes")
    @classmethod
    def _https_prefixes(cls, prefixes: tuple[str, ...]) -> tuple[str, ...]:
        for prefix in prefixes:
            if not _PREFIX.match(prefix):
                raise ValueError(f"allowed prefix {prefix!r} must look like https://host/ (lower case, ending at a path)")
            if urlsplit(prefix).hostname in REFUSED_HOSTS:
                raise ValueError(f"allowed prefix {prefix!r}: its robots.txt disallows fetching; use local_files")
        return prefixes

    @field_validator("url_pattern")
    @classmethod
    def _compiles(cls, pattern: str | None) -> str | None:
        if pattern is not None:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"url_pattern {pattern!r} is not a valid pattern: {exc}") from exc
        return pattern

    @model_validator(mode="after")
    def _urls_on_the_allowlist(self) -> Source:
        outside = [seed.url for seed in self.seeds if not self.allows(seed.url)]
        if self.sitemap is not None and not self.allows(self.sitemap):
            outside.append(self.sitemap)
        if outside:
            raise ValueError(f"{self.id}: outside the allowed prefixes {list(self.allowed_prefixes)}: {outside}")
        if (self.sitemap is None) != (self.url_pattern is None):
            raise ValueError(f"{self.id}: a sitemap needs a url_pattern, and only a sitemap has one")
        urls = [seed.url for seed in self.seeds]
        if len(urls) != len(set(urls)):
            raise ValueError(f"{self.id}: a seed is listed twice")
        return self

    def allows(self, url: str) -> bool:
        return url.startswith(self.allowed_prefixes)

    def wants(self, url: str) -> bool:
        """A sitemap page this source fetches: on the allowlist and matching url_pattern."""
        return self.url_pattern is not None and self.allows(url) and re.fullmatch(self.url_pattern, url) is not None


class SourceCatalogue(Frozen):
    version: int
    sources: tuple[Source, ...]

    @model_validator(mode="after")
    def _ids_unique(self) -> SourceCatalogue:
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate source id in {sorted(i for i in set(ids) if ids.count(i) > 1)}")
        return self


def load_sources(path: Path = SOURCES_FILE) -> tuple[Source, ...]:
    """Read and validate sources.yaml; raises ConfigError naming the file at fault."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read {path.name}: {exc.strerror or exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    try:
        return SourceCatalogue.model_validate(data).sources
    except ValidationError as exc:
        raise ConfigError(f"{path.name} is invalid:\n{exc}") from exc
