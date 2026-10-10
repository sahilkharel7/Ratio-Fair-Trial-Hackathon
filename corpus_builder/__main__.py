"""python -m corpus_builder <step>: build the precedent corpus for "Similar cases" (maintainer only, online).

The steps, in order. Each can be re-run: finished work is skipped or cached.

  fetch             download the seeds and sitemap pages in sources.yaml (robots.txt and rate limits obeyed)
  normalize         turn each download into a document's text
  extract           ask the model for fact patterns, quoting the public text (GEMINI_API_KEY, or --model ollama)
  verify            keep only the quotes found verbatim in the text
  embed             encode each verified fact quote's sentence, and every passage (for search), with the local MiniLM model
  build-db          write data/corpus/precedents.db, the file the app reads
  index-opensearch  load it into the optional OpenSearch index on 127.0.0.1
  snapshot/restore  save or install that index without crawling or a key
  collection        add, list or build documents you upload (no crawling; a private one is read only by the local model)
  status            what each step has done so far
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path

from pydantic import BaseModel, ValidationError

from corpus_builder.extract import CloudModelRefused, GeminiSetupError
from corpus_builder.fetch import fetch_all
from corpus_builder.normalize import normalize_all
from corpus_builder.sources import load_sources
from corpus_builder.store import BuildStore
from ratio.config import ConfigError, FactPatternTaxonomy, PrecedentSettings, default_config
from ratio.paths import build_db_path, corpus_db_path

WORK_TABLES = ("raw", "documents", "extractions", "verified", "vectors")
SNAPSHOT_NAME = "precedents"
_REPORT_ITEMS_SHOWN = 20  # per list in a printed report; the rest are counted


class CliError(Exception):
    """A usage problem, reported in one line."""


# --- steps --------------------------------------------------------------------------------------


def _fetch(args: argparse.Namespace) -> int:
    sources = load_sources()
    if args.source is not None and args.source not in {source.id for source in sources}:
        raise CliError(f"unknown source {args.source!r}; sources.yaml has {[source.id for source in sources]}")
    for report in fetch_all(sources, _store(args), only=args.source, limit=args.limit):
        _print_report(report)
    return 0


def _all_sources() -> list:
    """The crawled sources (sources.yaml) and the uploaded collections."""
    from corpus_builder.collections import list_collections

    return [*load_sources(), *list_collections()]


def _normalize(args: argparse.Namespace) -> int:
    _print_report(normalize_all(_store(args), _all_sources()))
    return 0


def _extract(args: argparse.Namespace) -> int:
    from corpus_builder import extract

    if args.model == "gemini":
        if not args.model_id:
            raise CliError("pass --model-id with the Gemini model to use (GeminiLLM.list_models() lists them)")
        llm = extract.GeminiLLM(args.model_id)
    else:
        llm = extract.OllamaLLM(model=args.model_id) if args.model_id else extract.OllamaLLM()
    report = extract.extract_all(_store(args), _taxonomy(), llm, only=args.only)
    _print_report(report)
    if report.private_refused:
        print(f"left out {len(report.private_refused)} private collection documents: {extract.PRIVATE_REFUSED} (--model ollama)")
        if args.only:
            raise CliError(f"{', '.join(report.private_refused)}: {extract.PRIVATE_REFUSED}; use --model ollama")
    return 0


def _verify(args: argparse.Namespace) -> int:
    from corpus_builder import verify

    _print_report(verify.verify_all(_store(args), _taxonomy()))
    return 0


def _embed(args: argparse.Namespace) -> int:
    from corpus_builder import embed
    from ratio.embeddings import MiniLMEmbedder

    embedder = MiniLMEmbedder()
    print(f"embedded {embed.embed_all(_store(args), embedder)} fact quotes")
    print(f"embedded {embed.embed_passages(_store(args), embedder)} new passages (for search)")
    return 0


def _build_db(args: argparse.Namespace) -> int:
    from corpus_builder import build_db, extract

    store = _store(args)
    out = Path(args.out) if args.out else corpus_db_path()
    meta = build_db.build_db(
        store, _taxonomy(), _all_sources(), out, extraction_model=_extraction_models(store.path), prompt_sha=extract.PROMPT_SHA
    )
    print(f"wrote {out}")
    _print_report(meta)
    return 0


def _index_opensearch(args: argparse.Namespace) -> int:
    settings = _opensearch_settings(args.url)
    from corpus_builder import opensearch

    count = opensearch.index_corpus(corpus_db_path(), settings.opensearch_url, settings.opensearch_index)
    print(f"indexed {count} fact quotes and passages into {settings.opensearch_index} at {settings.opensearch_url}")
    return 0


def _snapshot(args: argparse.Namespace) -> int:
    settings = _opensearch_settings(args.url)
    from corpus_builder import opensearch

    try:
        opensearch.snapshot(settings.opensearch_url, args.name, include_private=args.include_private)
    except ValueError as exc:
        raise CliError(f"{exc}; pass --include-private only if the snapshot stays on this computer") from None
    print(f"snapshot {args.name} saved")
    return 0


def _restore(args: argparse.Namespace) -> int:
    settings = _opensearch_settings(args.url)
    from corpus_builder import opensearch

    opensearch.restore(settings.opensearch_url, args.name)
    print(f"snapshot {args.name} restored")
    return 0


def _status(args: argparse.Namespace) -> int:
    store = _store(args)
    with closing(sqlite3.connect(store.path)) as db:
        counts = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in WORK_TABLES}  # noqa: S608 - fixed names
        fetched = dict(db.execute("SELECT source_id, COUNT(*) FROM raw GROUP BY source_id").fetchall())
        documents = dict(
            db.execute("SELECT r.source_id, COUNT(*) FROM documents d JOIN raw r ON r.url = d.raw_url GROUP BY r.source_id").fetchall()
        )
    print(f"work database {store.path}")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    print("sources")
    for source in load_sources():
        print(
            f"  {source.id}: {len(source.seeds)} seeds, {len(source.local_files)} local files, "
            f"{fetched.get(source.id, 0)} fetched, {documents.get(source.id, 0)} documents"
        )
    corpus = corpus_db_path()
    print(f"corpus file {corpus}: {'present' if corpus.exists() else 'not built yet'}")
    return 0


# --- helpers ------------------------------------------------------------------------------------


def _store(args: argparse.Namespace) -> BuildStore:
    return BuildStore(Path(args.work_db))


def _taxonomy() -> FactPatternTaxonomy:
    return default_config().fact_patterns


def _opensearch_settings(url: str | None) -> PrecedentSettings:
    """The configured OpenSearch, or another URL on this computer."""
    settings = default_config().settings.precedents
    if url is None:
        return settings
    try:
        return PrecedentSettings.model_validate({**settings.model_dump(), "opensearch_url": url})
    except ValidationError as exc:
        raise CliError(exc.errors()[0]["msg"]) from None


def _extraction_models(work_db: Path) -> str:
    """The model(s) whose answers the verified facets come from, for the corpus meta."""
    with closing(sqlite3.connect(work_db)) as db:
        rows = db.execute(
            "SELECT DISTINCT e.model FROM verified v JOIN extractions e ON e.cache_key = v.cache_key ORDER BY e.model"
        ).fetchall()
    return ", ".join(row[0] for row in rows) or "unknown"


def _print_report(report: object) -> None:
    """A step's report: counts for lists, and their first items."""
    if not isinstance(report, BaseModel):
        print(report)
        return
    for name, value in report:
        if not isinstance(value, (tuple, list)):
            print(f"{name}: {value}")
            continue
        print(f"{name}: {len(value)}")
        for item in value[:_REPORT_ITEMS_SHOWN]:
            print(f"  - {_one_line(item)}")
        if len(value) > _REPORT_ITEMS_SHOWN:
            print(f"  ... and {len(value) - _REPORT_ITEMS_SHOWN} more")


def _one_line(item: object) -> str:
    if isinstance(item, BaseModel):
        return ", ".join(f"{key}={value}" for key, value in item.model_dump().items())
    return str(item)


# --- collections: documents you upload, no crawling ------------------------------------------------


def _collection(args: argparse.Namespace) -> int:
    from corpus_builder import collections as col

    try:
        if args.action == "list":
            for found in col.list_collections():
                state = "private" if found.private else "public"
                print(f"{found.slug}: {found.name} ({state}, {found.region or 'no region'}), {len(col.files_in(found))} files")
            return 0
        if args.action == "add":
            return _collection_add(args, col)
        return _collection_build(args, col)
    except col.CollectionError as exc:
        raise CliError(str(exc)) from None


def _collection_add(args: argparse.Namespace, col: object) -> int:
    if not args.folder or not args.name:
        raise CliError("collection add needs a FOLDER and --name")
    folder = Path(args.folder)
    files = [path for path in sorted(folder.iterdir()) if path.is_file() and path.suffix.lower() in col.SUFFIXES] if folder.is_dir() else []
    if not files:
        raise CliError(f"no PDF, text or HTML files in {folder}")
    found = col.create_collection(args.name, region=args.region or "", private=not args.public)
    store = _store(args)
    for path in files:
        col.add_file(found, path.name, path.read_bytes(), store)
    print(f"added {len(files)} files to {found.name} ({'public' if args.public else 'private'}); next: python -m corpus_builder collection build")
    return 0


def _collection_build(args: argparse.Namespace, col: object) -> int:
    from corpus_builder import extract
    from corpus_builder.opensearch import index_corpus
    from ratio.config import default_config
    from ratio.embeddings import MiniLMEmbedder
    from ratio.precedent_opensearch import available

    chosen = [c for c in col.list_collections() if args.name is None or c.slug == col.collection_slug(args.name)]
    if not chosen:
        raise CliError("no such collection; add one with: python -m corpus_builder collection add FOLDER --name NAME")
    if args.model == "gemini" and any(c.private for c in chosen):
        raise CliError("a private collection is read only by the local model; drop --model gemini")
    if args.model == "gemini" and not args.model_id:
        raise CliError("--model gemini needs --model-id")
    llm = extract.GeminiLLM(args.model_id) if args.model == "gemini" else extract.OllamaLLM()
    settings = default_config().settings.precedents

    def index(path: Path | None) -> int | None:
        if args.no_index or not available(settings.opensearch_url):
            return None
        return index_corpus(path or corpus_db_path(), settings.opensearch_url, settings.opensearch_index, taxonomy=_taxonomy())

    report = col.build(_store(args), _taxonomy(), llm, MiniLMEmbedder(), collections=chosen, sources=load_sources(), index=index, progress=print)
    _print_report(report.meta)
    print(f"documents in the corpus from these collections: {len(report.documents)}; passages embedded: {report.passages}; "
          f"OpenSearch: {report.indexed if report.indexed is not None else 'not running (the corpus file is used)'}")  # fmt: skip
    for line in (*report.skipped, *report.failed):
        print(f"  - {line}")
    return 0


# --- parser -------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m corpus_builder", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-db", default=str(build_db_path()), help="the builder's work database (default: %(default)s)")
    steps = parser.add_subparsers(dest="step", required=True)

    def step(name: str, handler: Callable[[argparse.Namespace], int], help_text: str) -> argparse.ArgumentParser:
        sub = steps.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        return sub

    fetch = step("fetch", _fetch, "download the sources' documents")
    fetch.add_argument("--source", help="only this source id")
    fetch.add_argument("--limit", type=_positive_int, help="at most this many new downloads per source")
    step("normalize", _normalize, "turn downloads into document text")
    extract = step("extract", _extract, "extract fact patterns with quotes")
    extract.add_argument("--model", choices=("gemini", "ollama"), default="gemini")
    extract.add_argument("--model-id", help="the model id (required for gemini)")
    extract.add_argument("--only", help="only this precedent id")
    step("verify", _verify, "keep only verbatim quotes")
    step("embed", _embed, "embed verified fact quotes and every passage (for search)")
    build = step("build-db", _build_db, "write precedents.db")
    build.add_argument("--out", help="where to write it (default: data/corpus/precedents.db, or RATIO_CORPUS_DB)")
    for name, handler, help_text in (
        ("index-opensearch", _index_opensearch, "index precedents.db in OpenSearch"),
        ("snapshot", _snapshot, "snapshot the OpenSearch index"),
        ("restore", _restore, "restore the OpenSearch index from a snapshot"),
    ):
        sub = step(name, handler, help_text)
        sub.add_argument("--url", help="OpenSearch on this computer (default: settings.yaml)")
        if name != "index-opensearch":
            sub.add_argument("--name", default=SNAPSHOT_NAME, help="snapshot name (default: %(default)s)")
        if name == "snapshot":
            sub.add_argument("--include-private", action="store_true", help="also when the corpus holds private collections")
    coll = step("collection", _collection, "documents you upload (no crawling): add, list or build")
    coll.add_argument("action", choices=("add", "list", "build"))
    coll.add_argument("folder", nargs="?", help="add: the folder of PDF, text or HTML files")
    coll.add_argument("--name", help="the collection's name, e.g. Indonesia")
    coll.add_argument("--region", help="add: the region it covers")
    coll.add_argument("--public", action="store_true", help="add: public material (default: private, read only by the local model)")
    coll.add_argument("--model", choices=("ollama", "gemini"), default="ollama", help="build: gemini is refused for a private collection")
    coll.add_argument("--model-id", help="build: the Gemini model id")
    coll.add_argument("--no-index", action="store_true", help="build: do not load the corpus into OpenSearch (one index serves one corpus file)")
    step("status", _status, "what each step has done")
    return parser


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return args.handler(args)
    except (CliError, ConfigError, CloudModelRefused, GeminiSetupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
