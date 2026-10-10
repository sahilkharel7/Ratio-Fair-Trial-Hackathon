"""Collections: case documents a user uploads, so a region's cases can be searched without any crawling.

A collection is a folder, data/corpus/collections/<slug>/ (git-ignored; RATIO_COLLECTIONS_DIR moves it),
holding the uploaded files and collection.yaml (its name, region and whether it is private). Each file becomes a document of the
precedent corpus and is read exactly like a crawled one: its text, its fact patterns quoted word for word
and checked, and every passage embedded for search.

A private collection is read only by the local model (OllamaLLM on 127.0.0.1): `build` refuses any other.
This module makes no network call of its own, so the app may use it.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from corpus_builder.build_db import build_db, extraction_models
from corpus_builder.embed import Embedder, embed_all, embed_passages
from corpus_builder.extract import PROMPT_SHA, OllamaLLM, extract_all
from corpus_builder.fetch import HTML, MAX_BYTES, PDF
from corpus_builder.normalize import MAX_ID_CHARS, TEXT, normalize_all, slugify
from corpus_builder.store import BuildStore, RawItem
from corpus_builder.verify import verify_all
from ratio.config import FactPatternTaxonomy
from ratio.paths import collections_dir, corpus_db_path
from ratio.precedent_schema import COLLECTION_PREFIX, CorpusMeta, PrecedentDoc

log = logging.getLogger(__name__)

MANIFEST = "collection.yaml"
SUFFIXES = {".pdf": PDF, ".txt": TEXT, ".md": TEXT, ".html": HTML, ".htm": HTML}
MAX_SLUG_CHARS = 40
MAX_STEM_CHARS = 60
SHA_CHARS = 8  # of the file's sha256, in its document id: two files never share an id
Progress = Callable[[str], None]


class CollectionError(ValueError):
    """A collection or file that cannot be added or built; the message says why."""


@dataclass(frozen=True)
class Collection:
    """A folder of uploaded documents. It also serves as the documents' source in the corpus."""

    slug: str
    name: str
    region: str = ""
    private: bool = True
    created: str = ""
    kind: str = "collection_document"

    @property
    def id(self) -> str:
        return f"collection-{self.slug}"

    @property
    def body(self) -> str:
        return self.name

    @property
    def attribution(self) -> str:
        where = "private: read only on this computer" if self.private else "public material"
        return f"{self.name} collection, uploaded to this computer ({where})"

    @property
    def terms_url(self) -> str:
        return f"{COLLECTION_PREFIX}{self.slug}"

    @property
    def terms_checked(self) -> str:
        return self.created[:10]

    @property
    def allowed_prefixes(self) -> tuple[str, ...]:
        return (f"{COLLECTION_PREFIX}{self.slug}/",)

    def allows(self, url: str) -> bool:
        return url.startswith(self.allowed_prefixes)

    def holds(self, doc: PrecedentDoc) -> bool:
        """Whether this collection still holds the document's file, unchanged (a removed or replaced
        file, or a deleted and re-created collection, leaves the corpus at the next build)."""
        if not self.allows(doc.url):
            return False
        path = collections_dir() / self.slug / "files" / doc.url.rsplit("/", 1)[1]
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == doc.raw_sha256


def collection_slug(name: str) -> str:
    slug = slugify(name)[:MAX_SLUG_CHARS].strip("-")
    if not slug:
        raise CollectionError("A collection needs a name with letters or digits, for example 'Indonesia'.")
    return slug


def _manifest(collection: Collection) -> dict:
    return {"name": collection.name, "region": collection.region, "private": collection.private, "created": collection.created}


def load_collection(slug: str, root: Path | None = None) -> Collection | None:
    root = root or collections_dir()
    path = root / slug / MANIFEST
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Collection(slug=slug, name=str(data.get("name") or slug), region=str(data.get("region") or ""),
                      private=bool(data.get("private", True)), created=str(data.get("created") or ""))  # fmt: skip


def list_collections(root: Path | None = None) -> tuple[Collection, ...]:
    root = root or collections_dir()
    if not root.is_dir():
        return ()
    found = (load_collection(folder.name, root) for folder in sorted(root.iterdir()) if folder.is_dir())
    return tuple(collection for collection in found if collection is not None)


def create_collection(name: str, *, region: str = "", private: bool = True, root: Path | None = None) -> Collection:
    """The collection with this name, created if new. Its privacy cannot change once set."""
    root = root or collections_dir()
    slug = collection_slug(name)
    existing = load_collection(slug, root)
    if existing is not None:
        if existing.private != private:
            state = "private" if existing.private else "public"
            raise CollectionError(f"'{existing.name}' already exists as a {state} collection; choose another name.")
        return existing
    if (root / slug).exists():
        raise CollectionError(f"'{slug}' has a folder but no {MANIFEST}: restore it, or remove the folder, before adding to it.")
    collection = Collection(slug=slug, name=name.strip(), region=region.strip(), private=private, created=datetime.now(UTC).isoformat(timespec="seconds"))
    (root / slug / "files").mkdir(parents=True, exist_ok=True)
    (root / slug / MANIFEST).write_text(yaml.safe_dump(_manifest(collection), sort_keys=False), encoding="utf-8")
    return collection


def files_in(collection: Collection, root: Path | None = None) -> tuple[Path, ...]:
    root = root or collections_dir()
    folder = root / collection.slug / "files"
    return tuple(sorted(path for path in folder.iterdir() if path.is_file())) if folder.is_dir() else ()


def _content_type(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUFFIXES:
        raise CollectionError(f"{filename}: only PDF, text (.txt, .md) or HTML files can be added.")
    if len(data) > MAX_BYTES:
        raise CollectionError(f"{filename}: larger than {MAX_BYTES // (1024 * 1024)} MB.")
    content_type = SUFFIXES[suffix]
    if content_type == PDF and not data.startswith(b"%PDF"):
        raise CollectionError(f"{filename}: not a PDF file.")
    return content_type


def _title(filename: str) -> str:
    return " ".join(Path(filename).stem.replace("_", " ").replace("-", " ").split()) or filename


def _symbol(collection: Collection, stem: str, sha: str) -> str:
    """<slug>-<name>-<sha>, the name shortened so the document id (col-<symbol>) keeps the sha whole. The
    sha covers the file's place and contents, so no two files share an id, in one collection or two."""
    room = MAX_ID_CHARS - len("col-") - len(collection.slug) - SHA_CHARS - 2
    name = stem[: max(room, 1)].strip("-") or "file"
    return f"{collection.slug}-{name}-{sha[:SHA_CHARS]}"


def add_file(collection: Collection, filename: str, data: bytes, store: BuildStore, *, root: Path | None = None) -> RawItem:
    """Save one uploaded file in the collection and record it for the build. Uploads are untrusted
    data: the file is only read later, never executed."""
    root = root or collections_dir()
    content_type = _content_type(filename, data)
    sha = hashlib.sha256(data).hexdigest()
    suffix = Path(filename).suffix.lower()
    stem = slugify(Path(filename).stem)[:MAX_STEM_CHARS].strip("-") or sha[:12]
    folder = root / collection.slug / "files"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}{suffix}"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        stem = f"{stem}-{sha[:8]}"  # another file with the same name: keep both
        path = folder / f"{stem}{suffix}"
    path.write_bytes(data)
    item = RawItem(
        url=f"{COLLECTION_PREFIX}{collection.slug}/{path.name}", source_id=collection.id, sha256=sha, content_type=content_type,
        path=path, fetched_at=datetime.now(UTC).isoformat(timespec="seconds"), title=_title(filename),
        symbol=_symbol(collection, stem, hashlib.sha256(f"{collection.slug}/{path.name}\n".encode() + data).hexdigest()),
    )  # fmt: skip
    store.record_raw(item)
    return item


@dataclass(frozen=True)
class BuildReport:
    documents: tuple[str, ...]  # the collections' documents now in the corpus
    skipped: tuple[str, ...]  # files that could not be read, with the reason
    extracted: int
    failed: tuple[str, ...]  # documents the model could not read, with the reason
    passages: int
    meta: CorpusMeta
    indexed: int | None  # what was loaded into OpenSearch, None when it is not running


def _say(progress: Progress | None, message: str) -> None:
    log.info(message)
    if progress is not None:
        progress(message)


def build(
    store: BuildStore,
    taxonomy: FactPatternTaxonomy,
    llm: object,
    embedder: Embedder,
    *,
    collections: Sequence[Collection],
    sources: Sequence[object],
    out_path: Path | None = None,
    index: Callable[[Path | None], int | None] | None = None,
    progress: Progress | None = None,
) -> BuildReport:
    """Read the collections' new documents into the corpus and rebuild it with every other source.

    ``sources`` are the crawled sources (sources.yaml), kept in the corpus beside the collections;
    ``index`` loads the rebuilt corpus file into OpenSearch, when it runs."""
    if any(c.private for c in collections) and not isinstance(llm, OllamaLLM):
        raise CollectionError("A private collection is read only by the local model, never sent to a cloud model.")
    _say(progress, "Reading the uploaded documents")
    normalized = normalize_all(store, collections)
    prefixes = tuple(prefix for c in collections for prefix in c.allowed_prefixes)
    skipped = tuple(f"{issue.url}: {issue.reason}" for issue in normalized.skipped if issue.url.startswith(prefixes))
    wanted = {doc.id for doc in store.documents() if any(c.holds(doc) for c in collections)}
    model = "local" if isinstance(llm, OllamaLLM) else "cloud"
    _say(progress, f"Finding fact patterns in {len(wanted)} documents with the {model} model (each quote is checked)")
    extracted = extract_all(store, taxonomy, llm, only=wanted) if wanted else None
    _say(progress, "Checking every quote against its document")
    verify_all(store, taxonomy)
    _say(progress, "Embedding the facts and passages on this computer")
    embed_all(store, embedder)
    passages = embed_passages(store, embedder)
    _say(progress, "Building the corpus")
    _refuse_dropping_the_corpus(store, out_path or corpus_db_path())
    every = {c.slug: c for c in (*list_collections(), *collections)}  # the corpus keeps the collections not rebuilt now
    meta = build_db(store, taxonomy, [*sources, *every.values()], out_path, extraction_model=extraction_models(store), prompt_sha=PROMPT_SHA)
    indexed = index(out_path) if index is not None else None
    verified, cuts = store.verified(), store.passage_cuts()
    in_corpus = tuple(sorted(pid for pid in wanted if (pid in verified and verified[pid][0]) or cuts.get(pid)))
    failed = tuple(f"{pid}: {reason}" for pid, reason in getattr(extracted, "failed", ()))
    return BuildReport(documents=in_corpus, skipped=skipped, extracted=len(getattr(extracted, "extracted", ())),
                       failed=failed, passages=passages, meta=meta, indexed=indexed)  # fmt: skip


def _refuse_dropping_the_corpus(store: BuildStore, corpus_path: Path) -> None:
    """The corpus file is rebuilt from the work database, so a computer that has the public corpus file but
    not its build.db would lose the public documents: refuse, and say what to copy."""
    if not corpus_path.is_file():
        return
    with closing(sqlite3.connect(f"file:{corpus_path}?mode=ro", uri=True)) as db:
        try:
            public = {row[0] for row in db.execute("SELECT id FROM documents WHERE kind != 'collection_document'")}
        except sqlite3.OperationalError:
            return
    missing = public - {doc.id for doc in store.documents()}
    if missing:
        raise CollectionError(
            f"This computer has the corpus file but not the work database it was built from ({store.path}); rebuilding "
            f"would drop {len(missing)} public documents. Copy data/corpus/build.db from the maintainer first."
        )


# --- for the app: everything it needs to read a collection on this computer ----------------------


def local_model() -> OllamaLLM:
    """The local model (Ollama on 127.0.0.1): the only one the app ever uses to read a collection."""
    return OllamaLLM()


def crawled_sources() -> tuple[object, ...]:
    """The sources in sources.yaml, kept in the corpus beside the collections."""
    from corpus_builder.sources import load_sources

    return tuple(load_sources())


def index_if_running(path: Path | None, url: str, alias: str, taxonomy: FactPatternTaxonomy) -> int | None:
    """Load the rebuilt corpus into OpenSearch on this computer when it runs; None when it does not."""
    from corpus_builder.opensearch import index_corpus
    from ratio.paths import corpus_db_path
    from ratio.precedent_opensearch import available

    if not available(url):
        return None
    return index_corpus(path or corpus_db_path(), url, alias, taxonomy=taxonomy)
