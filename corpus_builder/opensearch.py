"""Load precedents.db into OpenSearch on this computer, and snapshot or restore that index.

`index_corpus` writes every fact quote with its vector to a new versioned index (`<alias>-v<N+1>`),
moves the alias to it in one step, then deletes the old versions, so the app never sees half an index.
A snapshot saved here (`data/corpus/snapshots`, mounted into the container) can be restored on another
computer with no crawling and no API key. The client comes from ratio.precedent_opensearch, which
refuses any OpenSearch that is not on this computer.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from contextlib import closing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ratio.config import FactPatternTaxonomy, default_config
from ratio.paths import corpus_db_path
from ratio.precedent_opensearch import connect, index_body
from ratio.precedent_store import SqlitePrecedentIndex

_log = logging.getLogger(__name__)

SNAPSHOT_REPOSITORY = "ratio"
SNAPSHOT_LOCATION = "/usr/share/opensearch/snapshots"  # path.repo in docker-compose.yml
BULK_CHUNK = 500  # documents per bulk request
LOAD_TIMEOUT_S = 120.0
_SNAPSHOT_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,200}$")


def _versions(client: Any, alias: str) -> dict[int, str]:
    """The versioned indices of this alias, by version number."""
    found = client.indices.get(index=f"{alias}-v*", expand_wildcards="all", allow_no_indices=True)
    versions = {}
    for name in found:
        suffix = name[len(alias) + 2 :]
        if name.startswith(f"{alias}-v") and suffix.isdigit():
            versions[int(suffix)] = name
    return versions


def _move_alias(client: Any, alias: str, index: str) -> None:
    """Point the alias at ``index`` only, in one atomic update."""
    current = list(client.indices.get_alias(name=alias)) if client.indices.exists_alias(name=alias) else []
    actions = [{"remove": {"index": name, "alias": alias}} for name in current if name != index]
    client.indices.update_aliases(body={"actions": [*actions, {"add": {"index": index, "alias": alias}}]})


def _documents(corpus: SqlitePrecedentIndex, index: str) -> Iterator[dict[str, Any]]:
    """The bulk request lines: one document per fact quote and per passage, with its vector. Only ids,
    offsets and vectors: the text stays in the corpus file (a private collection's included)."""
    for doc, facet_id, quote, vec in corpus.fact_vectors():
        span = quote.span
        yield {"index": {"_index": index, "_id": f"{doc.id}/{facet_id}/{span.start}-{span.end}"}}
        yield {"precedent_id": doc.id, "facet_id": facet_id, "role": "fact", "kind": doc.kind, "start": span.start, "end": span.end,
               "vec": [float(x) for x in vec]}  # fmt: skip
    for doc, start, end, vec in corpus.passage_vectors():
        yield {"index": {"_index": index, "_id": f"{doc.id}/passage/{start}-{end}"}}
        yield {"precedent_id": doc.id, "role": "passage", "kind": doc.kind, "start": start, "end": end,
               "vec": [float(x) for x in vec]}  # fmt: skip


def _bulk_load(client: Any, corpus: SqlitePrecedentIndex, index: str) -> int:
    lines = list(_documents(corpus, index))
    for first in range(0, len(lines), 2 * BULK_CHUNK):
        reply = client.bulk(body=lines[first : first + 2 * BULK_CHUNK])
        if reply.get("errors"):
            failed = next(item for item in reply["items"] if item["index"].get("error"))
            raise RuntimeError(f"OpenSearch refused a precedent quote: {failed['index']['error']}")
    return len(lines) // 2


def index_corpus(db_path: Path, url: str, alias: str, *, taxonomy: FactPatternTaxonomy | None = None) -> int:
    """Index the corpus file behind ``alias``; returns the number of fact quotes indexed."""
    corpus = SqlitePrecedentIndex(db_path, taxonomy)  # refuses a corpus built for another taxonomy or model
    client = connect(url, timeout=LOAD_TIMEOUT_S)
    old = _versions(client, alias)
    index = f"{alias}-v{max(old, default=0) + 1}"
    client.indices.create(index=index, body=index_body(corpus.meta()))
    try:
        count = _bulk_load(client, corpus, index)
        client.indices.refresh(index=index)
        _move_alias(client, alias, index)
    except Exception:
        client.indices.delete(index=index, ignore_unavailable=True)
        raise
    for name in old.values():
        client.indices.delete(index=name)
    _log.info("indexed %d fact quotes and passages into %s (alias %s)", count, index, alias)
    return count


def _repository(client: Any) -> None:
    body = {"type": "fs", "settings": {"location": SNAPSHOT_LOCATION}}
    client.snapshot.create_repository(repository=SNAPSHOT_REPOSITORY, body=body)


def _checked_name(name: str) -> str:
    if not _SNAPSHOT_NAME.match(name):
        raise ValueError(f"snapshot name {name!r}: use lowercase letters, digits, '.', '_' and '-'")
    return name


def _default_alias() -> str:
    return default_config().settings.precedents.opensearch_index


def private_documents(corpus_path: Path) -> int:
    """How many documents of private collections the corpus file holds (0 when it has none or is missing)."""
    if not Path(corpus_path).is_file():
        return 0
    with closing(sqlite3.connect(f"file:{corpus_path}?mode=ro", uri=True)) as db:
        try:
            return int(db.execute("SELECT COUNT(*) FROM documents WHERE private = 1").fetchone()[0])
        except sqlite3.OperationalError:  # a corpus from before collections
            return 0


def snapshot(url: str, name: str, *, alias: str | None = None, corpus_path: Path | None = None, include_private: bool = False) -> tuple[str, ...]:
    """Save the index behind the alias (and the alias) as snapshot ``name``; returns the indices saved. A
    snapshot is meant to be copied to other computers, and the index holds the ids and vectors of every
    document, so it is refused while the corpus holds a private collection, unless include_private."""
    private = private_documents(corpus_path or corpus_db_path())
    if private and not include_private:
        raise ValueError(f"the corpus holds {private} documents of private collections; a snapshot would carry their ids and vectors off this computer")
    alias = alias or _default_alias()
    client = connect(url, timeout=LOAD_TIMEOUT_S)
    _repository(client)
    body = {"indices": f"{alias}-v*", "include_global_state": False}
    reply = client.snapshot.create(repository=SNAPSHOT_REPOSITORY, snapshot=_checked_name(name), body=body, wait_for_completion=True)
    return tuple(reply["snapshot"]["indices"])


def restore(url: str, name: str, *, alias: str | None = None) -> str:
    """Replace the alias's indices with those of snapshot ``name``; returns the index the alias now names."""
    alias = alias or _default_alias()
    client = connect(url, timeout=LOAD_TIMEOUT_S)
    _repository(client)
    saved = client.snapshot.get(repository=SNAPSHOT_REPOSITORY, snapshot=_checked_name(name))["snapshots"][0]["indices"]
    versions = sorted((int(index.rsplit("-v", 1)[1]), index) for index in saved if re.fullmatch(rf"{re.escape(alias)}-v\d+", index))
    if not versions:
        raise ValueError(f"snapshot {name!r} holds no index of {alias!r}")
    for index in _versions(client, alias).values():
        client.indices.delete(index=index)
    body = {"indices": ",".join(index for _, index in versions), "include_aliases": True, "include_global_state": False}
    client.snapshot.restore(repository=SNAPSHOT_REPOSITORY, snapshot=name, body=body, wait_for_completion=True)
    newest = versions[-1][1]
    _move_alias(client, alias, newest)
    return newest
