"""The precedent library: search every paragraph of every collection by the facts you type, and add your
own documents (no crawling). A private collection is read only by the local model on this computer."""

from __future__ import annotations

import logging

import streamlit as st

import corpus_builder.collections as col
from corpus_builder.store import BuildStore
from ratio.display import md_escape
from ratio.embeddings import EmbeddingModelMissing
from ratio.precedent_schema import PassageHit, PrecedentDoc, PrecedentIndex
from ratio_ui import session, style, viewer, widgets

_log = logging.getLogger(__name__)
SEARCH_RESULTS = 8
UPLOAD_TYPES = ["pdf", "txt", "md", "html", "htm"]


def _hit(doc: PrecedentDoc, hit: PassageHit, number: int) -> None:
    with style.panel(f"library-hit-{number}"):
        st.markdown(f"**{md_escape(doc.title)}**")
        source, origin = widgets.label(f"precedent_{doc.kind}"), viewer.precedent_origin(doc)
        st.markdown(f":blue-badge[{md_escape(source)}] :gray-badge[{md_escape(origin)}]")
        details = [doc.body, doc.state, str(doc.year) if doc.year else None, f"closeness {hit.cosine:.2f}"]
        st.caption(md_escape(" · ".join(part for part in details if part)))
        st.html(widgets.precedent_quote_html(widgets.label("library_passage"), hit.quote, clip=False))
        if st.button("View source", key=f"library-source-{number}", help="Opens the passage, checked against its document"):
            viewer.show_precedent(hit.quote.span, doc, "Search result")


def _search(index: PrecedentIndex) -> None:
    st.header("Search the library")
    st.caption(md_escape(notes["library_search_intro"]))
    query = st.text_input("Facts to look for", placeholder="e.g. journalist held for five days before seeing a judge", key="library_query")
    docs = {doc.id: doc for doc in index.documents()}
    sources = sorted({doc.body for doc in docs.values()})
    chosen = st.multiselect("Only in these sources", sources, key="library_sources", placeholder="All sources")
    if not query.strip():
        return
    ids = [pid for pid, doc in docs.items() if doc.body in chosen] if chosen else None
    try:
        hits = index.search_passages(session.embedder().encode([query])[0], words=query, k=SEARCH_RESULTS, precedent_ids=ids)
    except EmbeddingModelMissing as exc:
        st.error(md_escape(str(exc)))
        return
    if not hits:
        st.caption(md_escape(notes["library_no_hits"]))
    for number, hit in enumerate(hits):
        _hit(docs[hit.precedent_id], hit, number)


def _add() -> None:
    st.header("Add your own documents")
    st.caption(md_escape(notes["library_add_intro"]))
    with st.form("library_add"):
        name = st.text_input("Collection name", placeholder="e.g. Indonesia")
        region = st.text_input("Region (optional)", placeholder="e.g. South-East Asia")
        private = st.toggle("Private: read only by the local model on this computer", value=True)
        files = st.file_uploader("Documents (PDF, text or HTML; English for now)", type=UPLOAD_TYPES, accept_multiple_files=True)
        submitted = st.form_submit_button("Add to the library")
    if not submitted:
        return
    if not files:
        st.error("Choose at least one document to add.")
        return
    try:
        found = col.create_collection(name, region=region, private=private)
        store = BuildStore()
        for upload in files:
            col.add_file(found, upload.name, upload.getvalue(), store)
    except col.CollectionError as exc:
        st.error(md_escape(str(exc)))
        return
    st.success(f"Added {len(files)} documents to {found.name}. Read them below with the local model.")


def _collections() -> None:
    found = col.list_collections()
    if not found:
        return
    st.subheader("Your collections")
    rows = [{"Collection": c.name, "Region": c.region, "Private": "yes" if c.private else "no", "Documents": len(col.files_in(c))} for c in found]
    st.dataframe(rows, hide_index=True)
    st.caption(md_escape(notes["library_build_intro"]))
    if st.button("Read the documents with the local model", key="library_build", type="primary"):
        _build(found)


def _build(found: tuple[col.Collection, ...]) -> None:
    cfg, settings = session.config(), session.precedent_settings()
    with st.status("Reading your collections on this computer", expanded=True) as box:
        try:
            report = col.build(
                BuildStore(), cfg.fact_patterns, col.local_model(), session.embedder(), collections=found, sources=col.crawled_sources(),
                index=lambda path: col.index_if_running(path, settings.opensearch_url, settings.opensearch_index, cfg.fact_patterns),
                progress=box.write,
            )  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - shown as a message; the reason goes to the log
            _log.warning("Reading the collections failed", exc_info=exc)
            box.update(label="Reading the collections failed", state="error")
            st.error(md_escape(f"{notes['library_build_failed']} ({type(exc).__name__}: {exc})"))
            return
        box.update(label=f"Done: {len(report.documents)} documents from your collections are in the library", state="complete")
    session.clear_precedent_index()
    for line in (*report.skipped, *report.failed):
        st.caption(md_escape(line))


notes = session.config().messages.notes
style.eyebrow("Precedent library")
st.title("Precedent library")
st.markdown(md_escape(notes["library_intro"]))
index, status = session.precedent_index()
if index is None:
    st.info(md_escape(notes["library_empty"]))
    st.caption(md_escape(status))
else:
    st.caption(md_escape(f"{status} · {index.meta().documents} documents"))
    _search(index)
_add()
_collections()
