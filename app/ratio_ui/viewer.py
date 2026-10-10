"""The source viewer: every finding's evidence opens here, highlighted in its own document.

A span is shown only after it is checked character for character against the stored document
(hard rule 2); spans from another case (the judge page links across cases) load that case, and a
public past case's passage (Similar cases) is checked against the precedent corpus text and shown
as an excerpt with its source address and attribution, never as the whole document.
"""

from __future__ import annotations

import streamlit as st

from ratio.display import Mark, excerpt, excerpt_html, marked_html, md_escape
from ratio.precedent_schema import PrecedentDoc
from ratio.provenance import span_is_valid
from ratio.schema import CaseRecord, Document, SourceSpan
from ratio_ui import session, style

SYNTHETIC_MARK = "SYNTHETIC"  # the first word of every test fixture document, never of a public one


@st.dialog("Source document", width="large")
def show_document(record: CaseRecord, doc_id: str) -> None:
    """Read an imported document without manufacturing an evidence span."""
    document = record.document(doc_id)
    if record.meta.synthetic:
        style.banner(session.config().messages.notes["synthetic_banner"])
    st.subheader(md_escape(document.title))
    st.caption(f"{document.type.replace('_', ' ').title()} · {len(document.text):,} characters · Original source text")
    with style.panel("source-document", height=560):
        st.html(marked_html(document.text, []))


def _document(span: SourceSpan, record: CaseRecord | None) -> tuple[CaseRecord, Document] | None:
    owner = record if record is not None and record.case_id == span.case_id else session.store().load_case(span.case_id)
    if owner is None:
        return None
    try:
        return owner, owner.document(span.doc_id)
    except KeyError:
        return None


@st.dialog("Source text", width="large")
def show_source(span: SourceSpan, record: CaseRecord | None = None, heading: str = "") -> None:
    found = _document(span, record)
    if found is None or not span_is_valid(span, {found[1].id: found[1].text}.get):
        st.error("This passage was not found in its source document, so it is not shown.")
        return
    owner, document = found
    cfg = session.config()
    if owner.meta.synthetic:
        style.banner(cfg.messages.notes["synthetic_banner"])
    if heading:
        st.markdown(f"**{md_escape(heading)}**")
    part = excerpt(document.text, span, cfg.settings.ui.context_chars)
    st.caption(
        f"{md_escape(document.title)} · line {part.line} · characters {span.start} to {span.end} · "
        "checked against the source: exact match"
    )
    st.html(excerpt_html(part))
    with st.expander("Whole document"):
        st.html(marked_html(document.text, [Mark(span.start, span.end, "span")]))


def precedent_origin(precedent: PrecedentDoc) -> str:
    """The label "Real public case", unless the document marks itself SYNTHETIC (a test fixture)."""
    if precedent.text.startswith(SYNTHETIC_MARK):
        key = "synthetic_precedent"
    elif precedent.private:
        key = "private_document"
    elif precedent.kind == "collection_document":
        key = "uploaded_public_document"
    else:
        key = "real_public_case"
    return session.config().messages.label(key)


@st.dialog("Source text", width="large")
def show_precedent(span: SourceSpan, precedent: PrecedentDoc, heading: str = "") -> None:
    """A public past case's passage, shown only after it is checked against the corpus text: the excerpt
    around it, the source address and the attribution."""
    index, _ = session.precedent_index()
    resolve = index.text if index is not None and span.doc_id == precedent.doc_id else None
    if resolve is None or not span_is_valid(span, resolve):
        st.error("This passage was not found in its source document, so it is not shown.")
        return
    text = resolve(span.doc_id)
    if heading:
        st.markdown(f"**{md_escape(heading)}**")
    part = excerpt(text, span, session.config().settings.ui.context_chars)
    st.caption(
        f"{md_escape(precedent_origin(precedent))}: {md_escape(precedent.title)} · {md_escape(precedent.body)} · "
        f"characters {span.start} to {span.end} · checked against the source: exact match"
    )
    st.html(excerpt_html(part))
    address = precedent.url.replace("`", "%60")  # as code, so it is shown as plain text and never linked
    st.caption(f"Source: `{address}` · {md_escape(precedent.attribution)}")  # quoted, never republished in full
