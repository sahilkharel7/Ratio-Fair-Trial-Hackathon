"""The source viewer: every finding's evidence opens here, highlighted in its own document.

A span is shown only after it is checked character for character against the stored document
(hard rule 2); spans from another case (the judge page links across cases) load that case.
"""

from __future__ import annotations

import streamlit as st

from ratio.display import Mark, excerpt, excerpt_html, marked_html, md_escape
from ratio.provenance import span_is_valid
from ratio.schema import CaseRecord, Document, SourceSpan
from ratio_ui import session, style


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
