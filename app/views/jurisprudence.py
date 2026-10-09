"""Jurisprudence: for each finding and each question for the monitor in this case, the General Comment
paragraphs and Committee decisions linked to it by the standard it concerns; then the whole corpus.
Nothing here says that an entry applies to the case: that is for the reviewing lawyer."""

from __future__ import annotations

import html

import streamlit as st

from ratio.display import md_escape
from ratio.jurisprudence import Reference, for_finding, for_follow_up, follow_ups
from ratio_ui import session, widgets


def _quote(ref: Reference) -> None:
    st.html(f'<div class="ratio-quote"><span class="ratio-where">{html.escape(ref.citation)}</span>{html.escape(ref.shown)}</div>')


def _item(title: str, status: str, references: tuple[Reference, ...]) -> None:
    with st.container(border=True):
        st.markdown(f"**{md_escape(title)}**")
        widgets.badge(status)
        if not references:
            st.caption("No entry of the corpus concerns this standard.")
        for ref in references:
            _quote(ref)


loaded = widgets.require_case()
cfg = session.config()
corpus = cfg.jurisprudence
widgets.header(loaded, "Jurisprudence", "General Comments 32 and 35 and the decisions they cite, by standard")
st.markdown(md_escape(cfg.messages.notes["jurisprudence_intro"]))
st.caption(md_escape(cfg.messages.notes["jurisprudence_caveat"]) + f" Checked against the sources on {md_escape(corpus.checked)}.")
st.subheader("Findings in this case")
for flag in session.findings(loaded):
    _item(f"{flag.standard_label}: {flag.message}", flag.status, for_finding(flag, corpus))
questions = follow_ups(loaded.analysis)
if questions:
    st.subheader("Questions for the monitor")
    rubric = {item.id: item for item in cfg.rubric.items}
    for question in questions:
        item = rubric[question.rubric_id]
        _item(f"{item.provision}: {item.name}", "no_evidence", for_follow_up(question, corpus))
with st.expander(f"The whole corpus ({len(corpus.entries)} entries)"):
    for entry in corpus.entries:
        ref = Reference(entry, corpus.sources[entry.source])
        _quote(ref)
        st.caption(md_escape("Standards: " + ", ".join(entry.standards) + (f"; only for: {', '.join(entry.statuses)}" if entry.statuses else "")))
