"""Jurisprudence: for each finding and each question for the monitor in this case, the General Comment
paragraphs and Committee decisions linked to it by the standard it concerns; then the whole corpus.
Nothing here says that an entry applies to the case: that is for the reviewing lawyer."""

from __future__ import annotations

import html

import streamlit as st

from ratio.config import JurisprudenceEntry
from ratio.display import md_escape
from ratio.jurisprudence import Reference, for_finding, for_follow_up, follow_ups
from ratio.schema import Flag
from ratio_ui import session, widgets


def _quote(ref: Reference) -> None:
    st.html(f'<div class="ratio-quote"><span class="ratio-where">{html.escape(ref.citation)}</span>{html.escape(ref.shown)}</div>')


def _item(title: str, status: str, references: tuple[Reference, ...], flag: Flag | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**{md_escape(title)}**")
        widgets.badge(status)
        if flag is not None:  # a finding is never shown without the text it rests on
            widgets.evidence(loaded.record, flag.evidence, key=f"jurisprudence-{flag.id}", heading=flag.standard_label)
        if not references:
            st.caption("No General Comment paragraph or Committee decision in the corpus concerns this standard.")
        for ref in references:
            _quote(ref)


def _standard_names() -> dict[str, str]:
    """Each standard under the name the findings use, so the corpus list never shows an internal id."""
    named = (*cfg.rubric.items, *cfg.benchmarks.benchmarks)
    return {item.id: f"{item.provision}: {item.name}" for item in named} | {
        standard_id: standard.label for standard_id, standard in cfg.standards.standards.items()
    }


def _scope(entry: JurisprudenceEntry, names: dict[str, str]) -> str:
    """Which findings a corpus entry is shown next to."""
    standards = "Standards: " + " · ".join(names.get(standard, standard) for standard in entry.standards) + "."
    if not entry.statuses:
        return standards
    return standards + " Shown only next to findings marked " + " or ".join(f"“{widgets.label(s)}”" for s in entry.statuses) + "."


loaded = widgets.require_case()
cfg = session.config()
corpus = cfg.jurisprudence
widgets.header(loaded, "Jurisprudence", "General Comments 32 and 35, and the Committee decisions they cite, by standard")
st.markdown(md_escape(cfg.messages.notes["jurisprudence_intro"]))
st.caption(md_escape(cfg.messages.notes["jurisprudence_caveat"]) + f" Last checked on {md_escape(corpus.checked)}.")
st.header("Findings in this case")
for flag in session.findings(loaded):
    _item(f"{flag.standard_label}: {flag.message}", flag.status, for_finding(flag, corpus), flag)
questions = follow_ups(loaded.analysis)
if questions:
    st.header("Questions for the monitor")
    rubric = {item.id: item for item in cfg.rubric.items}
    for question in questions:
        item = rubric[question.rubric_id]
        _item(f"{item.provision}: {item.name}", "no_evidence", for_follow_up(question, corpus))
with st.expander(f"The whole corpus ({len(corpus.entries)} entries)"):
    standard_names = _standard_names()
    for entry in corpus.entries:
        ref = Reference(entry, corpus.sources[entry.source])
        _quote(ref)
        st.caption(md_escape(_scope(entry, standard_names)))
