"""Page elements shared by the views: header, status badges, and evidence rows whose buttons open
the source viewer. Document and model text is always escaped before it is shown."""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st

from ratio.display import md_escape
from ratio.jurisprudence import Reference
from ratio.results import StateReply
from ratio.schema import CaseRecord, Evidence, SourceSpan
from ratio_ui import session, style, viewer
from ratio_ui.session import LoadedCase

QUOTE_CHARS = 420
BADGE_COLORS = {
    "evidence_of_compliance": "green",
    "evidence_of_violation": "red",
    "no_evidence": "gray",
    "incomplete": "orange",
    "exceeds_benchmark": "red",
    "may_exceed": "orange",
    "needs_review": "orange",
    "within_benchmark": "green",
    "measured": "gray",
    "cannot_compute": "gray",
    "verbatim_reuse": "orange",
    "paraphrase_reuse": "blue",
    "charge_wording": "gray",
    "unaddressed_argument": "orange",
    "addressed_argument": "green",
    "unchecked_argument": "gray",
    "confirmed": "blue",
    "needs_legal_review": "violet",
    "pattern_warrants_review": "blue",  # neutral: a prompt for review, never a finding about the judge
    "hidden_indicator": "gray",
    "repeated_grounds": "orange",
    "order_gap": "orange",
    "review_pending": "gray",
    "review_accepted": "green",
    "review_edited": "blue",
    "review_rejected": "orange",  # red stays reserved for the confirmed benchmark and evidence of violation
}


def label(key: str) -> str:
    return session.config().messages.label(key)


def badge(status: str) -> None:
    st.badge(label(status), color=BADGE_COLORS.get(status, "gray"))


def require_case() -> LoadedCase:
    loaded = session.current_case()
    if loaded is None or loaded.analysis is None:
        st.info("No case is loaded yet. Load the demo case or upload a case folder on the Case page.")
        st.page_link("views/case.py", label="Go to the Case page")
        st.stop()
    return loaded


def header(loaded: LoadedCase, title: str, caption: str) -> None:
    style.eyebrow("Case analysis")
    st.title(title)
    if loaded.record.meta.synthetic:
        style.banner(session.config().messages.notes["synthetic_banner"])
    meta = loaded.record.meta
    st.caption(f"{md_escape(meta.title)} · {md_escape(meta.court)}")
    st.caption(caption[0].upper() + caption[1:] if caption else "")


def _document_title(record: CaseRecord, span: SourceSpan) -> str:
    try:
        return record.document(span.doc_id).title
    except KeyError:
        return span.doc_id


def quote_html(record: CaseRecord, item: Evidence) -> str:
    text = item.span.text
    if len(text) > QUOTE_CHARS:
        text = text[:QUOTE_CHARS].rsplit(" ", 1)[0] + " …"
    where = f"{label('role_' + item.role)} · {_document_title(record, item.span)}"
    return f'<div class="ratio-quote"><span class="ratio-where">{html.escape(where)}</span>“{html.escape(text)}”</div>'


def source_button(span: SourceSpan, record: CaseRecord, key: str, text: str = "View source", heading: str = "") -> None:
    if st.button(text, key=key, help="Opens the exact passage, highlighted in its document and checked against it"):
        viewer.show_source(span, record, heading)


def evidence(record: CaseRecord, items: Sequence[Evidence], key: str, heading: str = "") -> None:
    for index, item in enumerate(items):
        text_column, button_column = st.columns([7, 2], vertical_alignment="center")  # room for "View source"
        text_column.html(quote_html(record, item))
        with button_column:
            source_button(item.span, record, key=f"{key}-{index}", heading=heading)


def progress_bar():
    """A progress bar for reading a case. Labels are document titles from the uploaded case.yaml, and
    st.progress renders Markdown (images included), so they are escaped: a title can never fetch anything."""
    bar = st.progress(0.0, text="Starting")

    def update(label: str, done: int, total: int) -> None:
        bar.progress(done / total, text=f"Reading {md_escape(label)} ({done} of {total})")

    return update


def jurisprudence(references: Sequence[Reference]) -> None:
    """The corpus entries linked to a finding or a question, with the fixed caveat."""
    if not references:
        return
    notes = session.config().messages.notes
    with st.expander(f"{notes['jurisprudence_heading']} ({len(references)})"):
        for ref in references:
            st.html(f'<div class="ratio-quote"><span class="ratio-where">{html.escape(ref.citation)}</span>{html.escape(ref.shown)}</div>')
        st.caption(md_escape(notes["jurisprudence_caveat"]))


def state_reply(record: CaseRecord, reply: StateReply | None, key: str, *, expanded: bool = False) -> None:
    """The State's strongest reply to one finding: each argument's reviewed ground, the model's wording
    (unverified) and the exact record quote it rests on, then the grounds the record did not support."""
    if reply is None:
        return
    cfg = session.config()
    notes = cfg.messages.notes
    grounds = {g.id: g for g in cfg.steelman.for_standard(reply.standard_id)}
    with st.expander(f"{notes['steelman_heading']} ({len(reply.arguments)})", expanded=expanded):
        if not reply.checked:
            st.warning(md_escape(reply.note or "The model gave no usable answer."))
        if reply.checked and not reply.arguments:
            st.caption("The model found no ground the record passages shown to it support.")
        for index, argument in enumerate(reply.arguments):
            ground = grounds.get(argument.ground_id)
            st.markdown(f"**{md_escape(ground.label if ground else argument.ground_id)}**")
            if ground is not None and ground.quote:
                st.caption(md_escape(f"{cfg.jurisprudence.sources[ground.source].symbol}, {ground.pinpoint}: “{ground.quote}”"))
            st.markdown(f"*Local model:* {md_escape(argument.argument)}")
            evidence(record, [Evidence(role="mention", span=argument.span)], key=f"{key}-{index}", heading="The State's reply rests on")
        unsupported = [grounds[g].label if g in grounds else g for g in reply.unsupported_grounds]
        if unsupported:
            st.caption(md_escape(f"{notes['steelman_unsupported']}: " + "; ".join(unsupported) + "."))
        st.caption(md_escape(notes["steelman_caveat"]))


def model_note(text: str | None) -> None:
    if text:
        with st.expander(session.config().messages.notes["model_note_label"]):
            st.markdown(md_escape(text))
