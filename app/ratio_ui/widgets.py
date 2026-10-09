"""Page elements shared by the views: header, status badges, and evidence rows whose buttons open
the source viewer. Document and model text is always escaped before it is shown."""

from __future__ import annotations

import html
import logging
from collections.abc import Mapping, Sequence

import streamlit as st

from ratio.display import md_escape
from ratio.jurisprudence import Reference
from ratio.precedent_schema import PrecedentDoc, PrecedentLink, PrecedentQuote, SharedFacet
from ratio.results import StateReply
from ratio.schema import CaseRecord, Evidence, SourceSpan
from ratio_ui import session, style, viewer
from ratio_ui.session import LoadedCase

_log = logging.getLogger(__name__)
QUOTE_CHARS = 420
VIEWS_FINDINGS = frozenset({"violation_found", "no_violation", "not_examined"})
FINDINGS_SHOWN: Mapping[str, frozenset[str]] = {  # the outcomes each kind of source states; any other is not shown
    "ccpr_views": VIEWS_FINDINGS,
    "wgad_opinion": VIEWS_FINDINGS,
    "trialwatch_report": frozenset({"monitor_assessment", "not_examined"}),  # a monitor assesses, it finds no violation
}
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


# --- Similar cases ------------------------------------------------------------------------------


def _clipped(text: str) -> str:
    return text if len(text) <= QUOTE_CHARS else text[:QUOTE_CHARS].rsplit(" ", 1)[0] + " …"


def finding_label(doc: PrecedentDoc, shared: SharedFacet) -> str | None:
    """How the deciding body's words are labelled, e.g. "Committee found a violation"; None for a kind of
    finding its kind of source does not make, which is then not shown."""
    kind = shared.finding_kind
    if kind not in FINDINGS_SHOWN.get(doc.kind, frozenset()):
        return None
    return session.config().messages.labels.get(f"finding_{kind}_{doc.kind}") or label(f"finding_{kind}")


def narrow_evidence(record: CaseRecord, items: Sequence[Evidence], key: str, heading: str = "") -> None:
    """Evidence rows for a half-width column: each quote with its View source button under it, so the
    button's label is never cut short."""
    for index, item in enumerate(items):
        st.html(quote_html(record, item))
        source_button(item.span, record, key=f"{key}-{index}", heading=heading)


def precedent_quote_html(where: str, quote: PrecedentQuote, *, clip: bool = True) -> str:
    text = _clipped(quote.span.text) if clip else quote.span.text
    return f'<div class="ratio-quote"><span class="ratio-where">{html.escape(where)}</span>“{html.escape(text)}”</div>'


def precedent_evidence(doc: PrecedentDoc, shared: SharedFacet, key: str) -> None:
    """A public past case's side of a shared fact pattern: its facts and, when quoted, what its deciding
    body said about them, each with a button that opens the passage checked against the corpus. A
    finding is shown whole: the corpus build bounds its length, and a cut could drop its "not"."""
    quotes = [(label("precedent_fact"), shared.precedent_fact, True)]
    finding = finding_label(doc, shared)
    if shared.precedent_finding is not None and finding is not None:  # an outcome only as the body's own words
        quotes.append((finding, shared.precedent_finding, False))
    for index, (where, quote, clip) in enumerate(quotes):  # laid out like narrow_evidence, beside it
        st.html(precedent_quote_html(where, quote, clip=clip))
        if st.button("View source", key=f"{key}-{index}", help="Opens the exact passage, highlighted in the public document and checked against it"):
            viewer.show_precedent(quote.span, doc, shared.label)


def similar_link(links: Sequence[PrecedentLink] | None) -> None:
    """The Case page's line for Similar cases, from session.similar_links_ready: nothing while the links
    are computing or when no corpus is installed. It never fails, as the Case page is the demo's main page."""
    if links is None:
        return
    try:
        count = len(links)
        cases = "1 public case shares" if count == 1 else f"{count} public cases share"
        min_shared = session.precedent_settings().min_shared
        st.page_link("views/similar.py", label=f"Similar cases: {cases} at least {min_shared} fact patterns")
    except Exception:  # noqa: BLE001 - an optional line: the Similar cases page shows what went wrong
        _log.warning("Similar cases line left out of the Case page", exc_info=True)
