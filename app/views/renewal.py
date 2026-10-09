"""Detention renewals: each detention order in date order, the share of its grounds that repeats the
orders before it (highlighted side by side), and the days between one order's end date and the next
order. Repetition shows where to look; whether detention was re-examined is for the reviewing lawyer."""

from __future__ import annotations

import streamlit as st

from ratio.display import Mark, marked_html, md_escape, percent
from ratio.jurisprudence import for_finding
from ratio.results import OrderSummary, RenewalResult
from ratio.schema import CaseRecord
from ratio_ui import session, widgets

PANEL_HEIGHT = 420
LEGEND = (
    '<div class="ratio-legend"><mark class="ratio-verbatim">repeated word for word</mark>'
    '<mark class="ratio-paraphrase">close paraphrase</mark>'
    '<span class="ratio-excluded">greyed text</span> is quoted law, the request, the operative part or the caption</div>'
)


def _table(renewal: RenewalResult) -> None:
    flagged = {flag.id for flag in renewal.flags}
    rows = [
        {
            "Order": order.title,
            "Date": order.date.isoformat() if order.date else "undated",
            "Detention until": order.until.isoformat() if order.until else "not read",
            "Days since the previous order": "" if order.days_since_previous is None else str(order.days_since_previous),
            "Grounds repeating earlier orders": "" if order.share_repeated is None else percent(order.share_repeated),
            "New grounds passages": f"{order.passages_new} of {order.passages_compared}" if order.share_repeated is not None else "",
            "Status": widgets.label("repeated_grounds") if order.flag_id in flagged else "",
        }
        for order in renewal.orders
    ]
    st.dataframe(rows, hide_index=True)


def _excluded(order: OrderSummary) -> list[Mark]:
    return [
        Mark(e.span.start, e.span.end, "excluded", "" if e.reason == "header_or_signature" else widgets.label(e.reason))
        for e in order.excluded
    ]


def _matched(order: OrderSummary, doc_id: str, side: str) -> list[Mark]:
    """The repeated text of ``order``'s pairs on one side: this order ("later") or an earlier one."""
    marks = []
    for pair in order.pairs:
        span, ranges = (pair.later, pair.later_ranges) if side == "later" else (pair.earlier, pair.earlier_ranges)
        if span.doc_id == doc_id:
            marks += [Mark(r.start, r.end, pair.kind) for r in ranges] or [Mark(span.start, span.end, pair.kind)]
    return marks


def _earlier_marks(renewal: RenewalResult, order: OrderSummary, doc_id: str) -> list[Mark]:
    """The earlier order's matched text, with its own quoted law and operative part greyed."""
    own = next((o for o in renewal.orders if o.doc_id == doc_id), None)
    return (_excluded(own) if own else []) + _matched(order, doc_id, "earlier")


def _grounds_start(record: CaseRecord, renewal: RenewalResult, doc_id: str) -> int:
    """Where the order's grounds section begins (its heading), so the panel opens on the grounds."""
    own = next((o for o in renewal.orders if o.doc_id == doc_id), None)
    passages = record.passages_of(doc_id)
    grounds = [p.span.start for p in passages if own is not None and p.id in own.grounds_passage_ids]
    if not grounds:
        return 0
    return max((p.span.start for p in passages if p.kind == "heading" and p.span.start <= grounds[0]), default=grounds[0])


def _panel(record: CaseRecord, renewal: RenewalResult, doc_id: str, marks: list[Mark]) -> None:
    document = record.document(doc_id)
    start = _grounds_start(record, renewal, doc_id)
    st.markdown(f"**{md_escape(document.title)}**")
    with st.container(height=PANEL_HEIGHT, border=True):
        st.html(marked_html(document.text[start:], marks, offset=start))


def _order(record: CaseRecord, renewal: RenewalResult, order: OrderSummary) -> None:
    flag = next((f for f in renewal.flags if f.id == order.flag_id), None)
    with st.container(border=True):
        heading, status = st.columns([8, 3], vertical_alignment="center")
        until = order.until.isoformat() if order.until else "not read"
        heading.markdown(f"**{md_escape(order.title)}** · detention until {until}")
        if flag is not None:
            with status:
                widgets.badge(flag.status)
        if flag is None:  # a flagged order's message says the same, with its sources
            st.markdown(
                f"{percent(order.share_repeated)} of its grounds repeat earlier orders; "
                f"{order.passages_new} of its {order.passages_compared} grounds passages long enough to compare are new."
            )
        else:
            st.markdown(md_escape(flag.message))
            st.caption(md_escape(flag.citation or ""))
            with st.expander(f"Sources ({len(flag.evidence)})"):
                widgets.evidence(record, flag.evidence, key=f"renewal-{flag.id}", heading=flag.standard_label)
            widgets.jurisprudence(for_finding(flag, session.config().jurisprudence))
            widgets.state_reply(record, _replies.for_flag(flag.id) if _replies else None, key=f"state-{flag.id}")
        sources = list(dict.fromkeys(pair.earlier.doc_id for pair in order.pairs))
        if not sources:
            return
        with st.expander("Side by side with the earlier order", expanded=flag is not None):
            earlier = sources[0]
            if len(sources) > 1:
                titles = {doc_id: record.document(doc_id).title for doc_id in sources}
                earlier = st.selectbox("Earlier order", sources, format_func=titles.get, key=f"earlier-{order.doc_id}")
            st.html(LEGEND)
            left, right = st.columns(2, gap="medium")
            with left:
                _panel(record, renewal, order.doc_id, _excluded(order) + _matched(order, order.doc_id, "later"))
            with right:
                _panel(record, renewal, earlier, _earlier_marks(renewal, order, earlier))


def _gaps(record: CaseRecord, renewal: RenewalResult) -> None:
    flags = {flag.id: flag for flag in renewal.flags}
    for gap in renewal.gaps:
        flag = flags.get(gap.flag_id or "")
        if flag is None:
            continue
        with st.container(border=True):
            with st.container(horizontal=True):
                widgets.badge(flag.status)
                widgets.badge(flag.review_status)
            st.markdown(md_escape(flag.message))
            st.caption(md_escape(flag.citation or ""))
            widgets.evidence(record, flag.evidence, key=f"gap-{gap.id}", heading=flag.standard_label)
            widgets.jurisprudence(for_finding(flag, session.config().jurisprudence))
            widgets.state_reply(record, _replies.for_flag(flag.id) if _replies else None, key=f"state-{flag.id}")


loaded = widgets.require_case()
record, renewal = loaded.record, loaded.analysis.renewal
_replies = loaded.analysis.steelman
widgets.header(loaded, "Detention renewals", "each detention order compared with the orders before it")
st.markdown(md_escape(session.config().messages.notes["renewal_intro"]))
if renewal is None or not renewal.orders:
    st.info("This case has no detention orders. List them in case.yaml with `type: detention_order`.")
    st.stop()
_table(renewal)
for note in renewal.notes:
    st.caption(md_escape(note))
st.subheader("Gaps between orders")
if not renewal.gaps:
    st.caption("Each order in the record was made no later than the day after the previous one ended.")
_gaps(record, renewal)
st.subheader("Grounds of each renewal")
renewals = [order for order in renewal.orders if order.share_repeated is not None]
if not renewals:
    st.caption("There is no later order with grounds to compare.")
for order in renewals:
    _order(record, renewal, order)
