"""Reasoning reuse: the judgment's reasoning side by side with the indictment, matched passages
highlighted, legitimate quotation greyed out, and the defence arguments the reasoning answers."""

from __future__ import annotations

import streamlit as st

from ratio.display import Mark, marked_html, md_escape, percent
from ratio.modules.reuse import by_passage
from ratio.results import ArgumentCheck, ReusePair, ReuseResult
from ratio.schema import CaseRecord, Evidence, Flag, SourceSpan
from ratio_ui import widgets

PANEL_HEIGHT = 560
LEGEND = (
    '<div class="ratio-legend"><mark class="ratio-verbatim">verbatim</mark>'
    '<mark class="ratio-paraphrase">close paraphrase</mark>'
    '<mark class="ratio-charge">restates the charge</mark>'
    '<span class="ratio-excluded">greyed text</span><span class="ratio-tag">charge recital</span> '
    "is quotation or not reasoning, with the reason · the same number marks both sides of a match</div>"
)
VIEWS = ("Court's reasoning", "Whole documents")


def _kind(pair: ReusePair, flags: dict[str, Flag]) -> str:
    flag = flags.get(pair.flag_id or "")
    return "charge" if flag is not None and flag.status == "charge_wording" else pair.kind


def _marks(reuse: ReuseResult, numbers: dict[str, str], side: str) -> list[Mark]:
    flags = {flag.id: flag for flag in reuse.flags}
    marks: list[Mark] = []
    if side == "judgment":
        marks += [
            Mark(e.span.start, e.span.end, "excluded", "" if e.reason == "header_or_signature" else widgets.label(e.reason))
            for e in reuse.excluded
        ]
    for pair in reuse.pairs:
        number, kind = numbers.get(pair.flag_id or "", ""), _kind(pair, flags)
        ranges = pair.judgment_ranges if side == "judgment" else pair.indictment_ranges
        span = pair.judgment if side == "judgment" else pair.indictment
        marks += [Mark(r.start, r.end, kind, number) for r in ranges] or [Mark(span.start, span.end, kind, number)]
    return marks


def _metrics(reuse: ReuseResult) -> None:
    def share(chars: int) -> float | None:
        return chars / reuse.reasoning_chars if reuse.reasoning_chars and reuse.score is not None else None

    columns = st.columns(4)
    columns[0].metric(
        "Traceable to the indictment", percent(reuse.score),
        help="Share of the characters of the court's own reasoning that match indictment text, after quotation is excluded.",
    )  # fmt: skip
    columns[1].metric("Verbatim", percent(share(reuse.verbatim_chars)))
    columns[2].metric("Close paraphrase", percent(share(reuse.paraphrase_chars)))
    columns[3].metric("Defence arguments without a response", sum(check.flag_id is not None for check in reuse.arguments))
    if reuse.score is None and reuse.score_note:
        st.caption(f"No score: {md_escape(reuse.score_note)}.")
    if reuse.charge_wording_chars:
        st.caption(f"Restating the charge: {percent(reuse.charge_wording_chars / reuse.reasoning_chars)} of the reasoning, shown but not scored.")


def _window(record: CaseRecord, doc_id: str, spans: list[SourceSpan], whole: bool) -> tuple[int, int]:
    """From the heading of the section holding the first span to the heading after the last one."""
    text = record.document(doc_id).text
    if whole or not spans:
        return 0, len(text)
    first, last = min(s.start for s in spans), max(s.end for s in spans)
    headings = [p.span for p in record.passages_of(doc_id) if p.kind == "heading"]
    start = max((h.start for h in headings if h.start <= first), default=0)
    end = min((h.start for h in headings if h.start >= last), default=len(text))
    return start, end


def _panel(record: CaseRecord, doc_id: str, marks: list[Mark], spans: list[SourceSpan], whole: bool) -> None:
    document = record.document(doc_id)
    start, end = _window(record, doc_id, spans, whole)
    st.markdown(f"**{md_escape(document.title)}**")
    with st.container(height=PANEL_HEIGHT, border=True):
        st.html(marked_html(document.text[start:end], marks, offset=start))


def _side_by_side(record: CaseRecord, reuse: ReuseResult, numbers: dict[str, str]) -> None:
    view = st.segmented_control("Show", VIEWS, default=VIEWS[0], key="reuse_view", label_visibility="collapsed")
    whole = view == VIEWS[1]
    st.html(LEGEND)
    reasoning = set(reuse.reasoning_passage_ids)
    left, right = st.columns(2, gap="medium")
    with left:
        spans = [p.span for p in record.passages if p.id in reasoning] + [pair.judgment for pair in reuse.pairs]
        _panel(record, reuse.judgment_doc_id, _marks(reuse, numbers, "judgment"), spans, whole)
    with right:
        if not reuse.indictment_doc_ids:
            st.info("This case has no indictment to compare with.")
            return
        doc_id = reuse.indictment_doc_ids[0]
        if len(reuse.indictment_doc_ids) > 1:
            titles = {i: record.document(i).title for i in reuse.indictment_doc_ids}
            doc_id = st.selectbox("Indictment", list(titles), format_func=titles.get, key="reuse_indictment")
        pairs = [pair.indictment for pair in reuse.pairs if pair.indictment.doc_id == doc_id]
        _panel(record, doc_id, _marks(reuse, numbers, "indictment"), pairs, whole)


def _match(record: CaseRecord, flag: Flag, number: str) -> None:
    with st.container(border=True):
        heading, status = st.columns([8, 3], vertical_alignment="center")
        heading.markdown(f"**{number}.** {md_escape(flag.message)}")
        with status:
            widgets.badge(flag.status)
        widgets.evidence(record, flag.evidence, key=f"match-{flag.id}", heading=f"Matched passage {number}")


def _argument(record: CaseRecord, reuse: ReuseResult, check: ArgumentCheck) -> None:
    flag = next((f for f in reuse.flags if f.id == check.flag_id), None)
    with st.container(border=True):
        widgets.badge("unchecked_argument" if not check.checked else "addressed_argument" if check.addressed else "unaddressed_argument")
        widgets.evidence(record, [Evidence(role="argument", span=check.argument)], key=f"argument-{check.argument_id}", heading="Defence argument")
        if flag is not None:
            st.markdown(md_escape(flag.message))
        if not check.checked:
            st.caption("The model's answer did not name any passage, so this argument was not checked.")
        if check.responding:
            st.caption(f"Answered in the reasoning ({check.passages_checked} of {check.passages_total} passages checked):")
            answers = [Evidence(role="judgment", span=span) for span in check.responding]
            widgets.evidence(record, answers, key=f"answer-{check.argument_id}", heading="Response in the judgment")
        widgets.model_note(check.model_note)


loaded = widgets.require_case()
record, reuse = loaded.record, loaded.analysis.reuse
widgets.header(loaded, "Reasoning reuse", "the judgment's reasoning compared with the indictment")
st.markdown(
    "Quoted statutes, the recited charge and positions attributed to a party are excluded first. What remains is the "
    "court's own reasoning; highlighted text in it matches the indictment word for word or as a close paraphrase. "
    "Similarity shows where to look; whether it matters is for the reviewing lawyer."
)
if reuse is None or reuse.judgment_doc_id is None:
    st.info("This case has no judgment to compare.")
    st.stop()
matches = {flag_id: pairs for flag_id, pairs in by_passage(reuse.pairs).items()}
numbers = {flag_id: str(n) for n, flag_id in enumerate(matches, start=1)}
flags = {flag.id: flag for flag in reuse.flags}
_metrics(reuse)
_side_by_side(record, reuse, numbers)
st.subheader("Matched passages")
if not matches:
    st.caption("No passage of the reasoning matches the indictment.")
for flag_id in matches:
    if flag_id in flags:
        _match(record, flags[flag_id], numbers[flag_id])
st.subheader("Defence arguments from the notes")
if not reuse.arguments:
    st.caption("No defence arguments were found in the notes.")
for check in reuse.arguments:
    _argument(record, reuse, check)
