"""Reasoning reuse: the judgment's reasoning side by side with the indictment, matched passages
highlighted, legitimate quotation greyed out, and the defence arguments the reasoning answers."""

from __future__ import annotations

import streamlit as st

from ratio.display import Mark, marked_html, md_escape, percent, reuse_marks
from ratio.modules.reuse import by_passage
from ratio.results import ArgumentCheck, ReuseResult
from ratio.schema import CaseRecord, Evidence, Flag, SourceSpan
from ratio_ui import style, widgets

PANEL_HEIGHT = 560
LEGEND = (
    '<div class="ratio-legend"><mark class="ratio-verbatim">verbatim</mark>'
    '<mark class="ratio-paraphrase">close paraphrase</mark>'
    '<mark class="ratio-charge">restates the charge</mark>'
    '<span class="ratio-excluded">greyed text</span><span class="ratio-tag">charge recital</span> '
    "is quotation or not reasoning, with the reason · the same number marks both sides of a match</div>"
)
VIEWS = ("Court's reasoning", "Whole documents")


def _metrics(reuse: ReuseResult) -> None:
    def share(chars: int) -> float | None:
        return chars / reuse.reasoning_chars if reuse.reasoning_chars and reuse.score is not None else None

    with st.container(key="ratio-metric-grid-reuse"):
        columns = st.columns(4)
        with columns[0]:
            style.metric("Traceable reasoning", percent(reuse.score), "Share matching indictment text")
        with columns[1]:
            style.metric("Verbatim", percent(share(reuse.verbatim_chars)), "Word-for-word matches")
        with columns[2]:
            style.metric("Close paraphrase", percent(share(reuse.paraphrase_chars)), "Similar wording in the reasoning")
        with columns[3]:
            style.metric("Unanswered arguments", sum(check.flag_id is not None for check in reuse.arguments), "Defence arguments with no response found")
    st.caption("Percentages measure characters of the court's own reasoning, with quotations excluded.")
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
    with style.panel(f"comparison-{doc_id}", height=PANEL_HEIGHT):
        st.html(marked_html(document.text[start:end], marks, offset=start))


def _side_by_side(record: CaseRecord, reuse: ReuseResult, numbers: dict[str, str]) -> None:
    view = st.segmented_control("Show", VIEWS, default=VIEWS[0], key="reuse_view", label_visibility="collapsed")
    whole = view == VIEWS[1]
    st.html(LEGEND)
    reasoning = set(reuse.reasoning_passage_ids)
    left, right = st.columns(2, gap="medium")
    with left:
        spans = [p.span for p in record.passages if p.id in reasoning] + [pair.judgment for pair in reuse.pairs]
        _panel(record, reuse.judgment_doc_id, reuse_marks(reuse, numbers, reuse.judgment_doc_id, widgets.label), spans, whole)
    with right:
        if not reuse.indictment_doc_ids:
            st.info("This case has no indictment to compare with.")
            return
        doc_id = reuse.indictment_doc_ids[0]
        if len(reuse.indictment_doc_ids) > 1:
            titles = {i: record.document(i).title for i in reuse.indictment_doc_ids}
            doc_id = st.selectbox("Indictment", list(titles), format_func=titles.get, key="reuse_indictment")
        pairs = [pair.indictment for pair in reuse.pairs if pair.indictment.doc_id == doc_id]
        _panel(record, doc_id, reuse_marks(reuse, numbers, doc_id, widgets.label), pairs, whole)


def _match(record: CaseRecord, flag: Flag, number: str) -> None:
    with style.panel(f"match-{flag.id}"):
        heading, status = st.columns([8, 3], vertical_alignment="center")
        heading.markdown(f"**{number}.** {md_escape(flag.message)}")
        with status:
            widgets.badge(flag.status)
        widgets.evidence(record, flag.evidence, key=f"match-{flag.id}", heading=f"Matched passage {number}")


def _argument(record: CaseRecord, reuse: ReuseResult, check: ArgumentCheck) -> None:
    flag = next((f for f in reuse.flags if f.id == check.flag_id), None)
    with style.panel(f"argument-{check.argument_id}"):
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
style.section("Document comparison", "Read the court's reasoning alongside the prosecution's indictment. Match numbers connect both documents.")
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
