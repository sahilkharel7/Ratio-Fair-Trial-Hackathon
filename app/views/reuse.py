"""Reasoning reuse: the judgment's reasoning side by side with the indictment, matched passages
highlighted, legitimate quotation greyed out, and the defence arguments the reasoning answers."""

from __future__ import annotations

import streamlit as st

from ratio.display import Mark, marked_html, md_escape, percent
from ratio.results import ArgumentCheck, ReusePair, ReuseResult
from ratio.schema import CaseRecord, Evidence, SourceSpan
from ratio_ui import widgets

PANEL_HEIGHT = 560
LEGEND = (
    '<div class="ratio-legend"><mark class="ratio-verbatim">verbatim</mark>'
    '<mark class="ratio-paraphrase">close paraphrase</mark>'
    '<span class="ratio-excluded">greyed text</span><span class="ratio-tag">charge recital</span> '
    "is quotation or not reasoning, with the reason · the same number marks both sides of a match</div>"
)
VIEWS = ("Court's reasoning", "Whole documents")


def _judgment_marks(reuse: ReuseResult, numbers: dict[str, str]) -> list[Mark]:
    marks = [
        Mark(e.span.start, e.span.end, "excluded", "" if e.reason == "header_or_signature" else widgets.label(e.reason))
        for e in reuse.excluded
    ]
    for pair in reuse.pairs:
        if pair.kind == "verbatim":
            marks += [Mark(r.start, r.end, "verbatim", numbers[pair.id]) for r in pair.judgment_ranges]
        else:
            marks.append(Mark(pair.judgment.start, pair.judgment.end, "paraphrase", numbers[pair.id]))
    return marks


def _indictment_marks(reuse: ReuseResult, numbers: dict[str, str]) -> list[Mark]:
    marks = []
    for pair in reuse.pairs:
        if pair.kind == "verbatim":
            marks += [Mark(r.start, r.end, "verbatim", numbers[pair.id]) for r in pair.indictment_ranges]
        else:
            marks.append(Mark(pair.indictment.start, pair.indictment.end, "paraphrase", numbers[pair.id]))
    return marks


def _metrics(reuse: ReuseResult) -> None:
    def share(chars: int) -> float | None:
        return chars / reuse.reasoning_chars if reuse.reasoning_chars else None

    columns = st.columns(4)
    columns[0].metric(
        "Traceable to the indictment", percent(reuse.score),
        help="Share of the characters of the court's own reasoning that match indictment text, after quotation is excluded.",
    )  # fmt: skip
    columns[1].metric("Verbatim", percent(share(reuse.verbatim_chars)))
    columns[2].metric("Close paraphrase", percent(share(reuse.paraphrase_chars)))
    columns[3].metric("Defence arguments without a response", sum(not c.addressed for c in reuse.arguments))


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
    reasoning = [p.span for p in record.passages if p.id in set(reuse.reasoning_passage_ids)]
    left, right = st.columns(2, gap="medium")
    with left:
        spans = reasoning + [pair.judgment for pair in reuse.pairs]
        _panel(record, reuse.judgment_doc_id, _judgment_marks(reuse, numbers), spans, whole)
    with right:
        if reuse.indictment_doc_id is None:
            st.info("This case has no indictment to compare with.")
            return
        _panel(record, reuse.indictment_doc_id, _indictment_marks(reuse, numbers), [pair.indictment for pair in reuse.pairs], whole)


def _pair(record: CaseRecord, reuse: ReuseResult, pair: ReusePair, number: str) -> None:
    flag = next(f for f in reuse.flags if f.id == pair.flag_id)
    with st.container(border=True):
        heading, status = st.columns([8, 3], vertical_alignment="center")
        heading.markdown(f"**{number}.** {md_escape(flag.message)}")
        with status:
            widgets.badge(flag.status)
        widgets.evidence(record, flag.evidence, key=f"pair-{pair.id}", heading=f"Matched passage {number}")


def _argument(record: CaseRecord, reuse: ReuseResult, check: ArgumentCheck) -> None:
    flag = next((f for f in reuse.flags if f.id == check.flag_id), None)
    with st.container(border=True):
        widgets.badge("addressed_argument" if check.addressed else "unaddressed_argument")
        widgets.evidence(record, [Evidence(role="argument", span=check.argument)], key=f"argument-{check.argument_id}", heading="Defence argument")
        if flag is not None:
            st.markdown(md_escape(flag.message))
        if check.responding:
            st.caption(f"Answered in the reasoning ({check.passages_checked} passages checked):")
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
numbers = {pair.id: str(n) for n, pair in enumerate(reuse.pairs, start=1)}
_metrics(reuse)
_side_by_side(record, reuse, numbers)
st.subheader("Matched passages")
if not reuse.pairs:
    st.caption("No passage of the reasoning matches the indictment.")
for pair in reuse.pairs:
    _pair(record, reuse, pair, numbers[pair.id])
st.subheader("Defence arguments from the notes")
if not reuse.arguments:
    st.caption("No defence arguments were found in the notes.")
for check in reuse.arguments:
    _argument(record, reuse, check)
