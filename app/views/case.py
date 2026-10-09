"""Case page: load the demo case or an uploaded case folder, see what was read, and run one note
through the live local model."""

from __future__ import annotations

import html
import re

import streamlit as st
from pydantic import ValidationError

from ratio.display import md_escape, percent
from ratio.embeddings import EmbeddingModelMissing
from ratio.extraction.loader import LoaderError, read_uploaded_files
from ratio.llm import LLMError
from ratio.pipeline import run_note_live
from ratio.schema import Evidence, SourceSpan
from ratio_ui import session, style, viewer, widgets
from ratio_ui.session import LoadedCase

LIVE_KEY = "ratio_live_note"
FAILURES = (LoaderError, LLMError, EmbeddingModelMissing, ValidationError)  # shown as a message, never a traceback


def _run(title: str, action) -> None:
    with st.status(title, expanded=True) as status:
        try:
            action(widgets.progress_bar())
        except FAILURES as exc:
            status.update(label="Stopped", state="error")
            st.error(md_escape(str(exc)))
            return
        status.update(label="Done", state="complete")
    st.session_state.pop(LIVE_KEY, None)
    st.rerun()


def _demo_card() -> None:
    with style.panel("sample-matter"):
        style.eyebrow("Sample matter")
        st.subheader("Republic of Calderra v. Daro Venn")
        st.markdown(
            "Explore a complete **synthetic** trial record: four hearing notes, an indictment, "
            "and a judgment. Review the findings and the exact passages behind them."
        )
        st.caption("Recorded analysis · Works offline · No running model required")
        if st.button("Load the demo case", type="primary", key="load_demo"):
            _run("Loading the demo case", lambda progress: session.load_demo(progress=progress))


def _upload_form(files: dict[str, bytes]) -> None:
    try:
        manifest, documents = read_uploaded_files(files)
    except LoaderError as exc:
        st.error(f"This folder cannot be read: {md_escape(str(exc))}")
        return
    source = f" (source: {md_escape(manifest.source_note)})" if manifest.source_note else ""
    st.markdown(
        f"**{md_escape(manifest.title)}**: {len(manifest.documents)} documents, declared "
        f"**{manifest.data_provenance}**{source}."
    )
    confirmed = st.checkbox(
        "I confirm these documents are synthetic or already public and contain no confidential monitoring material.",
        key="declaration",
    )
    if st.button("Analyze with the local model", type="primary", disabled=not confirmed, key="analyze_upload"):
        _run("Analyzing with the local model", lambda progress: session.analyze_upload(manifest, documents, progress=progress))


def _upload_card() -> None:
    with style.panel("import-matter"):
        style.eyebrow("New matter")
        st.subheader("Import a case record")
        st.markdown(
            "Select a case folder containing `case.yaml` and its supporting documents. "
            "Use public or synthetic material; analysis runs on this computer."
        )
        files = st.file_uploader(
            "Case folder", accept_multiple_files="directory", type=["yaml", "yml", "txt", "md", "pdf"],
            label_visibility="collapsed", key="upload",
        )  # fmt: skip
        if files:
            _upload_form({file.name: file.getvalue() for file in files})


def _workstream(title: str, value: str | int, description: str, page: str, link: str) -> None:
    with style.panel(page.split("/")[-1].split(".")[0]):
        st.html(
            '<div class="ratio-workstream">'
            f'<span class="ratio-workstream-title">{html.escape(title)}</span>'
            f'<span class="ratio-workstream-number">{html.escape(str(value))}</span></div>'
            f'<div class="ratio-workstream-description">{html.escape(description)}</div>'
        )
        st.page_link(page, label=link)


def _findings(loaded: LoadedCase) -> None:
    analysis = loaded.analysis
    absence, clock, reuse = analysis.absence, analysis.clock, analysis.reuse
    statuses = [a.status for a in absence.assessments] if absence else []
    red = [i for i in clock.intervals if i.status == "exceeds_benchmark"] if clock else []
    unanswered = [c for c in reuse.arguments if not c.addressed] if reuse else []
    left, right = st.columns(2, gap="medium")
    with left:
        _workstream(
            "Rights coverage", len(statuses),
            f"{statuses.count('evidence_of_violation')} with evidence of violation · "
            f"{statuses.count('evidence_of_compliance')} with evidence of compliance · "
            f"{statuses.count('no_evidence')} follow-up questions",
            "views/coverage.py", "Review rights coverage →",
        )
        _workstream(
            "Reasoning comparison", percent(reuse.score if reuse else None),
            f"Of the court's reasoning traceable to the indictment. "
            f"{len(unanswered)} defence {'argument' if len(unanswered) == 1 else 'arguments'} without a response.",
            "views/reuse.py", "Compare judgment and indictment →",
        )
    with right:
        _workstream(
            "Procedural timeline", len(red),
            "Intervals longer than a confirmed benchmark. Other intervals are measured for legal review.",
            "views/timeline.py", "Review chronology and benchmarks →",
        )
        with style.panel("judicial-history"):
            _judge_link(loaded)


def _judge_link(loaded: LoadedCase) -> None:
    """The judge of this case, if the stored cases hold coded rulings for them (the demo brings a history)."""
    try:
        report = session.judge_report()
    except LoaderError:
        st.html('<div class="ratio-workstream-title">Judicial history</div>')
        st.caption("The judge registry needs attention. Open judicial history to review the details.")
        st.page_link("views/judges.py", label="Review judicial history →")
        return
    profile = next((p for p in report.profiles if loaded.record.case_id in p.case_ids), None)
    if profile is not None:
        patterns = len(profile.flags)
        found = "1 pattern that warrants review" if patterns == 1 else f"{patterns} patterns that warrant review"
        cases = "1 case" if len(profile.case_ids) == 1 else f"{len(profile.case_ids)} cases"
        st.html(
            '<div class="ratio-workstream"><span class="ratio-workstream-title">Judicial history</span>'
            f'<span class="ratio-workstream-number">{len(profile.case_ids)}</span></div>'
            f'<div class="ratio-workstream-description">{html.escape(profile.display_name)} · '
            f'{html.escape(cases)} · {html.escape(found)}.</div>'
        )
    else:
        st.html('<div class="ratio-workstream-title">Judicial history</div>')
        st.caption("No coded history for this case's judge. Review the registry and available profiles.")
    st.page_link("views/judges.py", label="Review judicial history →")


def _summary(loaded: LoadedCase) -> None:
    record, analysis = loaded.record, loaded.analysis
    st.html(
        '<div class="ratio-matter"><div class="ratio-eyebrow">CURRENT MATTER</div>'
        f'<h2 class="ratio-matter-title">{html.escape(record.meta.title)}</h2>'
        '<div class="ratio-matter-meta">'
        f'<span><strong>Court</strong> &nbsp; {html.escape(record.meta.court)}</span>'
        f'<span><strong>Charge</strong> &nbsp; {html.escape(record.meta.charge_type)}</span>'
        + (f'<span><strong>Presiding judge</strong> &nbsp; {html.escape(record.meta.presiding_judge)}</span>' if record.meta.presiding_judge else "")
        + '</div></div>'
    )
    flags = analysis.all_flags()
    follow_ups = len(analysis.absence.follow_ups) if analysis.absence else 0
    with st.container(key="ratio-metric-grid"):
        columns = st.columns(4)
        with columns[0]:
            style.metric("Source documents", len(record.documents), f"{len(record.observations)} monitoring note sentences")
        with columns[1]:
            style.metric("Findings to review", len(flags), "Each linked to its exact source")
        with columns[2]:
            style.metric("Monitor follow-ups", follow_ups, "Gaps in the monitoring record", "review")
        with columns[3]:
            events = [e for e in analysis.clock.timeline if e.type != "hearing" and e.date is not None] if analysis.clock else []
            style.metric("Procedural events", len(events), "Distinct dated events, excluding hearings")
    style.section("Review workstreams", "Move from the overview to the evidence behind each result.")
    _findings(loaded)
    with st.expander("Analysis record and method"):
        mode = "answers replayed from the recorded cache" if analysis.llm_mode == "replay" else "run live on this computer"
        st.markdown(
            f"**{len(flags)} findings**, each linked to the exact text it rests on. "
            f"{analysis.dropped_flags} dropped because their source text could not be found. "
            f"Model: {md_escape(analysis.llm_model or 'none')} ({mode})."
        )
        st.caption("Findings support legal review. The reviewing lawyer evaluates their legal significance.")
    _documents(loaded)


def _documents(loaded: LoadedCase) -> None:
    record = loaded.record
    style.section("Source documents", "Search the record or open a document to read its full text.")
    query = st.text_input("Search source documents", placeholder="Search titles or exact words in the record…", key="document_search")
    # Literal, case-insensitive matching. Offsets are taken from the original text,
    # so Unicode case conversion cannot change the source span's character positions.
    pattern = re.compile(re.escape(query), re.IGNORECASE) if query.strip() else None
    documents = [d for d in record.documents if pattern is None or pattern.search(d.title) or pattern.search(d.text)]
    if not documents:
        st.info("No source documents match this search. Try a shorter phrase or a document title.")
        return
    if pattern:
        st.caption(f"{len(documents)} of {len(record.documents)} documents match. Search uses exact words, not semantic similarity.")
    rows = [
        {"Document": d.title, "Type": d.type.replace("_", " "), "Hearing date": d.date.isoformat() if d.date else "", "Characters": len(d.text)}
        for d in documents
    ]  # only monitoring notes carry a hearing date; the others show an empty cell, not "None"
    st.dataframe(rows, hide_index=True, width="stretch")
    select, action = st.columns([5, 1], vertical_alignment="bottom")
    with select:
        titles = {d.id: d.title for d in documents}
        doc_id = st.selectbox("Document to read", list(titles), format_func=titles.get, key="read_document")
    with action:
        if st.button("Open document", key="open_document", width="stretch"):
            doc = record.document(doc_id)
            match = pattern.search(doc.text) if pattern else None
            if match:
                span = SourceSpan(doc_id=doc.id, start=match.start(), end=match.end(), text=match.group())
                viewer.show_source(span, record, "Search result")
            else:
                viewer.show_document(record, doc_id)


def _live_note(loaded: LoadedCase) -> None:
    record = loaded.record
    notes = {doc.id: doc.title for doc in record.documents_of_type("monitoring_note")}
    if not notes:
        return
    st.subheader("Live model check")
    st.markdown("Run one note through the local model now, with no cache, and compare its answer with the recorded one.")
    doc_id = st.selectbox("Note", list(notes), format_func=notes.get, key="live_doc")
    if st.button("Run this note live", key="live_run"):
        try:
            with st.spinner("Asking the local model"):
                st.session_state[LIVE_KEY] = run_note_live(record, doc_id, session.config(), model=loaded.analysis.llm_model)
        except LLMError as exc:
            st.warning(f"The local model is not available ({md_escape(str(exc))}). The rest of the demo works without it.")
    live = st.session_state.get(LIVE_KEY)
    if live is not None and live.doc_id in notes:
        same = "the same as" if live.same_as_record else "different from"
        st.success(f"{md_escape(live.model)} answered in {live.seconds:g} s on this computer: {same} the recorded answer.")
        items = [Evidence(role="mention", span=e.span) for e in live.events]
        items += [Evidence(role="argument", span=a.span) for a in live.arguments]
        widgets.evidence(record, items, key="live")


def _intake() -> None:
    demo_column, upload_column = st.columns(2, gap="medium")
    with demo_column:
        _demo_card()
    with upload_column:
        _upload_card()


loaded = session.current_case()
st.title("Case overview")
if loaded is not None and loaded.analysis is not None:
    if loaded.record.meta.synthetic:
        style.banner(session.config().messages.notes["synthetic_banner"])
    _summary(loaded)
    with st.expander("Open or import another matter"):
        _intake()
    with st.expander("Run a live model check"):
        _live_note(loaded)
else:
    st.html('<div class="ratio-intro">A considered review starts with a clear record. '
            'Bring the chronology, fair-trial guarantees, and judicial reasoning into one evidence-led workspace.</div>')
    _intake()
    st.html(
        '<div class="ratio-process">'
        '<div class="ratio-process-item"><strong>01 &nbsp; Assemble the record</strong>Hearing notes, indictment, and judgment in one matter.</div>'
        '<div class="ratio-process-item"><strong>02 &nbsp; Review the findings</strong>Coverage, chronology, and reasoning against cited standards.</div>'
        '<div class="ratio-process-item"><strong>03 &nbsp; Return to the source</strong>Read every relevant passage in its original context.</div></div>'
    )
    st.html('<div class="ratio-review-note">Designed for the reviewing lawyer. '
            'Analysis runs locally; every finding links to the record. Legal conclusions remain yours.</div>')
