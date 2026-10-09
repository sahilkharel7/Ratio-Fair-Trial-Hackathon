"""Case page: load the demo case or an uploaded case folder, see what was read, and run one note
through the live local model."""

from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from ratio.display import md_escape, percent
from ratio.embeddings import EmbeddingModelMissing
from ratio.extraction.loader import LoaderError, read_uploaded_files
from ratio.llm import LLMError
from ratio.pipeline import run_note_live
from ratio.schema import Evidence
from ratio_ui import session, style, widgets
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
    with st.container(border=True):
        st.subheader("Demo case")
        st.markdown(
            "*Republic of Calderra v. Daro Venn*, a **synthetic** case: four hearing notes, the indictment "
            "and the judgment. The local model's answers are replayed from a recorded cache, so this works "
            "with Ollama stopped and the network off."
        )
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
    with st.container(border=True):
        st.subheader("Your case folder")
        st.markdown(
            "A folder with `case.yaml`, the monitoring notes, the indictment and the judgment (.txt, .md or .pdf). "
            "Public or synthetic material only. The local model reads it; nothing leaves this computer."
        )
        files = st.file_uploader(
            "Case folder", accept_multiple_files="directory", type=["yaml", "yml", "txt", "md", "pdf"],
            label_visibility="collapsed", key="upload",
        )  # fmt: skip
        if files:
            _upload_form({file.name: file.getvalue() for file in files})


def _findings(loaded: LoadedCase) -> None:
    analysis = loaded.analysis
    absence, clock, reuse = analysis.absence, analysis.clock, analysis.reuse
    statuses = [a.status for a in absence.assessments] if absence else []
    red = [i for i in clock.intervals if i.status == "exceeds_benchmark"] if clock else []
    unanswered = [c for c in reuse.arguments if not c.addressed] if reuse else []
    st.page_link(
        "views/coverage.py",
        label=f"Rights coverage: {statuses.count('evidence_of_violation')} with evidence of violation, "
        f"{statuses.count('evidence_of_compliance')} with evidence of compliance, {statuses.count('no_evidence')} follow-up questions",
    )
    st.page_link("views/timeline.py", label=f"Timeline: {len(red)} interval longer than a confirmed benchmark")
    st.page_link(
        "views/reuse.py",
        label=f"Reasoning reuse: {percent(reuse.score if reuse else None)} of the reasoning traceable to the indictment; "
        f"{len(unanswered)} defence argument without a response",
    )
    _judge_link(loaded)


def _judge_link(loaded: LoadedCase) -> None:
    """The judge of this case, if the stored cases hold coded rulings for them (the demo brings a history)."""
    try:
        report = session.judge_report()
    except LoaderError:
        return  # the judge page explains the problem
    profile = next((p for p in report.profiles if loaded.record.case_id in p.case_ids), None)
    if profile is not None:
        patterns = len(profile.flags)
        found = "1 pattern that warrants review" if patterns == 1 else f"{patterns} patterns that warrant review"
        cases = "1 case" if len(profile.case_ids) == 1 else f"{len(profile.case_ids)} cases"
        compared = "1 indicator" if profile.k_compared == 1 else f"{profile.k_compared} indicators"
        st.page_link("views/judges.py", label=f"Judge profile: {md_escape(profile.display_name)}, {found} ({cases}, {compared} compared)")


def _summary(loaded: LoadedCase) -> None:
    record, analysis = loaded.record, loaded.analysis
    st.divider()
    st.subheader(md_escape(record.meta.title))
    judge = f" · Presiding judge: {md_escape(record.meta.presiding_judge)}" if record.meta.presiding_judge else ""
    st.caption(f"{md_escape(record.meta.court)} · {md_escape(record.meta.charge_type)}{judge}")
    columns = st.columns(4)
    columns[0].metric("Documents", len(record.documents))
    columns[1].metric("Note sentences", len(record.observations))
    columns[2].metric("Dated events", sum(e.type != "hearing" for e in record.events))
    columns[3].metric("Party arguments", len(record.arguments))
    flags = analysis.all_flags()
    mode = "answers replayed from the recorded cache" if analysis.llm_mode == "replay" else "run live on this computer"
    st.markdown(
        f"**{len(flags)} findings**, each linked to the exact text it rests on. "
        f"{analysis.dropped_flags} dropped because their source text could not be found. "
        f"Model: {md_escape(analysis.llm_model or 'none')} ({mode})."
    )
    _findings(loaded)
    rows = [
        {"Document": d.title, "Type": d.type.replace("_", " "), "Hearing date": d.date.isoformat() if d.date else "", "Characters": len(d.text)}
        for d in record.documents
    ]  # only monitoring notes carry a hearing date; the others show an empty cell, not "None"
    st.dataframe(rows, hide_index=True)


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


loaded = session.current_case()
if loaded is not None and loaded.record.meta.synthetic:
    style.banner(session.config().messages.notes["synthetic_banner"])
st.title("Ratio")
st.markdown(
    "Offline analysis of trial-monitoring records for a reviewing lawyer. Every finding links to the exact "
    "text it rests on, and nothing leaves this computer."
)
demo_column, upload_column = st.columns(2, gap="large")
with demo_column:
    _demo_card()
with upload_column:
    _upload_card()
if loaded is not None and loaded.analysis is not None:
    _summary(loaded)
    _live_note(loaded)
