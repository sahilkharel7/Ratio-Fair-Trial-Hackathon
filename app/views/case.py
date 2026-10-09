"""Case page: load the demo case or an uploaded case folder, see what was read, and run one note
through the live local model."""

from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from ratio.display import md_escape, percent
from ratio.embeddings import EmbeddingModelMissing
from ratio.extraction.loader import LoaderError, read_uploaded_files
from ratio.feedback import current
from ratio.llm import LLMError
from ratio.pipeline import run_note_live
from ratio.report import draft_report
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


def _count(number: int, noun: str) -> str:
    """'1 finding', '3 findings': a count on screen reads as plain English."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _demo_card() -> None:
    with st.container(border=True):
        st.subheader("Demo case")
        st.markdown(
            "*Republic of Calderra v. Daro Venn*, a **synthetic** case: four hearing notes, the indictment, "
            "the judgment and three detention orders. The local model's answers were recorded earlier and are "
            "replayed, so the demo works offline, without the model running."
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
            "Choose a folder that holds `case.yaml` and the case documents: the monitoring notes, the indictment "
            "and the judgment, as .txt, .md or .pdf files. Use public or synthetic material only. "
            "The local model reads the folder on this computer; nothing is sent anywhere."
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
        label=f"Rights coverage: {_count(statuses.count('evidence_of_violation'), 'guarantee')} with evidence of violation, "
        f"{statuses.count('evidence_of_compliance')} with evidence of compliance, "
        f"{_count(statuses.count('no_evidence'), 'follow-up question')}",
    )
    st.page_link("views/timeline.py", label=f"Timeline: {_count(len(red), 'interval')} longer than a confirmed benchmark")
    _renewal_link(analysis)
    st.page_link(
        "views/reuse.py",
        label=f"Reasoning reuse: {percent(reuse.score if reuse else None)} of the court's reasoning traceable to the indictment; "
        f"{_count(len(unanswered), 'defence argument')} without a response",
    )
    _judge_link(loaded)
    _review_link(loaded)


def _review_link(loaded: LoadedCase) -> None:
    flags = session.findings(loaded)
    decided = current(session.store().reviews(loaded.record.case_id))
    reviewed = sum(flag.id in decided for flag in flags)
    st.page_link("views/review.py", label=f"Review: {reviewed} of {len(flags)} findings reviewed by a lawyer")


def _renewal_link(analysis) -> None:
    renewal = analysis.renewal
    if renewal is None or not renewal.orders:
        return
    repeated = sum(flag.status == "repeated_grounds" for flag in renewal.flags)
    st.page_link(
        "views/renewal.py",
        label=f"Detention renewals: {_count(len(renewal.orders), 'detention order')}, {repeated} with grounds repeating "
        f"earlier orders, {_count(len(renewal.gaps), 'gap')} between orders",
    )


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


def _export(loaded: LoadedCase) -> None:
    """The findings as a Markdown report draft, built here and downloaded from this computer's own server."""
    cases, case_id = session.store(), loaded.record.case_id
    history = [record for record in cases.all_records() if record.case_id != case_id]
    report = draft_report(
        loaded.record, loaded.analysis, session.judge_report_or_none(), session.config(),
        history=history, reviews=cases.reviews(case_id), missed=cases.missed_issues(case_id),
    )  # fmt: skip
    st.download_button(
        "Download report draft (.md)", data=report, file_name=f"{loaded.record.case_id}-report-draft.md",
        mime="text/markdown", key="export_report",
    )  # fmt: skip
    st.caption("An editable text file: every finding, with the passage it rests on quoted word for word. It is saved on this computer.")


def _summary(loaded: LoadedCase) -> None:
    record, analysis = loaded.record, loaded.analysis
    st.divider()
    st.header(md_escape(record.meta.title))
    judge = f" · Presiding judge: {md_escape(record.meta.presiding_judge)}" if record.meta.presiding_judge else ""
    st.caption(f"{md_escape(record.meta.court)} · {md_escape(record.meta.charge_type)}{judge}")
    columns = st.columns(4)
    columns[0].metric("Documents", len(record.documents))
    columns[1].metric("Note sentences", len(record.observations))
    columns[2].metric("Dated events", sum(e.type != "hearing" for e in record.events))
    columns[3].metric("Party arguments", len(record.arguments))
    flags = analysis.all_flags()
    mode = "answers replayed from a recording" if analysis.llm_mode == "replay" else "run live on this computer"
    st.subheader("Findings")
    st.markdown(
        f"**{_count(len(flags), 'finding')}**, each linked to the exact passage it rests on. "
        f"Findings whose passage could not be found in the record are dropped: {analysis.dropped_flags} in this case. "
        f"Local model: {md_escape(analysis.llm_model or 'none')} ({mode})."
    )
    _findings(loaded)
    _export(loaded)
    st.subheader("Documents in this case")
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
    st.header("Live model check")
    st.markdown(
        "Run one monitoring note through the local model now, without the recording, and compare its answer "
        "with the recorded one."
    )
    doc_id = st.selectbox("Monitoring note", list(notes), format_func=notes.get, key="live_doc")
    if st.button("Run this note live", key="live_run"):
        try:
            with st.spinner("The local model is reading the note"):
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
    "Ratio is an offline reading aid for one trial-monitoring record: the monitoring notes, the indictment, "
    "the judgment and the detention orders. Every finding links to the exact passage it rests on. "
    "The legal evaluation stays with the reviewing lawyer, and nothing leaves this computer."
)
st.header("Load a case")
demo_column, upload_column = st.columns(2, gap="large")
with demo_column:
    _demo_card()
with upload_column:
    _upload_card()
if loaded is not None and loaded.analysis is not None:
    _summary(loaded)
    _live_note(loaded)
