"""One-violation review: charge, stakes, sources, precedent reading, human next step."""

from __future__ import annotations

import streamlit as st

from ratio.display import md_escape
from ratio.library import LibraryStore
from ratio.outcomes import registry
from ratio.workspace import focus_report
from ratio_ui import session, style, viewer, widgets

loaded = session.current_case()
if loaded is None:
    st.info("Open a case from the collection.")
    st.page_link("views/case_collection.py", label="Open case collection")
    st.stop()
record = loaded.record
library = LibraryStore(session.store())
focus = library.focus(record.case_id)
st.page_link("views/case_collection.py", label="← Case collection")
st.title(md_escape(focus.defendant))
if record.meta.synthetic:
    style.banner(session.config().messages.notes["synthetic_banner"])
st.caption(md_escape(f"{record.meta.court} · {record.case_id}"))
st.header("Charge and what is at stake")
st.markdown(f"**Charge:** {md_escape(focus.charge)}")
if focus.charge_span:
    widgets.source_button(
        focus.charge_span,
        record,
        key="charge-source",
        heading="Charge as stated in the record",
    )
else:
    st.caption("Charge entered in the case metadata; verify it against the record.")
columns = st.columns(3)
for column, stage, title in zip(
    columns,
    ("requested", "imposed", "statutory"),
    ("Prosecution seeks", "Sentence imposed", "Quoted statutory penalty"),
    strict=True,
):
    items = [p for p in focus.penalties if p.stage == stage]
    item = next(
        (p for p in items if p.max_months is not None or p.kind != "imprisonment"),
        items[0] if items else None,
    )
    with column:
        st.subheader(title)
        st.markdown(
            f"**{md_escape(item.label) if item else 'Not stated in the record'}**"
        )
        if item:
            widgets.source_button(
                item.span, record, key="penalty-" + stage, heading=title
            )
st.header("Presumption of innocence")
st.caption(focus.standard)
st.caption(focus.screening_note)
decisions = library.decisions(record.case_id)
latest = {}
for decision in decisions:
    if decision["decision"] == "reopened":
        latest.pop(decision["prompt_id"], None)
    else:
        latest[decision["prompt_id"]] = decision
if not focus.prompts:
    st.info(
        "No explicit screening passage was found. This does not establish compliance. Verify who carried the burden of proof and whether public statements or courtroom presentation prejudged guilt."
    )
for prompt in focus.prompts:
    with style.panel("focus-" + prompt.id):
        st.subheader(prompt.title)
        st.markdown(md_escape(prompt.question))
        widgets.source_button(
            prompt.span, record, key="prompt-" + prompt.id, heading=prompt.title
        )
        st.markdown(md_escape("“" + prompt.span.text + "”"))
        saved = latest.get(prompt.id)
        if saved:
            st.caption(
                md_escape(
                    f"Reviewer assessment: {saved['decision']}. {saved['reason']}"
                )
            )
        with st.form("assessment-" + prompt.id):
            choice = st.selectbox(
                "Your assessment",
                ["follow_up", "supported", "not_supported"],
                format_func=lambda v: {
                    "follow_up": "Need more evidence",
                    "supported": "Concern supported by the record",
                    "not_supported": "Concern not supported",
                }[v],
            )
            reason = st.text_area("Reason and next step")
            if st.form_submit_button("Save assessment"):
                try:
                    library.save_decision(
                        {
                            "case_id": record.case_id,
                            "prompt_id": prompt.id,
                            "decision": choice,
                            "reason": reason,
                            "source_snapshot": prompt.model_dump(mode="json"),
                        }
                    )
                except ValueError as exc:
                    st.error(md_escape(str(exc)))
                else:
                    st.rerun()
        if saved and st.button("Reopen assessment", key="reopen-" + prompt.id):
            library.save_decision(
                {
                    "case_id": record.case_id,
                    "prompt_id": prompt.id,
                    "decision": "reopened",
                    "reason": "",
                    "source_snapshot": prompt.model_dump(mode="json"),
                }
            )
            st.rerun()
st.header("International review context")
st.caption(
    "Read claim-specific outcomes. A favourable finding is not a prediction or proof of release or acquittal."
)
patterns = {p.pattern for p in focus.prompts}
examples = registry()["records"]
if patterns:
    examples = [e for e in examples if e["pattern"] in patterns]
for item in examples:
    with st.expander(f"{item['title']} · {item['outcome'].replace('_', ' ')}"):
        st.markdown(md_escape(item["summary"]))
        st.caption(md_escape(item["citation"]))
        st.markdown(md_escape("“" + item["excerpt"] + "”"))
        st.link_button("Open official source", item["source_url"])
st.header("Next step")
with st.form("working-note"):
    note = st.text_area(
        "Case working note",
        placeholder="Missing evidence, a question for the monitor, or the action to take next",
    )
    if st.form_submit_button("Save note to SQLite"):
        try:
            library.add_note(record.case_id, note)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.rerun()
for note in library.notes(record.case_id):
    st.caption(md_escape(f"{note['created_at']}: {note['note']}"))
st.download_button(
    "Download focused case worksheet (.md)",
    focus_report(library, record.case_id),
    file_name=record.case_id + "-case-review.md",
    mime="text/markdown",
)
with st.expander("Original documents and supporting analysis"):
    st.page_link(
        "views/library.py",
        label="Search or add reference collections in the Precedent library",
    )
    titles = {d.id: d.title for d in record.documents}
    id = st.selectbox("Source document", list(titles), format_func=titles.get)
    if st.button("Read original source text"):
        viewer.show_document(record, id)
    if loaded.analysis is not None:
        st.page_link("views/case.py", label="Open full analysis record")
        st.page_link("views/similar.py", label="Open Sahil’s Similar cases comparison")
