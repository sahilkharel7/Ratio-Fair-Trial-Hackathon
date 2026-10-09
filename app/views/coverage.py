"""Rights coverage: ICCPR Art. 14(3)(a) to (g), read against the monitoring notes."""

from __future__ import annotations

import streamlit as st

from ratio.display import guarantee_badge, md_escape
from ratio.messages import render
from ratio.results import GuaranteeAssessment
from ratio_ui import session, widgets
from ratio_ui.session import LoadedCase

SELECTED = "ratio_coverage_item"


def _grid(assessments: tuple[GuaranteeAssessment, ...]) -> None:
    for assessment in assessments:
        with st.container(border=True):
            st.markdown(f"**{md_escape(assessment.provision)}**  \n{md_escape(assessment.name)}")
            # status and button at their natural width: fixed columns cut them to "O…" on a laptop screen
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                widgets.badge(guarantee_badge(assessment))
                if st.button("Open", key=f"open-{assessment.rubric_id}"):
                    st.session_state[SELECTED] = assessment.rubric_id


def _parts(assessment: GuaranteeAssessment) -> None:
    rows = [
        {
            "Part": part.label,
            "Status": widgets.label(part.status),
            "Required": "yes" if part.required else "no",
            "Hearings without evidence": ", ".join([h.isoformat() for h in part.hearings_missing] + list(part.notes_missing)),
        }
        for part in assessment.parts
    ]
    st.dataframe(rows, hide_index=True)


def _uncovered(record, part) -> str:
    gaps = [f"hearing of {h.isoformat()}" for h in part.hearings_missing]
    gaps += [f"{record.document(doc_id).title} (no hearing date)" for doc_id in part.notes_missing]
    return f"{part.label} ({'; '.join(gaps)})" if gaps else part.label


def _details(loaded: LoadedCase, assessment: GuaranteeAssessment) -> None:
    record, absence = loaded.record, loaded.analysis.absence
    messages = session.config().messages
    st.subheader(f"{assessment.provision}: {assessment.name}")
    with st.container(horizontal=True):
        widgets.badge(guarantee_badge(assessment))
        widgets.badge(assessment.review_status)
    st.caption(md_escape(assessment.citation))
    flag = next((f for f in absence.flags if f.id == assessment.flag_id), None)
    if flag is not None:
        st.markdown(md_escape(flag.message))
        widgets.evidence(record, flag.evidence, key=f"flag-{assessment.rubric_id}", heading=flag.standard_label)
        widgets.model_note(flag.model_note)
    elif assessment.follow_up is not None:
        follow_up = assessment.follow_up
        text = render(messages, "absence_follow_up", provision=assessment.provision, name=assessment.name, question=follow_up.question)
        st.info(md_escape(text))
        uncovered = [_uncovered(loaded.record, part) for part in assessment.parts if part.required and part.status == "no_evidence"]
        if assessment.unlabelled_notes:
            uncovered.append(f"{assessment.unlabelled_notes} shortlisted notes were not labelled by the model")
        if uncovered:
            st.markdown("Not covered by the notes:\n" + "\n".join(f"- {md_escape(item)}" for item in uncovered))
        if follow_up.context:
            st.caption("Possibly relevant, not counted as evidence:")
            widgets.evidence(record, follow_up.context, key=f"context-{assessment.rubric_id}", heading="Context")
    _parts(assessment)


loaded = widgets.require_case()
absence = loaded.analysis.absence
widgets.header(loaded, "Rights coverage", "ICCPR Art. 14(3)(a) to (g) against the monitoring notes")
st.markdown(
    "Each guarantee is marked **evidence of compliance**, **evidence of violation**, or **no evidence**, "
    "which becomes a follow-up question for the monitor rather than a finding. Open a guarantee to see the notes it rests on."
)
if absence is None or not absence.assessments:
    st.info("This case has no monitoring notes to read.")
    st.stop()
grid_column, detail_column = st.columns([5, 7], gap="large")
with grid_column:
    _grid(absence.assessments)
selected_id = st.session_state.get(SELECTED, absence.assessments[0].rubric_id)
selected = next((a for a in absence.assessments if a.rubric_id == selected_id), absence.assessments[0])
with detail_column, st.container(border=True):
    _details(loaded, selected)
