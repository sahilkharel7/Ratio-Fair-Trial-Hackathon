"""The State's strongest reply: for each finding the State would contest, the arguments the local model
made for the State from the record, each on a drafted ground (needs legal review) and an exact quote, and the
grounds the record did not support. Model-generated and unverified; never a finding."""

from __future__ import annotations

import streamlit as st

from ratio.display import md_escape
from ratio_ui import session, widgets

loaded = widgets.require_case()
notes = session.config().messages.notes
widgets.header(loaded, "The State's strongest reply", "testing each finding against the best case for the State")
st.markdown(md_escape(notes["steelman_intro"]))
st.caption(md_escape(notes["steelman_caveat"]))
result = loaded.analysis.steelman
if result is None or not result.replies:
    st.info(
        "No finding in this case is one the State would contest. If the case was analysed with an earlier "
        "version of Ratio, load it again on the Case page to add the State's reply."
    )
    st.stop()
flags = {flag.id: flag for flag in loaded.analysis.all_flags()}
argued = sum(bool(reply.arguments) for reply in result.replies)
columns = st.columns(3)
columns[0].metric("Findings the State would contest", len(result.replies))
columns[1].metric("With a reply the record supports", argued)
columns[2].metric("Arguments that passed the checks", sum(len(reply.arguments) for reply in result.replies))
st.caption(
    f"Local model: {md_escape(result.model or 'not recorded')}. Arguments that failed a check are dropped "
    "and listed, with the reason, under their finding."
)
for reply in result.replies:
    flag = flags.get(reply.flag_id)
    if flag is None:
        continue
    with st.container(border=True):
        st.markdown(f"**{md_escape(flag.standard_label)}**")
        widgets.badge(flag.status)
        st.markdown(md_escape(flag.message))
        widgets.evidence(loaded.record, flag.evidence, key=f"steelman-finding-{flag.id}", heading=flag.standard_label)
        widgets.state_reply(loaded.record, reply, key=f"page-{reply.flag_id}", expanded=bool(reply.arguments))
        if reply.dropped:
            with st.expander(f"Arguments dropped by the checks ({len(reply.dropped)})"):
                for reason in reply.dropped:
                    st.markdown(f"- {md_escape(reason)}")
