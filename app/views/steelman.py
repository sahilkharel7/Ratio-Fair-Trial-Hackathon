"""The State's strongest reply: for each finding the State would contest, the arguments the local model
made for the State from the record, each on a reviewed ground and resting on an exact quote, and the
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
    st.info("No finding in this case is one the State would contest, or the case was analysed before this step existed: load it again.")
    st.stop()
flags = {flag.id: flag for flag in loaded.analysis.all_flags()}
argued = sum(bool(reply.arguments) for reply in result.replies)
columns = st.columns(3)
columns[0].metric("Findings contested", len(result.replies))
columns[1].metric("With a reply the record supports", argued)
columns[2].metric("Arguments kept", sum(len(reply.arguments) for reply in result.replies))
st.caption(f"Model: {md_escape(result.model or 'none')}. Arguments that failed a check are dropped and counted under each finding.")
for reply in result.replies:
    flag = flags.get(reply.flag_id)
    if flag is None:
        continue
    with st.container(border=True):
        st.markdown(f"**{md_escape(flag.standard_label)}**")
        widgets.badge(flag.status)
        st.markdown(md_escape(flag.message))
        widgets.state_reply(loaded.record, reply, key=f"page-{reply.flag_id}", expanded=bool(reply.arguments))
        if reply.dropped:
            with st.expander(f"Dropped by the checks ({len(reply.dropped)})"):
                for reason in reply.dropped:
                    st.markdown(f"- {md_escape(reason)}")
