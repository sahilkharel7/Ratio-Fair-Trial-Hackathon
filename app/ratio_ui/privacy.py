"""Startup check (hard rule 1): the app refuses to run unless the privacy settings of
.streamlit/config.toml are in effect: no usage statistics, served on loopback only, and no
error messages that link to outside websites."""

from __future__ import annotations

import streamlit as st

from ratio.netguard import is_loopback_host


def problems() -> list[str]:
    found = []
    if st.get_option("browser.gatherUsageStats") is not False:
        found.append("browser.gatherUsageStats must be false (no usage statistics)")
    address = st.get_option("server.address")
    if not address or not is_loopback_host(str(address)):
        found.append("server.address must be 127.0.0.1 (served to this computer only)")
    if st.get_option("client.showErrorLinks") is not False:
        found.append("client.showErrorLinks must be false (no links to outside websites in error messages)")
    return found
