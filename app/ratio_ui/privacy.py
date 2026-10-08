"""Startup check (hard rule 1): the app refuses to run unless the privacy settings of
.streamlit/config.toml are in effect: no usage statistics, served on loopback only, no error
messages that link to outside websites, and no theme value (such as a font) loaded from elsewhere."""

from __future__ import annotations

import re

import streamlit as st
from streamlit import config as st_config

from ratio.netguard import is_loopback_host

_REMOTE = re.compile(r"(?:^|[^\w])//|url\(", re.IGNORECASE)  # http://, https://, //host, url(...)


def _remote(value: object) -> bool:
    if isinstance(value, str):
        return bool(_REMOTE.search(value))
    if isinstance(value, dict):
        return any(_remote(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_remote(item) for item in value)
    return False


def problems() -> list[str]:
    found = []
    if st.get_option("browser.gatherUsageStats") is not False:
        found.append("browser.gatherUsageStats must be false (no usage statistics)")
    address = st.get_option("server.address")
    if not address or not is_loopback_host(str(address)):
        found.append("server.address must be 127.0.0.1 (served to this computer only)")
    if st.get_option("client.showErrorLinks") is not False:
        found.append("client.showErrorLinks must be false (no links to outside websites in error messages)")
    for name in sorted(key for key in st_config.get_config_options() if key.startswith("theme.")):
        if _remote(st.get_option(name)):
            where = st_config.get_where_defined(name)
            found.append(f"{name} must not load anything from outside this computer (it is set in {where}: remove it there)")
    return found
