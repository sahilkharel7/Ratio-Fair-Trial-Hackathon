"""Ratio: offline analysis of trial-monitoring records. Run it from the repository root, so the
privacy settings in .streamlit/config.toml apply:

    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

from ratio import netguard  # noqa: E402
from ratio_ui import privacy, session, style  # noqa: E402

st.set_page_config(page_title="Ratio", page_icon="⚖️", layout="wide")  # an emoji is drawn locally; a Material icon would be fetched online
netguard.install()  # any connection that is not to this computer now fails loudly

problems = privacy.problems()
if problems:
    st.error(
        "Ratio will not start because these privacy settings are not in effect. Run "
        "`streamlit run app/main.py` from the repository root so that .streamlit/config.toml applies.\n\n"
        + "\n".join(f"- {problem}" for problem in problems)
    )
    st.stop()

style.inject()
# In the reviewer's order: the case, the findings in it, the context that tests them, and the
# reviewer's own decisions. A flat list, because Streamlit's sectioned menu is not a valid list for
# screen readers.
pages = [
    st.Page("views/case_collection.py", title="Case collection", default=True),
    st.Page("views/focus.py", title="Case review"),
    st.Page("views/case.py", title="Full analysis record"),
    st.Page("views/coverage.py", title="Rights coverage"),
    st.Page("views/timeline.py", title="Timeline"),
    st.Page("views/renewal.py", title="Detention renewals"),
    st.Page("views/reuse.py", title="Reasoning reuse"),
    st.Page("views/judges.py", title="Judge profile"),
    st.Page("views/jurisprudence.py", title="Jurisprudence"),
    st.Page("views/similar.py", title="Similar cases"),
    st.Page("views/library.py", title="Precedent library"),
    st.Page("views/steelman.py", title="State's reply"),
    st.Page("views/review.py", title="Review"),
]
page = st.navigation(pages, position="hidden")
with st.sidebar:
    st.page_link("views/case_collection.py",label="Case collection")
    st.page_link("views/focus.py",label="Case review · Presumption of innocence")
    with st.expander("Supporting analysis"):
        for name,label in (("case","Full analysis record"),("coverage","Rights coverage"),("timeline","Timeline"),("renewal","Detention renewals"),("reuse","Reasoning comparison"),("judges","Judicial history"),("similar","Similar cases"),("library","Precedent library"),("jurisprudence","Jurisprudence"),("steelman","State’s reply"),("review","Full finding review")):
            st.page_link(f"views/{name}.py",label=label)
    style.offline_note(session.config().messages.notes["offline_note"])
page.run()
