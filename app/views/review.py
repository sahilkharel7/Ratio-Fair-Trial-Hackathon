"""Review: the reviewing lawyer accepts, rewords or rejects each finding with a reason, and records
issues Ratio did not flag. Decisions are saved on this computer (append-only, so the history can be
audited), appear in the report draft, and are measured by `python -m ratio feedback`."""

from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

from ratio import feedback
from ratio.display import md_escape
from ratio.feedback import MODULES, MissedIssue, Review
from ratio.jurisprudence import for_finding
from ratio.messages import contains_blocked_term, render
from ratio.schema import Flag
from ratio_ui import session, widgets
from ratio_ui.session import LoadedCase

REVIEWER_KEY = "ratio_reviewer"
FILTER_KEY = "review_filter"
FILTERS = ("All", "Not reviewed", "Reviewed")
CHOICES = {"accepted": "Accept", "edited": "Reword", "rejected": "Reject"}


def _reviewer() -> str | None:
    name = (st.session_state.get(REVIEWER_KEY) or "").strip()
    return name or None


def _problem(exc: ValidationError) -> str:
    return "; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors())


def _decide(case_id: str, flag: Flag, decision: str, note: str, wording: str | None) -> None:
    if decision == "edited" and wording and contains_blocked_term(wording, session.config().messages.block_list):
        st.error("Not saved: the wording describes a person's character or motives. Describe the pattern or the facts instead.")
        return
    try:
        review = Review(
            case_id=case_id, flag_id=flag.id, decision=decision, note=note.strip(),
            edited_message=wording.strip() if decision == "edited" and wording else None, reviewer=_reviewer(), flag=flag,
        )  # fmt: skip
    except ValidationError as exc:
        st.error(f"Not saved: {md_escape(_problem(exc))}.")
        return
    session.store().add_review(review)
    st.rerun()


def _when(review: Review) -> str:
    by = f" by {review.reviewer}" if review.reviewer else ""
    return f"{widgets.label('review_' + review.decision) if review.decision != 'reopened' else 'Reopened'} on {review.created_at[:10]}{by}"


def _finding(loaded: LoadedCase, flag: Flag, review: Review | None, history: list[Review]) -> None:
    case_id = loaded.record.case_id
    with st.container(border=True):
        st.markdown(f"**{md_escape(flag.standard_label)}**")
        with st.container(horizontal=True):
            widgets.badge(flag.status)
            widgets.badge(f"review_{review.decision}" if review else "review_pending")
        st.markdown(md_escape(feedback.message(flag, review)))
        if review is not None:
            reason = f" Reason: {review.note}" if review.note else ""
            st.caption(md_escape(f"{_when(review)}.{reason}"))
            if review.decision == "edited":
                st.caption(md_escape(f"{session.config().messages.notes['report_ratio_wording']}: {flag.message}"))
        # the first source on the card, so findings with the same wording (matched passages) stay distinct
        widgets.evidence(loaded.record, flag.evidence[:1], key=f"review-src-{flag.id}", heading=flag.standard_label)
        if len(flag.evidence) > 1:
            with st.expander(f"More sources ({len(flag.evidence) - 1})"):
                widgets.evidence(loaded.record, flag.evidence[1:], key=f"review-more-{flag.id}", heading=flag.standard_label)
        widgets.model_note(flag.model_note)
        widgets.jurisprudence(for_finding(flag, session.config().jurisprudence))
        if loaded.analysis.steelman is not None:
            widgets.state_reply(loaded.record, loaded.analysis.steelman.for_flag(flag.id), key=f"review-state-{flag.id}")
        with st.form(key=f"review-{flag.id}", border=False):
            choices = list(CHOICES)
            index = choices.index(review.decision) if review else 0
            decision = st.radio("Decision", choices, index=index, format_func=CHOICES.get, horizontal=True, key=f"decision-{flag.id}")
            note = st.text_area("Reason (needed to reword or reject)", value=review.note if review else "", key=f"note-{flag.id}", height=68)
            wording = st.text_area(
                "Your wording (used when you reword)", value=feedback.message(flag, review), key=f"wording-{flag.id}", height=68
            )
            if st.form_submit_button("Save decision", type="primary", key=f"save-{flag.id}"):
                _decide(case_id, flag, decision, note, wording)
        if review is not None and st.button("Reopen", key=f"reopen-{flag.id}", help="Withdraw the decision; the history keeps it"):
            _decide(case_id, flag, "reopened", "", None)
        if len(history) > 1:
            with st.expander(f"History ({len(history)} decisions)"):
                for past in history:
                    reason = f": {past.note}" if past.note else ""
                    st.markdown(f"- {md_escape(_when(past) + reason)}")


def _missed(loaded: LoadedCase) -> None:
    labels = {module: widgets.label(f"module_{module}") for module in MODULES}
    case_id = loaded.record.case_id
    st.subheader(session.config().messages.notes["report_missed"])
    for issue in session.store().missed_issues(case_id):
        text, button = st.columns([10, 2], vertical_alignment="center")
        by = f", {issue.reviewer}" if issue.reviewer else ""
        text.markdown(md_escape(f"{labels[issue.module]}, {issue.standard}: {issue.note} ({issue.created_at[:10]}{by})"))
        if button.button("Withdraw", key=f"withdraw-{issue.id}"):
            session.store().withdraw_missed_issue(issue.id)
            st.rerun()
    with st.form(key="missed-issue", clear_on_submit=True):
        module = st.selectbox("Module that should have flagged it", list(MODULES), format_func=labels.get, key="missed-module")
        standard = st.text_input("Standard or guarantee", placeholder="e.g. ICCPR Art. 14(3)(f)", key="missed-standard")
        note = st.text_area("What the record shows", key="missed-note", height=68)
        if st.form_submit_button("Record missed issue", key="missed-save"):
            try:
                issue = MissedIssue(case_id=case_id, module=module, standard=standard.strip(), note=note.strip(), reviewer=_reviewer())
            except ValidationError as exc:
                st.error(f"Not saved: {md_escape(_problem(exc))}.")
            else:
                session.store().add_missed_issue(issue)
                st.rerun()


def _stale(stale: list[Review]) -> None:
    messages = session.config().messages
    with st.expander(render(messages, "review_stale", n=len(stale))):
        for review in stale:
            st.markdown(f"- {md_escape(review.flag.standard_label)}: {md_escape(review.flag.message)}  \n  {md_escape(_when(review))}")


loaded = widgets.require_case()
case_id = loaded.record.case_id
widgets.header(loaded, "Review", "the reviewing lawyer's decision on each finding")
messages = session.config().messages
st.markdown(md_escape(messages.notes["review_intro"]))
flags = session.findings(loaded)
reviews = session.store().reviews(case_id)
in_force = feedback.current(reviews)
shown = {flag.id for flag in flags}
decided = [in_force[flag.id] for flag in flags if flag.id in in_force]
missed = session.store().missed_issues(case_id)
columns = st.columns(5)
columns[0].metric("Findings", len(flags))
columns[1].metric("Reviewed", len(decided))
for column, decision, title in zip(columns[2:], ("accepted", "edited", "rejected"), ("Accepted", "Reworded", "Rejected"), strict=True):
    column.metric(title, sum(r.decision == decision for r in decided))
st.text_input("Reviewer (optional, saved with each decision)", key=REVIEWER_KEY)
stale = [review for flag_id, review in in_force.items() if flag_id not in shown]
if stale:
    _stale(stale)
choice = st.segmented_control("Show", FILTERS, default=FILTERS[0], key=FILTER_KEY, label_visibility="collapsed") or FILTERS[0]
for module in MODULES:
    selected = [
        flag for flag in flags
        if flag.module == module and (choice == "All" or (flag.id in in_force) == (choice == "Reviewed"))
    ]  # fmt: skip
    if not selected:
        continue
    st.subheader(widgets.label(f"module_{module}"))
    for flag in selected:
        _finding(loaded, flag, in_force.get(flag.id), [r for r in reviews if r.flag_id == flag.id])
_missed(loaded)
