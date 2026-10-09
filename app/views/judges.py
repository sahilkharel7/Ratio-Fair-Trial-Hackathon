"""Judge profile: rates from the coded rulings of every stored case, each against the baseline of the
other judges of the same court for the same charge type, with n and intervals. The page never
characterises a judge: a clear difference is a "pattern that warrants review", every rate states its
sample size, and the selection-bias note is fixed at the top."""

from __future__ import annotations

import html
import statistics

import streamlit as st

from ratio.display import md_escape, percent
from ratio.extraction.loader import LoaderError
from ratio.history import decisions_file
from ratio.messages import render
from ratio.paths import REPO_ROOT
from ratio.results import AliasCandidate, DataNote, Descriptive, Indicator, JudgeProfile, JudgeReport, RateEstimate
from ratio.schema import Evidence
from ratio_ui import session, style, widgets

JUDGE_KEY = "ratio_judge_id"
KIND_KEY = "ratio_judge_data"
KINDS = {True: "Synthetic data", False: "Public data"}  # never on one view: hard rule 4


def _notes() -> dict[str, str]:
    return session.config().messages.notes


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _rate(estimate: RateEstimate) -> str:
    return (
        f"{estimate.k} of {estimate.n} cases, {percent(estimate.rate)} "
        f"({estimate.confidence:.0%} interval {percent(estimate.ci_low)} to {percent(estimate.ci_high)})"
    )


def _points(value: float) -> str:
    return f"{round(100 * value):+d}"


def _row(evidence: Evidence, where: str, key: str, heading: str) -> None:
    text_column, button_column = st.columns([10, 2], vertical_alignment="center")
    quote = f'<div class="ratio-quote"><span class="ratio-where">{html.escape(where)}</span>“{html.escape(evidence.span.text)}”</div>'
    text_column.html(quote)
    with button_column:
        widgets.source_button(evidence.span, None, key=key, heading=heading)  # the viewer loads the ruling's own case


def _indicator(profile: JudgeProfile, indicator: Indicator, key: str) -> None:
    with style.panel(f"indicator-{key}"):
        title, status = st.columns([8, 3], vertical_alignment="center")
        title.markdown(f"**{md_escape(indicator.label)}**")
        with status:
            if indicator.pattern:
                widgets.badge("pattern_warrants_review")
            elif not indicator.shown:
                widgets.badge("hidden_indicator")
        if indicator.shown:
            judge_column, baseline_column = st.columns(2)
            judge_column.markdown(f"**This judge:** {_rate(indicator.judge)}")
            others = "1 other judge" if indicator.baseline_judges == 1 else f"{indicator.baseline_judges} other judges"
            baseline_column.markdown(f"**Baseline, {others}:** {_rate(indicator.baseline)}")
            low, high = indicator.difference_ci
            st.caption(
                f"Difference from the baseline: {_points(low)} to {_points(high)} percentage points "
                f"({indicator.difference_confidence:.1%} interval, for {profile.k_compared} indicators compared)."
            )
        st.markdown(md_escape(indicator.message))
        if indicator.outcomes:
            label = f"The {len(indicator.outcomes)} cases counted for this judge, each with its coded ruling (yes: counted in the {indicator.judge.k})"
            with st.expander(label):
                for number, outcome in enumerate(indicator.outcomes):
                    where = f"{outcome.title} · {'yes' if outcome.counted else 'no'}"
                    _row(outcome.ruling, where, f"{key}-{number}", f"{outcome.title}: coded ruling")


def _descriptive(item: Descriptive, titles: dict[str, str], key: str) -> None:
    with style.panel(f"descriptive-{key}"):
        title, status = st.columns([8, 3], vertical_alignment="center")
        title.markdown(f"**{md_escape(item.label)}**")
        if not item.shown:
            with status:
                widgets.badge("hidden_indicator")
        else:
            values = sorted(item.values)
            st.markdown(f"{_plural(item.cases, 'case')}: median {statistics.median(values):g}, range {values[0]:g} to {values[-1]:g}")
        st.caption(md_escape(item.message))
        if item.shown:
            with st.expander(f"The {len(item.values)} values, each with its coded ruling"):
                for number, (value, evidence) in enumerate(zip(item.values, item.evidence, strict=True)):
                    case_title = titles.get(evidence.span.case_id, evidence.span.case_id)
                    _row(evidence, f"{case_title} · {value:g}", f"{key}-{number}", f"{case_title}: coded ruling")


def _manual_note(synthetic: bool) -> str:
    where = decisions_file(synthetic).relative_to(REPO_ROOT).as_posix()
    return render(session.config().messages, "manual_confirmation", file=where)


def _waiting(candidates: tuple[AliasCandidate, ...]) -> None:
    rows = [
        {"Case": c.case_id, "Name as written": c.raw_name, "Court": c.court, "Possible match": c.candidate_name or "none",
         "Similarity": c.score, "Why it waits": c.reason}
        for c in candidates
    ]  # fmt: skip
    st.dataframe(rows, hide_index=True)


def _signals(profile: JudgeProfile) -> None:
    st.subheader(_notes()["not_attributed"])
    if not profile.case_signals:
        st.caption("None of this judge's cases has been analysed by the other modules yet.")
        return
    for signal in profile.case_signals:
        text, button = st.columns([10, 2], vertical_alignment="center")
        text.markdown(
            f"{md_escape(signal.title)}: {_plural(signal.flags, 'finding')}; "
            f"{percent(signal.reuse_score)} of the judgment's reasoning traceable to the indictment."
        )
        if button.button("Open case", key=f"open-{signal.case_id}"):
            session.open_case(signal.case_id)
            st.switch_page("views/case.py")


def _method() -> str:
    cfg = session.config()
    alpha = cfg.settings.judges.alpha
    return render(cfg.messages, "judge_method", confidence=f"{1 - alpha:.0%}", alpha=f"{alpha:g}", minimum=cfg.settings.judges.min_case_count)


def _lists(synthetic: bool, waiting: list[AliasCandidate], notes: list[DataNote]) -> None:
    """Every name of this kind of data that waits for a decision, and the data notes."""
    if waiting:
        with st.expander(f"Manual confirmation list, all courts ({len(waiting)})"):
            st.caption(md_escape(_manual_note(synthetic)))
            _waiting(tuple(waiting))
    if notes:
        with st.expander(f"Data notes ({len(notes)})"):
            for note in notes:
                st.markdown(f"- {md_escape(note.text)}")


def _profile(profile: JudgeProfile) -> None:
    variants = ", ".join(profile.name_variants)
    st.caption(
        f"{md_escape(profile.court)} · {md_escape('; '.join(profile.charge_types))} · {_plural(len(profile.case_ids), 'case')} · "
        f"name as written: {md_escape(variants)}"
    )
    st.markdown(f"Indicators compared on this page: **{profile.k_compared}**. Rates count each case once.")
    with st.expander("How these numbers are computed"):
        st.markdown(md_escape(_method()))
    titles = {summary.case_id: summary.title for summary in session.store().list_cases()}
    for charge_index, charge_type in enumerate(profile.charge_types):
        if len(profile.charge_types) > 1:
            st.subheader(md_escape(charge_type))
        for indicator in (i for i in profile.indicators if i.charge_type == charge_type):
            _indicator(profile, indicator, key=f"{profile.judge_id}-{charge_index}-{indicator.rate_id}")
        for item in (d for d in profile.descriptive if d.charge_type == charge_type):
            _descriptive(item, titles, key=f"{profile.judge_id}-{charge_index}-{item.code}")
    if profile.pending:
        st.subheader("Names waiting for confirmation")
        st.caption(md_escape(_manual_note(profile.synthetic)))
        _waiting(profile.pending)
    _signals(profile)


def _current_case_id() -> str | None:
    loaded = session.current_case()
    return loaded.record.case_id if loaded is not None else None


def _kinds(report: JudgeReport) -> list[bool]:
    """The kinds of data with something to show: a judge, a name waiting for a decision, or a note."""
    found = {p.synthetic for p in report.profiles} | {c.synthetic for c in report.manual_confirmations}
    return sorted(found | {n.synthetic for n in report.notes if n.synthetic is not None}, reverse=True)


def _default_kind(report: JudgeReport, kinds: list[bool]) -> bool:
    """The kind of data of the case being viewed, else synthetic first."""
    loaded = session.current_case()
    return loaded.record.meta.synthetic if loaded is not None and loaded.record.meta.synthetic in kinds else kinds[0]


def _default_judge(profiles: dict[str, JudgeProfile]) -> str:
    """The judge of the case being viewed, else the first by name (the list is alphabetical, never ranked)."""
    case_id = _current_case_id()
    return next((judge_id for judge_id, p in profiles.items() if case_id in p.case_ids), next(iter(profiles)))


try:
    report = session.judge_report()
except LoaderError as exc:
    st.error(f"The judge registry cannot be built: {md_escape(str(exc))}")
    st.stop()
kinds = _kinds(report)
if not kinds:
    st.title("Judge profile")
    st.info("No coded rulings are stored yet. Load the demo case on the Case page: it brings the synthetic history of its court.")
    st.page_link("views/case.py", label="Go to the Case page")
    st.stop()
if len(kinds) == 1:
    st.session_state.pop(KIND_KEY, None)  # no choice to remember: a later case of the other kind opens on its own view
    synthetic = kinds[0]
else:
    if st.session_state.get(KIND_KEY) not in kinds:
        st.session_state[KIND_KEY] = _default_kind(report, kinds)
    synthetic = st.session_state[KIND_KEY]
profiles = {p.judge_id: p for p in report.profiles if p.synthetic == synthetic}
waiting = [c for c in report.manual_confirmations if c.synthetic == synthetic]
notes = [note for note in report.notes if note.synthetic in (None, synthetic)]
if synthetic:
    style.banner(_notes()["synthetic_banner"])
style.eyebrow("Cross-case analysis")
st.title("Judge profile")
st.warning(_notes()["selection_bias"])
if len(kinds) > 1:
    st.segmented_control("Data", kinds, format_func=KINDS.get, key=KIND_KEY, help="Synthetic and public cases are never shown or compared together.")
if report.dropped_flags:
    st.warning(f"{_plural(report.dropped_flags, 'pattern')} not shown: the source text of a coded ruling could not be found.")
if profiles:
    if st.session_state.get(JUDGE_KEY) not in profiles:
        st.session_state[JUDGE_KEY] = _default_judge(profiles)
    st.selectbox("Judge", list(profiles), format_func=lambda judge_id: f"{profiles[judge_id].display_name} · {profiles[judge_id].court}", key=JUDGE_KEY)
    _profile(profiles[st.session_state[JUDGE_KEY]])
else:
    st.info("No judge of this kind of data is registered yet: every name below waits for a person's decision.")
_lists(synthetic, waiting, notes)
