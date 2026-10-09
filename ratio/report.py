"""Report draft export: one analysed case as a Markdown document for the reviewing lawyer.

Pure assembly of results that have already passed the provenance checks: no model call and no new
finding. Every finding is numbered (F1, F2, ...) in the order the body cites it, and its exact source
text is quoted in the annex with the document and line it comes from. Reviewers' decisions
(ratio/feedback.py) are shown next to the findings they concern: a reworded finding shows the
reviewer's wording, with Ratio's kept in the annex. All document, model and reviewer text is
escaped (``md``), so a document can never inject markup into the report, and all wording comes from
the reviewed templates in messages.yaml.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from ratio import feedback, jurisprudence
from ratio.config import Benchmark, Benchmarks, Messages, RatioConfig
from ratio.display import duration_text, guarantee_badge, line_number, percent
from ratio.messages import render
from ratio.results import (
    AbsenceResult,
    CaseAnalysis,
    ClockResult,
    Indicator,
    Interval,
    JudgeProfile,
    JudgeReport,
    RateEstimate,
    RenewalResult,
    ReuseResult,
    TimelineEvent,
)
from ratio.schema import CaseRecord, Document, Flag, SourceSpan


# Inline markup (emphasis, code, links, HTML, tables) anywhere, and block markup at the start of a
# line. Lighter than display.md_escape, which also covers Streamlit directives, so the file stays
# readable in a plain editor.
_INLINE = re.compile(r"([\\`*_\[\]<>|~&])")
_LINE_START = re.compile(r"^(\s*)(#|>|[-+=]|\d+[.)])", re.MULTILINE)


def md(text: str) -> str:
    """Document or model text, made safe to embed in Markdown: shown literally."""
    return _LINE_START.sub(r"\1\\\2", _INLINE.sub(r"\\\1", text))


def interval_text(interval: Interval, benchmark: Benchmark, flags: Iterable[Flag], messages: Messages) -> str:
    """The sentence shown for one interval: its flag's message when it has one, else a measurement."""
    names = messages.event_labels
    flag = next((f for f in flags if f.id == interval.flag_id), None)
    if flag is not None:
        return flag.message
    if interval.min_hours is None or interval.max_hours is None:
        return f"{names[benchmark.from_event]} to {names[benchmark.to_event]}: cannot be measured. {interval.note}".strip()
    duration = duration_text(interval.min_hours, interval.max_hours)
    if interval.status == "measured":
        return render(messages, "clock_measured", from_label=names[benchmark.from_event], to_label=names[benchmark.to_event], duration=duration)
    return f"{names[benchmark.from_event]} to {names[benchmark.to_event]}: {duration}, within the {benchmark.threshold_hours:g}-hour benchmark."


def interval_citation(interval: Interval, benchmark: Benchmark, flags: Iterable[Flag], benchmarks: Benchmarks) -> str:
    flag = next((f for f in flags if f.id == interval.flag_id), None)
    if flag is not None and flag.citation:
        return flag.citation
    source = benchmarks.sources[benchmark.citation.instrument]
    return f"{benchmark.provision}; {source.symbol}, para. {benchmark.citation.paras}"


def _quote(text: str) -> str:
    """Exact source text as a Markdown blockquote, escaped so it is shown literally."""
    return "  \n".join(f"> {line}" for line in md(text).split("\n"))  # two spaces: a line break inside the quote


def _date(event: TimelineEvent) -> str:
    if event.date is None:
        return "undated"
    formats = {"datetime": "%Y-%m-%d %H:%M", "month": "%Y-%m", "year": "%Y"}
    return event.date.strftime(formats.get(event.precision, "%Y-%m-%d"))


def _rate(estimate: RateEstimate) -> str:
    return (
        f"{estimate.k} of {estimate.n} cases, {percent(estimate.rate)} "
        f"({estimate.confidence:.0%} interval {percent(estimate.ci_low)} to {percent(estimate.ci_high)})"
    )


class _Report:
    def __init__(
        self, record: CaseRecord, config: RatioConfig, history: Sequence[CaseRecord], reviews: dict[str, feedback.Review]
    ) -> None:
        self.record = record
        self.reviews = reviews
        self.steelman = None
        self.config = config
        self.messages = config.messages
        self.documents: dict[str, Document] = {d.id: d for r in (record, *history) for d in r.documents}
        self.titles: dict[str, str] = {r.case_id: r.meta.title for r in history}
        self.numbered: dict[str, tuple[str, Flag]] = {}  # flag id -> (F-number, flag), in citation order
        self.lines: list[str] = []

    def label(self, key: str) -> str:
        return self.messages.label(key)

    def note(self, key: str) -> str:
        return self.messages.notes[key]

    def ref(self, flag: Flag | None) -> str:
        if flag is None:
            return ""
        if flag.id not in self.numbered:
            self.numbered[flag.id] = (f"F{len(self.numbered) + 1}", flag)
        review = self.reviews.get(flag.id)
        decision = f" *({md(self.label('review_' + review.decision))})*" if review is not None else ""
        return f" **[{self.numbered[flag.id][0]}]**{decision}"

    def message(self, flag: Flag) -> str:
        return feedback.message(flag, self.reviews.get(flag.id))

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def where(self, span: SourceSpan) -> str:
        document = self.documents.get(span.doc_id)
        if document is None:
            return md(span.doc_id)
        case = f"{md(self.titles.get(document.case_id, document.case_id))}: " if document.case_id != self.record.case_id else ""
        return f"{case}{md(document.title)}, line {line_number(document.text, span.start)}"

    # --- sections ---------------------------------------------------------------------------

    def header(self, analysis: CaseAnalysis, flags: Sequence[Flag], missed: int, stale: int) -> None:
        meta = self.record.meta
        self.add(f"# {md(self.note('report_title'))}: {md(meta.title)}", "")
        if meta.synthetic:
            self.add(f"> **{md(self.note('synthetic_banner'))}**", "")
        judge = f" · **Presiding judge:** {md(meta.presiding_judge)}" if meta.presiding_judge else ""
        self.add(f"**Court:** {md(meta.court)} · **Charge type:** {md(meta.charge_type)}{judge}", "")
        if meta.source_note:
            self.add(f"**Source:** {md(meta.source_note)}", "")
        mode = "answers replayed from the recorded cache" if analysis.llm_mode == "replay" else "run live on this computer"
        self.add(
            f"{len(flags)} findings, each quoted with its source text in the annex. {analysis.dropped_flags} dropped because "
            f"their source text could not be found. Model: {md(analysis.llm_model or 'none')} ({mode}).",
            "",
            f"*{md(self.note('report_disclaimer'))}*",
            "",
        )
        decided = [self.reviews[flag.id] for flag in flags if flag.id in self.reviews]
        if decided or missed or stale:
            count = lambda decision: sum(r.decision == decision for r in decided)  # noqa: E731
            status = render(
                self.messages, "review_status", reviewed=len(decided), total=len(flags),
                accepted=count("accepted"), edited=count("edited"), rejected=count("rejected"), missed=missed,
            )  # fmt: skip
            self.add(f"**{md(status)}**", "")
            if stale:
                self.add(md(render(self.messages, "review_stale", n=stale)), "")

    def coverage(self, absence: AbsenceResult | None) -> None:
        self.add(f"## 1. {md(self.note('report_coverage'))}", "")
        if absence is None or not absence.assessments:
            self.add("This case has no monitoring notes to read.", "")
            return
        flags = {flag.id: flag for flag in absence.flags}
        for assessment in absence.assessments:
            self.add(
                f"### {md(assessment.provision)}: {md(assessment.name)}",
                "",
                f"**{md(self.label(guarantee_badge(assessment)))}** · {md(self.label(assessment.review_status))}",
                "",
            )
            flag = flags.get(assessment.flag_id or "")
            if flag is not None:
                self.add(md(self.message(flag)) + self.ref(flag), "")
            elif assessment.follow_up is not None:
                question = render(
                    self.messages, "absence_follow_up",
                    provision=assessment.provision, name=assessment.name, question=assessment.follow_up.question,
                )  # fmt: skip
                self.add(md(question), "")
                references = jurisprudence.for_follow_up(assessment.follow_up, self.config.jurisprudence)
                if references:
                    self.add(f"{md(self.note('jurisprudence_heading'))}: " + "; ".join(md(ref.citation) for ref in references) + ".", "")
            if assessment.unlabelled_notes:
                n = assessment.unlabelled_notes
                sentences = "1 possibly relevant sentence in the notes was" if n == 1 else f"{n} possibly relevant sentences in the notes were"
                self.add(f"{sentences} not classified by the local model.", "")
            self.add(f"*{md(assessment.citation)}*", "")

    def timeline(self, clock: ClockResult | None) -> None:
        self.add(f"## 2. {md(self.note('report_timeline'))}", "")
        if clock is None or not clock.timeline:
            self.add("No dated events were found in this case.", "")
            return
        benchmarks = {b.id: b for b in self.config.benchmarks.benchmarks}
        names = self.messages.event_labels
        for interval in sorted(clock.intervals, key=lambda i: i.status != "exceeds_benchmark"):
            benchmark = benchmarks[interval.benchmark_id]
            flag = next((f for f in clock.flags if f.id == interval.flag_id), None)
            text = self.message(flag) if flag is not None else interval_text(interval, benchmark, clock.flags, self.messages)
            citation = interval_citation(interval, benchmark, clock.flags, self.config.benchmarks)
            self.add(
                f"- **{md(names[benchmark.from_event])} to {md(names[benchmark.to_event])}** "
                f"({md(benchmark.provision)}): {md(self.label(interval.status))}, "
                f"{md(self.label(interval.review_status))}. {md(text)}{self.ref(flag)}  ",
                f"  *{md(citation)}*",
            )
        self.add("", "| Date | Event | Date status | Sources |", "|---|---|---|---|")
        for event in sorted(clock.timeline, key=lambda e: (e.date is None, e.date)):
            sources = "; ".join(self.where(mention.span) for mention in event.mentions)
            self.add(
                f"| {_date(event)} | {md(names.get(event.type, event.type))} | "
                f"{md(self.label('timeline_' + event.state))} | {sources} |"
            )
        self.add("")

    def renewal(self, renewal: RenewalResult | None) -> None:
        self.add(f"## 3. {md(self.note('report_renewal'))}", "")
        if renewal is None or not renewal.orders:
            self.add("This case has no detention orders.", "")
            return
        self.add("| Order | Date | Detention until | Days since the previous order | Grounds repeating earlier orders | New grounds passages |",
                 "|---|---|---|---|---|---|")  # fmt: skip
        for order in renewal.orders:
            new = f"{order.passages_new} of {order.passages_compared}" if order.share_repeated is not None else ""
            self.add(
                f"| {md(order.title)} | {order.date.isoformat() if order.date else 'undated'} | "
                f"{order.until.isoformat() if order.until else 'not read'} | {'' if order.days_since_previous is None else order.days_since_previous} | "
                f"{'' if order.share_repeated is None else percent(order.share_repeated)} | {new} |"
            )
        self.add("")
        for flag in renewal.flags:
            self.add(f"- **{md(self.label(flag.status))}:** {md(self.message(flag))}{self.ref(flag)}")
        for note in renewal.notes:
            self.add(f"- {md(note)}")
        self.add("")

    def reuse(self, reuse: ReuseResult | None) -> None:
        self.add(f"## 4. {md(self.note('report_reuse'))}", "")
        if reuse is None or reuse.judgment_doc_id is None:
            self.add("This case has no judgment to compare.", "")
            return
        score = f"{percent(reuse.score)} of the court's own reasoning is traceable to the indictment"
        self.add(f"{score}{f' (no score: {md(reuse.score_note)})' if reuse.score is None and reuse.score_note else ''}.", "")
        matches = [flag for flag in reuse.flags if flag.status != "unaddressed_argument"]
        if not matches:
            self.add("No passage of the reasoning matches the indictment.", "")
        for flag in matches:
            self.add(f"- **{md(self.label(flag.status))}:** {md(self.message(flag))}{self.ref(flag)}")
        if matches:
            self.add("")
        self.add("**Defence arguments from the notes**", "")
        if not reuse.arguments:
            self.add("No defence arguments were found in the notes.", "")
        flags = {flag.id: flag for flag in reuse.flags}
        for check in reuse.arguments:
            status = "unchecked_argument" if not check.checked else "addressed_argument" if check.addressed else "unaddressed_argument"
            flag = flags.get(check.flag_id or "")
            message = f" {md(self.message(flag))}" if flag is not None else ""
            self.add(f"- **{md(self.label(status))}** ({self.where(check.argument)}):{message}{self.ref(flag)}", "", _quote(check.argument.text), "")

    def judge(self, judges: JudgeReport | None) -> None:
        self.add(f"## 5. {md(self.note('report_judge'))}", "")
        profile = next((p for p in judges.profiles if self.record.case_id in p.case_ids), None) if judges else None
        self.add(f"*{md(self.note('selection_bias'))}*", "")
        if profile is None:
            self.add(md(self.note("report_no_judge")), "")
            return
        cases = "1 case" if len(profile.case_ids) == 1 else f"{len(profile.case_ids)} cases"
        self.add(
            f"{md(profile.display_name)}, {md(profile.court)}: {cases}, "
            f"{profile.k_compared} indicators compared.",
            "",
        )
        flags = {flag.id: flag for flag in profile.flags}
        for indicator in profile.indicators:
            self.add(self.indicator(profile, indicator, flags.get(indicator.flag_id or "")))
        self.add("")

    def indicator(self, profile: JudgeProfile, indicator: Indicator, flag: Flag | None) -> str:
        line = f"- **{md(indicator.label)}** ({md(indicator.charge_type)})"
        if indicator.shown:
            low, high = indicator.difference_ci
            line += (
                f": this judge {_rate(indicator.judge)}; baseline of {indicator.baseline_judges} other judges "
                f"{_rate(indicator.baseline)}; difference {round(100 * low):+d} to {round(100 * high):+d} percentage points "
                f"({indicator.difference_confidence:.1%} interval, for {profile.k_compared} indicators compared)."
            )
        else:
            line += ":"
        return f"{line} {md(self.message(flag) if flag is not None else indicator.message)}{self.ref(flag)}"

    def missed(self, issues: Sequence[feedback.MissedIssue]) -> None:
        self.add(f"## 6. {md(self.note('report_missed'))}", "")
        for issue in issues:
            by = f" ({md(issue.reviewer)}, {issue.created_at[:10]})" if issue.reviewer else f" ({issue.created_at[:10]})"
            self.add(f"- **{md(self.label('module_' + issue.module))}, {md(issue.standard)}:** {md(issue.note)}{by}")
        self.add("")

    def decision(self, review: feedback.Review) -> None:
        by = f", {md(review.reviewer)}" if review.reviewer else ""
        reason = f" Reason: {md(review.note)}" if review.note.strip() else ""
        self.add(f"**{md(self.label('review_' + review.decision))}** ({review.created_at[:10]}{by}).{reason}", "")
        if review.decision == "edited":
            self.add(f"*{md(self.note('report_ratio_wording'))}:* {md(review.flag.message)}", "")

    def state_reply(self, flag: Flag) -> None:
        reply = self.steelman.for_flag(flag.id) if self.steelman is not None else None
        if reply is None:
            return
        grounds = {g.id: g for g in self.config.steelman.for_standard(reply.standard_id)}
        self.add(f"*{md(self.note('steelman_heading'))}:*", "")
        if not reply.checked:
            self.add(f"- {md(reply.note or 'The model gave no usable answer.')}")
        for argument in reply.arguments:
            label = grounds[argument.ground_id].label if argument.ground_id in grounds else argument.ground_id
            self.add(f"- **{md(label)}.** {md(argument.argument)} It rests on {self.where(argument.span)}:", "", _quote(argument.span.text), "")
        unsupported = [grounds[g].label if g in grounds else g for g in reply.unsupported_grounds]
        if unsupported:
            self.add(f"- {md(self.note('steelman_unsupported'))}: " + "; ".join(md(u) for u in unsupported) + ".")
        self.add("")

    def annex(self) -> None:
        self.add(f"## {md(self.note('report_annex'))}", "", f"*{md(self.note('jurisprudence_caveat'))}*", "", f"*{md(self.note('steelman_caveat'))}*", "")
        for number, flag in self.numbered.values():
            self.add(
                f"### {number} · {md(flag.standard_label)} · {md(self.label(flag.status))}",
                "",
                md(self.message(flag)),
                "",
            )
            if flag.id in self.reviews:
                self.decision(self.reviews[flag.id])
            if flag.citation:
                self.add(f"*{md(flag.citation)}*", "")
            for item in flag.evidence:
                self.add(f"{md(self.label('role_' + item.role))} · {self.where(item.span)}:", "", _quote(item.span.text), "")
            if flag.model_note:
                self.add(f"*{md(self.note('model_note_label'))}:* {md(flag.model_note)}", "")
            self.state_reply(flag)
            references = jurisprudence.for_finding(flag, self.config.jurisprudence)
            if references:
                self.add(f"*{md(self.note('jurisprudence_heading'))}:*", "")
                for ref in references:
                    self.add(f"- {md(ref.citation)}: {md(ref.shown)}")
                self.add("")


def draft_report(
    record: CaseRecord,
    analysis: CaseAnalysis,
    judges: JudgeReport | None,
    config: RatioConfig,
    *,
    history: Sequence[CaseRecord] = (),
    reviews: Sequence[feedback.Review] = (),
    missed: Sequence[feedback.MissedIssue] = (),
) -> str:
    """The case as a Markdown report draft. ``history`` holds the other stored cases, so a judge
    pattern's coded rulings are located by document title and line; ``reviews`` and ``missed`` are
    the reviewers' decisions and missed issues recorded for this case, in the order they were made."""
    flags = feedback.case_findings(analysis, judges)
    in_force = feedback.current(r for r in reviews if r.case_id == record.case_id)
    shown = {flag.id for flag in flags}
    report = _Report(record, config, history, {k: v for k, v in in_force.items() if k in shown})
    report.steelman = analysis.steelman
    missed = [issue for issue in missed if issue.case_id == record.case_id]
    report.header(analysis, flags, len(missed), stale=sum(k not in shown for k in in_force))
    report.coverage(analysis.absence)
    report.timeline(analysis.clock)
    report.renewal(analysis.renewal)
    report.reuse(analysis.reuse)
    report.judge(judges)
    if missed:
        report.missed(missed)
    for flag in flags:  # a finding the body did not cite still reaches the annex
        report.ref(flag)
    report.annex()
    return "\n".join(report.lines).rstrip() + "\n"
