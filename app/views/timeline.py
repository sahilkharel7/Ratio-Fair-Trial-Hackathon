"""Procedural timeline: events merged from every document, and intervals measured against cited
benchmarks. Red only where a confirmed benchmark is exceeded (hard rule 5)."""

from __future__ import annotations

import streamlit as st

from ratio.display import md_escape
from ratio.jurisprudence import for_finding
from ratio.report import interval_citation, interval_text
from ratio.results import ClockResult, TimelineEvent
from ratio_ui import session, widgets
from ratio_ui.session import LoadedCase

# Vega-Lite needs its colours in the spec. Each one has at least 3:1 contrast on the light and the dark
# theme background (WCAG 1.4.11),
# and red stays reserved for an interval longer than the confirmed benchmark.
STATUS_COLOURS = {
    "exceeds_benchmark": "#c0392b",
    "may_exceed": "#d97706",
    "needs_review": "#d97706",
    "within_benchmark": "#2e7d32",
    "measured": "#64748b",
    "cannot_compute": "#64748b",
}
EVENT_COLOUR = "#2563eb"
DATE_FORMAT = "%-d %B %Y"  # d3 time format: "4 March 2025"


def _when(event: TimelineEvent) -> str:
    """The event's date written out, no more precise than its source: "4 March 2025, 22:10", "March 2025"."""
    if event.date is None:
        return "Undated"
    if event.precision == "year":
        return str(event.date.year)
    if event.precision == "month":
        return f"{event.date:%B %Y}"
    day = f"{event.date.day} {event.date:%B %Y}"
    return f"{day}, {event.date:%H:%M}" if event.precision == "datetime" else day


def _intervals(loaded: LoadedCase, clock: ClockResult) -> None:
    cfg = session.config()
    benchmarks = {b.id: b for b in cfg.benchmarks.benchmarks}
    for interval in sorted(clock.intervals, key=lambda i: i.status != "exceeds_benchmark"):
        benchmark = benchmarks[interval.benchmark_id]
        with st.container(border=True):
            st.markdown(f"**{md_escape(benchmark.name)}** · {md_escape(benchmark.provision)}")
            with st.container(horizontal=True):
                widgets.badge(interval.status)
                widgets.badge(interval.review_status)
            st.markdown(md_escape(interval_text(interval, benchmark, clock.flags, cfg.messages)))
            st.caption(md_escape(interval_citation(interval, benchmark, clock.flags, cfg.benchmarks)))
            if benchmark.note:
                st.caption(md_escape(benchmark.note))
            if interval.evidence:
                with st.expander(f"Sources ({len(interval.evidence)})", expanded=interval.status == "exceeds_benchmark"):
                    widgets.evidence(loaded.record, interval.evidence, key=f"interval-{interval.id}", heading=benchmark.name)
            flag = next((f for f in clock.flags if f.id == interval.flag_id), None)
            if flag is not None:
                widgets.jurisprudence(for_finding(flag, cfg.jurisprudence))
                if loaded.analysis.steelman is not None:
                    widgets.state_reply(loaded.record, loaded.analysis.steelman.for_flag(flag.id), key=f"state-{flag.id}")


def _chart(clock: ClockResult) -> None:
    cfg = session.config()
    names = cfg.messages.event_labels
    dated = {event.id: event for event in clock.timeline if event.date is not None}
    points = [
        {"row": names.get(e.type, e.type), "date": e.date.isoformat(), "kind": "Event", "detail": widgets.label(f"timeline_{e.state}")}
        for e in sorted(dated.values(), key=lambda e: e.date)
    ]
    benchmarks = {b.id: b for b in cfg.benchmarks.benchmarks}
    bars = [
        {"row": benchmarks[i.benchmark_id].name, "start": dated[i.from_event_id].date.isoformat(),
         "end": dated[i.to_event_id].date.isoformat(), "kind": widgets.label(i.status), "detail": widgets.label(i.review_status)}
        for i in clock.intervals
        if i.from_event_id in dated and i.to_event_id in dated
    ]  # fmt: skip
    order = list(dict.fromkeys(p["row"] for p in points)) + [b["row"] for b in bars]
    shown = {bar["kind"] for bar in bars}
    legend = [("Event", EVENT_COLOUR)] + [
        (widgets.label(status), colour) for status, colour in STATUS_COLOURS.items() if widgets.label(status) in shown
    ]
    scale = {"domain": [name for name, _ in legend], "range": [colour for _, colour in legend]}
    y = {"field": "row", "type": "nominal", "sort": order, "title": None, "axis": {"labelLimit": 260}}
    tooltip = [{"field": "row", "title": "Event or benchmark"}, {"field": "kind", "title": "Status"}, {"field": "detail", "title": "Review"}]
    bar_dates = [{"field": "start", "type": "temporal", "title": "From", "format": DATE_FORMAT},
                 {"field": "end", "type": "temporal", "title": "To", "format": DATE_FORMAT}]  # fmt: skip
    point_date = [{"field": "date", "type": "temporal", "title": "Date", "format": DATE_FORMAT}]
    spec = {
        "height": 30 * len(order) + 40,
        "layer": [
            {
                "data": {"values": bars},
                "mark": {"type": "rule", "strokeWidth": 8, "strokeCap": "round"},
                "encoding": {"y": y, "x": {"field": "start", "type": "temporal", "title": None}, "x2": {"field": "end"},
                             "color": {"field": "kind", "type": "nominal", "scale": scale, "legend": {"title": None, "orient": "bottom"}},
                             "tooltip": tooltip + bar_dates},
            },
            {
                "data": {"values": points},
                "mark": {"type": "point", "filled": True, "size": 90},
                "encoding": {"y": y, "x": {"field": "date", "type": "temporal", "title": None},
                             "color": {"field": "kind", "type": "nominal", "scale": scale}, "tooltip": tooltip + point_date},
            },
        ],
    }  # fmt: skip
    st.vega_lite_chart(spec=spec, width="stretch")


def _events(loaded: LoadedCase, clock: ClockResult) -> None:
    names = session.config().messages.event_labels
    for event in sorted(clock.timeline, key=lambda e: (e.date is None, e.date)):
        sources = f"{len(event.mentions)} source" + ("" if len(event.mentions) == 1 else "s")
        with st.expander(f"{_when(event)} · {names.get(event.type, event.type)} · {widgets.label('timeline_' + event.state)} · {sources}"):
            if event.review_reasons:
                st.caption(md_escape("Check before relying on this event: " + "; ".join(event.review_reasons) + "."))
            widgets.evidence(loaded.record, event.mentions, key=f"event-{event.id}", heading=names.get(event.type, event.type))


loaded = widgets.require_case()
clock = loaded.analysis.clock
widgets.header(loaded, "Procedural timeline", "events from every document, measured against cited benchmarks")
st.markdown(
    "Only one benchmark is confirmed: General Comment 35's 48 hours to bring a detainee before a judge. An interval "
    f"longer than that is labelled **{widgets.label('exceeds_benchmark')}** and shown in red. The other benchmarks are "
    f"cited and measured but never shown in red, and stay labelled **{widgets.label('needs_legal_review')}** until "
    "legal review confirms them."
)
if clock is None or not clock.timeline:
    st.info("No dated events were found in this case.")
    st.stop()
st.header("Intervals measured against benchmarks")
_intervals(loaded, clock)
st.header("Chart of events and intervals")
st.caption(
    "Points are events and bars are the intervals above, coloured by the status named in the legend. "
    "Every date and status in the chart is also written out in the lists on this page."
)
_chart(clock)
st.header("Events in date order")
st.caption("Each event brings together every mention of it in the documents. Open an event to read its sources.")
_events(loaded, clock)
