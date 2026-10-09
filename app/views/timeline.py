"""Procedural timeline: events merged from every document, and intervals measured against cited
benchmarks. Red only where a confirmed benchmark is exceeded (hard rule 5)."""

from __future__ import annotations

import streamlit as st

from ratio.config import Benchmark
from ratio.display import duration_text, md_escape
from ratio.messages import render
from ratio.results import ClockResult, Interval
from ratio_ui import session, style, widgets
from ratio_ui.session import LoadedCase

STATUS_COLOURS = {
    "exceeds_benchmark": "#a8353c",
    "may_exceed": "#896324",
    "needs_review": "#896324",
    "within_benchmark": "#32684c",
    "measured": "#8995a3",
    "cannot_compute": "#8995a3",
}
EVENT_COLOUR = "#28587d"


def _interval_text(interval: Interval, benchmark: Benchmark, clock: ClockResult) -> str:
    messages = session.config().messages
    names = messages.event_labels
    flag = next((f for f in clock.flags if f.id == interval.flag_id), None)
    if flag is not None:
        return flag.message
    if interval.min_hours is None or interval.max_hours is None:
        return f"{names[benchmark.from_event]} to {names[benchmark.to_event]}: cannot be measured. {interval.note}".strip()
    duration = duration_text(interval.min_hours, interval.max_hours)
    if interval.status == "measured":
        return render(messages, "clock_measured", from_label=names[benchmark.from_event], to_label=names[benchmark.to_event], duration=duration)
    return f"{names[benchmark.from_event]} to {names[benchmark.to_event]}: {duration}, within the {benchmark.threshold_hours:g}-hour benchmark."


def _citation(interval: Interval, benchmark: Benchmark, clock: ClockResult) -> str:
    flag = next((f for f in clock.flags if f.id == interval.flag_id), None)
    if flag is not None and flag.citation:
        return flag.citation
    source = session.config().benchmarks.sources[benchmark.citation.instrument]
    return f"{benchmark.provision}; {source.symbol}, para. {benchmark.citation.paras}"


def _intervals(loaded: LoadedCase, clock: ClockResult) -> None:
    benchmarks = {b.id: b for b in session.config().benchmarks.benchmarks}
    names = session.config().messages.event_labels
    for interval in sorted(clock.intervals, key=lambda i: i.status != "exceeds_benchmark"):
        benchmark = benchmarks[interval.benchmark_id]
        with style.panel(f"interval-{interval.id}"):
            st.markdown(f"**{names[benchmark.from_event]} → {names[benchmark.to_event]}** · {md_escape(benchmark.provision)}")
            with st.container(horizontal=True):
                widgets.badge(interval.status)
                widgets.badge(interval.review_status)
            st.markdown(md_escape(_interval_text(interval, benchmark, clock)))
            st.caption(md_escape(_citation(interval, benchmark, clock)))
            if benchmark.note:
                st.caption(md_escape(benchmark.note))
            if interval.evidence:
                with st.expander(f"Sources ({len(interval.evidence)})", expanded=interval.status == "exceeds_benchmark"):
                    widgets.evidence(loaded.record, interval.evidence, key=f"interval-{interval.id}", heading=benchmark.name)


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
    tooltip = [{"field": "row", "title": "Item"}, {"field": "kind", "title": "Status"}, {"field": "detail", "title": "Detail"}]
    spec = {
        "height": 30 * len(order) + 40,
        "background": "#ffffff",
        "config": {
            "font": "Arial",
            "view": {"stroke": None},
            "axis": {"labelColor": "#5e6875", "labelFontSize": 11, "gridColor": "#eef0f3", "domainColor": "#dce0e5", "labelPadding": 8},
            "legend": {"labelColor": "#5e6875", "labelFontSize": 11, "padding": 16},
        },
        "layer": [
            {
                "data": {"values": bars},
                "mark": {"type": "rule", "strokeWidth": 8, "strokeCap": "round"},
                "encoding": {"y": y, "x": {"field": "start", "type": "temporal", "title": None}, "x2": {"field": "end"},
                             "color": {"field": "kind", "type": "nominal", "scale": scale, "legend": {"title": None, "orient": "bottom"}},
                             "tooltip": tooltip},
            },
            {
                "data": {"values": points},
                "mark": {"type": "point", "filled": True, "size": 90},
                "encoding": {"y": y, "x": {"field": "date", "type": "temporal", "title": None},
                             "color": {"field": "kind", "type": "nominal", "scale": scale}, "tooltip": tooltip},
            },
        ],
    }  # fmt: skip
    st.vega_lite_chart(spec=spec, width="stretch")


def _events(loaded: LoadedCase, clock: ClockResult) -> None:
    names = session.config().messages.event_labels
    for event in sorted(clock.timeline, key=lambda e: (e.date is None, e.date)):
        when = event.date.date().isoformat() if event.date else "undated"
        with st.expander(f"{when} · {names.get(event.type, event.type)} · {widgets.label('timeline_' + event.state)} · {len(event.mentions)} source(s)"):
            if event.review_reasons:
                st.caption(md_escape("; ".join(event.review_reasons)))
            widgets.evidence(loaded.record, event.mentions, key=f"event-{event.id}", heading=names.get(event.type, event.type))


loaded = widgets.require_case()
clock = loaded.analysis.clock
widgets.header(loaded, "Procedural timeline", "events from every document, measured against cited benchmarks")
st.markdown(
    "Intervals are coloured only against a **confirmed** benchmark: General Comment 35's 48 hours to bring a "
    "detainee before a judge. Other benchmarks are cited and measured, never coloured, until legal review confirms them."
)
if clock is None or not clock.timeline:
    st.info("No dated events were found in this case.")
    st.stop()
style.section("Chronology", "Events and measured intervals from the source record. Hover for status details.")
with style.panel("chronology"):
    _chart(clock)
style.section("Benchmark review", "Confirmed benchmarks appear first. Each interval includes the source dates and its citation.")
_intervals(loaded, clock)
style.section("Event record", "Open an event to read every source mention and any date conflicts.")
_events(loaded, clock)
