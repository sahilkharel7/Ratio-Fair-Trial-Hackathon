"""Build the deterministic part of a case record, with no model involved:
documents, metadata, observations (one per note sentence), passages of court documents,
statute citations, hearing events from note headers, the verdict date from a judgment's caption
("Delivered on 14 July 2025"), detention orders and their extension events (extraction/orders.py),
and hand-coded rulings.
The LLM layer adds events and arguments on top of this record.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from pathlib import Path

from ratio.extraction.loader import (
    CaseManifest,
    build_documents,
    build_meta,
    build_rulings,
    read_case_folder,
)
from ratio.extraction.dates import find_written_dates, parse_date_text
from ratio.extraction.orders import extension_events, read_orders
from ratio.extraction.segment import Segment, caption_lines, find_citations, segment_document
from ratio.schema import CaseRecord, Citation, Document, Event, Observation, Passage, SourceSpan, stable_id

OBSERVATION_SOURCES = frozenset({"monitoring_note", "transcript"})
_DELIVERY_LINE = re.compile(r"(?:delivered(?: on)?:?|date of judgment:?|date:)\s", re.IGNORECASE)  # a judgment caption line that dates it


def _span(doc: Document, start: int, end: int) -> SourceSpan:
    return SourceSpan(doc_id=doc.id, start=start, end=end, text=doc.text[start:end])


def _observation(doc: Document, segment: Segment) -> Observation:
    span = _span(doc, segment.start, segment.end)
    return Observation(id=stable_id(doc.id, segment.start, segment.end, "obs"), text=span.text, hearing_date=doc.date, span=span)


def _passage(doc: Document, index: int, segment: Segment) -> Passage:
    return Passage(
        id=stable_id(doc.id, segment.start, segment.end, "passage"),
        doc_id=doc.id,
        index=index,
        section=segment.section,
        chapter=segment.chapter,
        kind=segment.kind,
        span=_span(doc, segment.start, segment.end),
    )


def _hearing_event(doc: Document) -> Event | None:
    if doc.date is None or doc.date_span is None:
        return None
    line_start = doc.text.rfind("\n", 0, doc.date_span.start) + 1
    line_end = doc.text.find("\n", doc.date_span.end)
    line = _span(doc, line_start, len(doc.text) if line_end == -1 else line_end)
    return Event(
        id=stable_id(doc.id, line.start, "hearing"),
        type="hearing",
        span=line,
        quote_span=line,
        date_span=doc.date_span,
        date_text=doc.date_span.text,
        parsed_date=dt.datetime.combine(doc.date, dt.time()),
        precision="date",
    )


def _verdict_event(doc: Document) -> Event | None:
    """The verdict date from a judgment's caption line ("Delivered on 14 July 2025", "Date: ..."),
    with that line as the source, so the date is always shown with the text that gives it."""
    if doc.type != "judgment":
        return None
    for start, end in caption_lines(doc.text):
        dates = find_written_dates(doc.text, start, end)
        parsed = parse_date_text(doc.text[slice(*dates[0])]) if len(dates) == 1 else None
        if parsed is None or parsed.precision != "date" or not _DELIVERY_LINE.match(doc.text, start):
            continue
        line, date_span = _span(doc, start, end), _span(doc, *dates[0])
        return Event(
            id=stable_id(doc.id, start, "verdict"),
            type="verdict",
            span=line,
            quote_span=line,
            date_span=date_span,
            date_text=date_span.text,
            parsed_date=parsed.value,
            precision=parsed.precision,
        )
    return None


def header_events(doc: Document) -> tuple[Event, ...]:
    """Events read from a document's header, with no model: a note's hearing date, a judgment's verdict date."""
    event = _hearing_event(doc) if doc.type in OBSERVATION_SOURCES else _verdict_event(doc)
    return (event,) if event is not None else ()


def build_base_record(manifest: CaseManifest, files: Mapping[str, bytes]) -> CaseRecord:
    documents = build_documents(manifest, files)
    observations: list[Observation] = []
    passages: list[Passage] = []
    citations: list[Citation] = []
    built_events: list[Event] = []  # hearings from note headers, the verdict from a judgment caption
    for doc in documents:
        built_events.extend(header_events(doc))
        segments = segment_document(doc.text)
        if doc.type in OBSERVATION_SOURCES:
            observations.extend(_observation(doc, seg) for seg in segments if seg.kind == "body")
        else:
            passages.extend(_passage(doc, index, seg) for index, seg in enumerate(segments))
            citations.extend(
                Citation(id=stable_id(doc.id, s, e, "citation"), text=doc.text[s:e], span=_span(doc, s, e))
                for s, e in find_citations(doc.text)
            )
    rulings_data = files.get(manifest.rulings) if manifest.rulings else None
    orders = read_orders(documents)
    return CaseRecord(
        meta=build_meta(manifest, documents),
        documents=documents,
        events=(*built_events, *extension_events(orders, documents)),
        observations=tuple(observations),
        citations=tuple(citations),
        passages=tuple(passages),
        rulings=build_rulings(manifest, documents, rulings_data),
        orders=orders,
    )


def load_case(folder: Path) -> CaseRecord:
    """Deterministic record for a case folder containing case.yaml."""
    manifest, files = read_case_folder(folder)
    return build_base_record(manifest, files)
