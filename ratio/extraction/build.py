"""Build the deterministic part of a case record, with no model involved:
documents, metadata, observations (one per note sentence), passages of court documents,
statute citations, hearing events from note headers, and hand-coded rulings.
The LLM layer adds events and arguments on top of this record.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path

from ratio.extraction.loader import (
    CaseManifest,
    build_documents,
    build_meta,
    build_rulings,
    read_case_folder,
)
from ratio.extraction.segment import Segment, find_citations, segment_document
from ratio.schema import CaseRecord, Citation, Document, Event, Observation, Passage, SourceSpan, stable_id

OBSERVATION_SOURCES = frozenset({"monitoring_note", "transcript"})


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


def build_base_record(manifest: CaseManifest, files: Mapping[str, bytes]) -> CaseRecord:
    documents = build_documents(manifest, files)
    observations: list[Observation] = []
    passages: list[Passage] = []
    citations: list[Citation] = []
    hearings: list[Event] = []
    for doc in documents:
        segments = segment_document(doc.text)
        if doc.type in OBSERVATION_SOURCES:
            observations.extend(_observation(doc, seg) for seg in segments if seg.kind == "body")
            hearing = _hearing_event(doc)
            if hearing is not None:
                hearings.append(hearing)
        else:
            passages.extend(_passage(doc, index, seg) for index, seg in enumerate(segments))
            citations.extend(
                Citation(id=stable_id(doc.id, s, e, "citation"), text=doc.text[s:e], span=_span(doc, s, e))
                for s, e in find_citations(doc.text)
            )
    rulings_data = files.get(manifest.rulings) if manifest.rulings else None
    return CaseRecord(
        meta=build_meta(manifest, documents),
        documents=documents,
        events=tuple(hearings),
        observations=tuple(observations),
        citations=tuple(citations),
        passages=tuple(passages),
        rulings=build_rulings(manifest, documents, rulings_data),
    )


def load_case(folder: Path) -> CaseRecord:
    """Deterministic record for a case folder containing case.yaml."""
    manifest, files = read_case_folder(folder)
    return build_base_record(manifest, files)
