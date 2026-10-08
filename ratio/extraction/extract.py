"""LLM extraction of dated events and party arguments, one model call per chunk.

The prompt lists the chunk's sentences that contain a written date, numbered. The model returns
exact quotes; each quote is located in the source (align.py) and expanded to its sentence, which
becomes the evidence. An event's date must be written in the sentence where its quote starts: it
is parsed in code, and the model's own date is kept only to flag disagreements for review.
Anything that cannot be located is dropped and counted in the report.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ratio.config import ExtractionSettings
from ratio.context import LLMClient
from ratio.extraction.align import Located, locate_quote, negation_mismatch, sentence_bounds
from ratio.extraction.build import OBSERVATION_SOURCES
from ratio.extraction.chunking import Chunk, chunk_spans
from ratio.extraction.dates import find_written_dates, has_explicit_year, parse_date_text, parse_iso_date
from ratio.extraction.prompts import (
    EXTRACTION_SYSTEM,
    ChunkExtraction,
    ModelArgument,
    ModelEvent,
    extraction_user_prompt,
)
from ratio.extraction.segment import sentence_spans
from ratio.schema import Argument, CaseRecord, Document, Event, SourceSpan, stable_id

ProgressCallback = Callable[[str, int, int], None]
_DATE_SEPARATORS = re.compile(r"\s+(?:and|or|to|until)\s+|[,;&]")


@dataclass(frozen=True)
class ExtractionReport:
    documents: int
    chunks: int
    model_calls: int
    events_kept: int
    arguments_kept: int
    dropped: tuple[str, ...]
    reviews: int


def _span(doc: Document, start: int, end: int) -> SourceSpan:
    return SourceSpan(doc_id=doc.id, start=start, end=end, text=doc.text[start:end])


def _dated_sentences(doc: Document, chunk: Chunk) -> list[tuple[int, int]]:
    return [
        (start, end)
        for start, end in sentence_spans(doc.text)
        if chunk.start <= start and end <= chunk.end and find_written_dates(doc.text, start, end)
    ]


def _locate(
    doc: Document, windows: Iterable[tuple[int, int]], quote: str, used: set, settings: ExtractionSettings
) -> Located | None:
    for window in windows:
        located = locate_quote(
            doc.text,
            quote,
            window=window,
            used=used,
            min_chars=settings.min_quote_chars,
            fuzzy_min_chars=settings.fuzzy_min_quote_chars,
            fuzzy_threshold=settings.fuzzy_threshold,
        )
        if located is not None:
            return located
    return None


def _quote_reviews(doc: Document, located: Located, sentence: SourceSpan) -> list[str]:
    reasons = []
    if located.match == "fuzzy":
        reasons.append("quote matched the source only approximately")
    if negation_mismatch(sentence.text, doc.text[located.start : located.end]):
        reasons.append("the source sentence contains a negation the quote left out")
    return reasons


def _find_date_text(doc: Document, sentence: SourceSpan, date_text: str) -> tuple[int, str] | None:
    """Locate the model's date text in the sentence; if it joined several dates, use the first one found."""
    parts = [date_text, *(part.strip() for part in _DATE_SEPARATORS.split(date_text))] if date_text else []
    for candidate in parts:
        if len(candidate) >= 4:
            offset = doc.text.find(candidate, sentence.start, sentence.end)
            if offset != -1:
                return offset, candidate
    return None


def _date_fields(doc: Document, sentence: SourceSpan, item: ModelEvent) -> tuple[dict, list[str]] | None:
    """Parse the date written in the sentence. None when the sentence holds no such date."""
    found = _find_date_text(doc, sentence, item.date_text.strip())
    if found is None:
        return None
    offset, date_text = found
    model_date = parse_iso_date(item.iso_date)
    fields: dict = {
        "model_date": model_date.isoformat() if model_date else None,
        "precision": "unknown",
        "date_span": _span(doc, offset, offset + len(date_text)),
        "date_text": date_text,
    }
    reasons: list[str] = []
    base = dt.datetime.combine(doc.date, dt.time()) if doc.date else None
    parsed = parse_date_text(date_text, base=base)
    if parsed is None:
        return fields, ["date text could not be parsed"]
    fields.update(parsed_date=parsed.value, precision=parsed.precision)
    if not has_explicit_year(date_text):
        reasons.append("the year was inferred, not written")
    if model_date and parsed.value.date() != model_date:
        reasons.append(f"model and dateparser disagree ({model_date.isoformat()} vs {parsed.value.date().isoformat()})")
    return fields, reasons


def _event(
    doc: Document, chunk: Chunk, dated: list[tuple[int, int]], item: ModelEvent, settings: ExtractionSettings
) -> Event | str:
    numbered = [dated[item.sentence - 1]] if 1 <= item.sentence <= len(dated) else []
    located = _locate(doc, [*numbered, (chunk.start, chunk.end)], item.quote, set(), settings)
    if located is None:
        return f"{doc.path}: {item.type} quote not found in source: {item.quote[:60]!r}"
    first_sentence = _span(doc, *sentence_bounds(doc.text, located.start, located.start + 1))
    dated_fields = _date_fields(doc, first_sentence, item)
    if dated_fields is None:
        return f"{doc.path}: {item.type} dropped, no date written in its sentence: {first_sentence.text[:60]!r}"
    date_fields, date_reasons = dated_fields
    evidence = _span(doc, *sentence_bounds(doc.text, located.start, located.end))
    reasons = _quote_reviews(doc, located, evidence) + date_reasons
    return Event(
        id=stable_id(doc.id, located.start, item.type, date_fields["date_span"].start, "event"),
        type=item.type,
        span=evidence,
        quote_span=_span(doc, located.start, located.end),
        actors=tuple(a for a in item.actors if a.strip()),
        match=located.match,
        needs_review=bool(reasons),
        review_reasons=tuple(reasons),
        **date_fields,
    )


def _argument(doc: Document, chunk: Chunk, item: ModelArgument, used: set, settings: ExtractionSettings) -> Argument | str:
    if doc.type not in OBSERVATION_SOURCES:
        return f"{doc.path}: argument ignored (arguments are taken from monitoring notes only)"
    located = _locate(doc, [(chunk.start, chunk.end)], item.quote, used, settings)
    if located is None:
        return f"{doc.path}: argument quote not found in source: {item.quote[:60]!r}"
    sentence = _span(doc, *sentence_bounds(doc.text, located.start, located.end))
    if not any(marker in sentence.text.lower() for marker in settings.argument_markers):
        return f"{doc.path}: argument dropped (no argument verb such as 'argued'): {sentence.text[:60]!r}"
    used.add((located.start, located.end))
    return Argument(
        id=stable_id(doc.id, located.start, item.party, "argument"),
        party=item.party,
        text=sentence.text,
        hearing_date=doc.date,
        span=sentence,
        quote_span=_span(doc, located.start, located.end),
        match=located.match,
        needs_review=bool(_quote_reviews(doc, located, sentence)),
    )


def _dedupe(items: list, key: Callable) -> list:
    seen: set = set()
    unique = []
    for item in items:
        if key(item) not in seen:
            seen.add(key(item))
            unique.append(item)
    return unique


def _event_key(event: Event) -> tuple:
    return (event.type, event.span.doc_id, event.span.start, event.date_span.start if event.date_span else None)


def _argument_key(argument: Argument) -> tuple:
    return (argument.party, argument.span.doc_id, argument.span.start, argument.span.end)


def extract_record(
    base: CaseRecord,
    llm: LLMClient,
    settings: ExtractionSettings,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[CaseRecord, ExtractionReport]:
    """Add model-extracted events and arguments to a deterministic base record."""
    plan = [
        (doc, chunk)
        for doc in base.documents
        for chunk in chunk_spans(doc.text, max_chars=settings.chunk_chars, overlap_chars=settings.chunk_overlap_chars)
    ]
    events: list[Event] = []
    arguments: list[Argument] = []
    dropped: list[str] = []
    calls = 0
    for done, (doc, chunk) in enumerate(plan, start=1):
        dated = _dated_sentences(doc, chunk)
        is_note = doc.type in OBSERVATION_SOURCES
        if dated or is_note:
            calls += 1
            reply = llm.complete_json(
                system=EXTRACTION_SYSTEM,
                user=extraction_user_prompt(
                    doc.type,
                    doc.title,
                    [doc.text[s:e] for s, e in dated],
                    doc.text[chunk.start : chunk.end] if is_note else None,
                ),
                schema=ChunkExtraction,
                purpose="extraction",
            )
            for item in reply.events:
                result = _event(doc, chunk, dated, item, settings)
                (events.append if isinstance(result, Event) else dropped.append)(result)
            used: set[tuple[int, int]] = set()
            for item in reply.arguments:
                result = _argument(doc, chunk, item, used, settings)
                (arguments.append if isinstance(result, Argument) else dropped.append)(result)
        if progress is not None:
            progress(doc.title, done, len(plan))
    kept_events = _dedupe(events, _event_key)
    kept_arguments = _dedupe(arguments, _argument_key)
    record = CaseRecord.model_validate(
        {**base.model_dump(), "events": [*base.model_dump()["events"], *(e.model_dump() for e in kept_events)],
         "arguments": [a.model_dump() for a in kept_arguments]}
    )
    report = ExtractionReport(
        documents=len(base.documents),
        chunks=len(plan),
        model_calls=calls,
        events_kept=len(kept_events),
        arguments_kept=len(kept_arguments),
        dropped=tuple(dropped),
        reviews=sum(e.needs_review for e in kept_events) + sum(a.needs_review for a in kept_arguments),
    )
    return record, report
