"""LLM extraction of dated events and party arguments, one model call per chunk.

The prompt lists the chunk's sentences that contain a written date, numbered. The model returns
exact quotes; each quote is located in the source (align.py) and expanded to its sentence, which
becomes the evidence. An event's date must be written in the sentence where its quote starts: it
is parsed in code, and the model's own date is kept only to flag disagreements for review.
Anything that cannot be located is dropped and counted in the report. Detention orders are read in
code (extraction/orders.py), so they are never sent to the model.
"""

from __future__ import annotations

import datetime as dt
import functools
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ratio.config import ExtractionSettings
from ratio.context import LLMClient
from ratio.extraction.align import PLACEHOLDERS, Located, locate_quote, negation_mismatch, sentence_bounds
from ratio.extraction.backstop import missed_arguments
from ratio.extraction.build import OBSERVATION_SOURCES
from ratio.extraction.chunking import Chunk, chunk_spans
from ratio.extraction.dates import find_written_dates, has_explicit_year, locate_date_text, parse_date_text, parse_iso_date
from ratio.extraction.prompts import (
    EXTRACTION_SYSTEM,
    ChunkExtraction,
    ModelArgument,
    ModelEvent,
    extraction_user_prompt,
)
from ratio.extraction.segment import sentence_spans
from ratio.llm import LLMResponseError, PromptTooLong
from ratio.schema import PARAGRAPH_BREAK, Argument, CaseRecord, Document, Event, SourceSpan, stable_id

ProgressCallback = Callable[[str, int, int], None]
READ_IN_CODE = frozenset({"detention_order"})  # their dates are read by extraction/orders.py
_DATE_SEPARATORS = re.compile(r"\s+(?:and|or|to|until)\s+|[,;&]")
_OPENING_MARKS = " \t\"'“‘«(["  # a numbered point may open inside a quotation or a bracket
_LIST_COUNT = r"(?:\d+|two|three|four|five|six|several)"  # "one point" needs no "First,"


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
    """Exact and normalised matches in every window first; fuzzy matching only after that."""
    windows = list(windows)
    for fuzzy_min_chars in (sys.maxsize, settings.fuzzy_min_quote_chars):
        for window in windows:
            located = locate_quote(
                doc.text,
                quote,
                window=window,
                used=used,
                min_chars=settings.min_quote_chars,
                fuzzy_min_chars=fuzzy_min_chars,
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


def _model_dates(iso_text: str | None, count: int) -> tuple[list[str | None], list[str]]:
    """The model's ISO date for each of ``count`` dates found, and any review reason."""
    if count == 1:
        return [iso_text], []
    parts = [part.strip() for part in re.split(r"[,;]|\s+and\s+", iso_text or "") if part.strip()]
    if len(parts) == count:
        return list(parts), []
    return [None] * count, ["the model gave several dates in one answer"]


def _find_dates(doc: Document, sentence: SourceSpan, date_text: str) -> list[tuple[int, int]]:
    """Every date the model's date text names in the sentence: the whole text, or each date in it
    when the model joined several ("19 March 2025, 14 May 2025")."""
    if not date_text.strip():
        return []
    whole = locate_date_text(doc.text, sentence.start, sentence.end, date_text)
    if whole is not None:
        return [whole]
    parts = [date_text[a:b] for a, b in find_written_dates(date_text)] or _DATE_SEPARATORS.split(date_text)
    found: list[tuple[int, int]] = []
    for part in (part.strip() for part in parts):
        located = locate_date_text(doc.text, sentence.start, sentence.end, part) if len(part) >= 4 else None
        if located is not None and located not in found:
            found.append(located)
    return found


def _within_quote(doc: Document, located: Located, date: tuple[int, int], only: bool) -> tuple[tuple[int, int], list[str]]:
    """Prefer the date written inside the quote: a model can attach another clause's date."""
    quote_dates = find_written_dates(doc.text, located.start, located.end)
    if not quote_dates or (located.start <= date[0] and date[1] <= located.end):
        return date, []
    if only and len(quote_dates) == 1:
        return quote_dates[0], ["the model's date was not the date in its quote; the quote's date was used"]
    return date, ["the date is outside the quoted words"]


def _date_fields(doc: Document, date: tuple[int, int], model_iso: str | None) -> tuple[dict, list[str]]:
    """Parse the date written at ``date`` in code; the model's own date is kept for comparison."""
    reasons: list[str] = []
    parsed = parse_date_text(doc.text[date[0] : date[1]], base=_base(doc))
    if parsed is None:  # "on Monday 14 February 2025": fall back to the full date inside the text
        inner = find_written_dates(doc.text, *date)
        if len(inner) == 1:
            date = inner[0]
            parsed = parse_date_text(doc.text[date[0] : date[1]], base=_base(doc))
    model_date = parse_iso_date(model_iso)
    if model_iso and model_iso.strip().lower() not in PLACEHOLDERS and model_date is None:
        reasons.append("the model's own date could not be read")
    date_text = doc.text[date[0] : date[1]]
    fields: dict = {
        "model_date": model_date.isoformat() if model_date else None,
        "precision": "unknown",
        "date_span": _span(doc, *date),
        "date_text": date_text,
    }
    if parsed is None:
        return fields, [*reasons, "date text could not be parsed"]
    fields.update(parsed_date=parsed.value, precision=parsed.precision)
    if not has_explicit_year(date_text):
        reasons.append("the year was inferred, not written")
    if model_date and parsed.value.date() != model_date:
        reasons.append(f"model and dateparser disagree ({model_date.isoformat()} vs {parsed.value.date().isoformat()})")
    return fields, reasons


def _base(doc: Document) -> dt.datetime | None:
    return dt.datetime.combine(doc.date, dt.time()) if doc.date else None


def _event(
    doc: Document, chunk: Chunk, dated: list[tuple[int, int]], item: ModelEvent, settings: ExtractionSettings
) -> list[Event] | str:
    """One event per date the model named; the date must be written in the quote's first sentence."""
    numbered = [dated[item.sentence - 1]] if 1 <= item.sentence <= len(dated) else []
    located = _locate(doc, [*numbered, (chunk.start, chunk.end)], item.quote, set(), settings)
    if located is None:
        return f"{doc.path}: {item.type} quote not found in source: {item.quote[:60]!r}"
    first_sentence = _span(doc, *sentence_bounds(doc.text, located.start, located.start + 1))
    dates = _find_dates(doc, first_sentence, item.date_text)
    if not dates:
        return f"{doc.path}: {item.type} dropped, no date written in its sentence: {first_sentence.text[:60]!r}"
    evidence = _span(doc, *sentence_bounds(doc.text, located.start, located.end))
    if not _names_event(item.type, evidence.text, settings):
        return f"{doc.path}: {item.type} dropped, its sentence does not mention a {item.type.replace('_', ' ')}: {evidence.text[:60]!r}"
    quote_reasons = _quote_reviews(doc, located, evidence)
    model_dates, joined_reasons = _model_dates(item.iso_date, len(dates))
    events = []
    for date, model_iso in zip(dates, model_dates, strict=True):
        date, quote_date_reasons = _within_quote(doc, located, date, only=len(dates) == 1)
        date_fields, date_reasons = _date_fields(doc, date, model_iso)
        reasons = [*quote_reasons, *joined_reasons, *quote_date_reasons, *date_reasons]
        events.append(
            Event(
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
        )
    return events


@functools.cache
def _support(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"\b(?:{pattern})", re.IGNORECASE | re.DOTALL)


def _names_event(event_type: str, sentence: str, settings: ExtractionSettings) -> bool:
    """The sentence names this kind of event (settings.event_support); types without a pattern pass."""
    pattern = settings.event_support.get(event_type)
    return pattern is None or _support(pattern).search(sentence) is not None


@functools.cache
def _list_announcement(nouns: tuple[str, ...]) -> re.Pattern[str]:
    """'two points', 'three main grounds': a count, at most one more word, then a list noun."""
    return re.compile(rf"(?<![\w-]){_LIST_COUNT}\s+(?:[a-z-]+\s+)?(?:{'|'.join(map(re.escape, nouns))})\b")


def _argued(doc: Document, sentence: SourceSpan, settings: ExtractionSettings) -> bool:
    """The sentence has an argument verb, or opens with "First," (and the like) in a paragraph where an
    earlier sentence has one and announces a list: "Counsel argued two points. First, ... Second, ..."."""
    text = sentence.text.lower()
    if any(marker in text for marker in settings.argument_markers):
        return True
    if not settings.argument_list_nouns or not text.lstrip(_OPENING_MARKS).startswith(settings.argument_continuations):
        return False
    breaks = [found.end() for found in PARAGRAPH_BREAK.finditer(doc.text, 0, sentence.start)]
    paragraph_start = breaks[-1] if breaks else 0
    announcement = _list_announcement(settings.argument_list_nouns)
    earlier = (doc.text[s:e].lower() for s, e in sentence_spans(doc.text) if paragraph_start <= s and e <= sentence.start)
    return any(announcement.search(text) and any(m in text for m in settings.argument_markers) for text in earlier)


def _argument(doc: Document, chunk: Chunk, item: ModelArgument, used: set, settings: ExtractionSettings) -> Argument | str:
    if doc.type not in OBSERVATION_SOURCES:
        return f"{doc.path}: argument ignored (arguments are taken from monitoring notes only)"
    located = _locate(doc, [(chunk.start, chunk.end)], item.quote, used, settings)
    if located is None:
        return f"{doc.path}: argument quote not found in source: {item.quote[:60]!r}"
    sentence = _span(doc, *sentence_bounds(doc.text, located.start, located.end))
    if not _argued(doc, sentence, settings):
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
    """One item per key (chunk overlaps repeat items): the best copy, in first-seen order."""
    best: dict = {}
    for item in items:
        current = best.get(key(item))
        if current is None or _quality(item) < _quality(current):
            best[key(item)] = item
    return list(best.values())


def _quality(item: Event | Argument) -> tuple:
    """Lower is better: dated, day-level, clean, exact."""
    undated = getattr(item, "parsed_date", True) is None
    vague = getattr(item, "precision", "date") not in ("date", "datetime")
    return (undated, vague, item.needs_review, item.match != "exact")


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
        if doc.type not in READ_IN_CODE
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
            try:
                reply = llm.complete_json(
                    system=EXTRACTION_SYSTEM,
                    user=extraction_user_prompt(
                        doc.type,
                        doc.title,
                        [" ".join(doc.text[s:e].split()) for s, e in dated],  # one line per numbered sentence
                        doc.text[chunk.start : chunk.end] if is_note else None,
                    ),
                    schema=ChunkExtraction,
                    purpose="extraction",
                )
            except (PromptTooLong, LLMResponseError) as exc:  # one unreadable part must not lose the case
                dropped.append(f"{doc.path}: part {chunk.index + 1} skipped: {exc}")
                reply = ChunkExtraction(events=[], arguments=[])
            for item in reply.events:
                result = _event(doc, chunk, dated, item, settings)
                (events.extend if isinstance(result, list) else dropped.append)(result)
            used: set[tuple[int, int]] = set()
            for item in reply.arguments:
                result = _argument(doc, chunk, item, used, settings)
                (arguments.append if isinstance(result, Argument) else dropped.append)(result)
        if progress is not None:
            progress(doc.title, done, len(plan))
    dropped += [
        f"{doc.path}: no written date was recognised, so no events were read from it"
        for doc in base.documents
        if doc.type not in OBSERVATION_SOURCES | READ_IN_CODE and not find_written_dates(doc.text)
    ]
    built = {_event_key(event) for event in base.events}  # a model mention of a caption date is already there
    kept_events = [event for event in _dedupe(events, _event_key) if _event_key(event) not in built]
    kept_arguments = _dedupe(arguments, _argument_key)
    kept_arguments += missed_arguments(base, kept_arguments)
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
