"""Detention orders, read in code and never by the model: the date each order was made, and the date
it authorises detention until.

An order's date is the value of a caption line "Date:", "Order date:" or "Dated:". Its end date is the
written date right after "until" in the last sentence of the order that speaks of detention, custody
or remand ("The detention of the accused is extended until 12 May 2025."): the operative part comes
last. Both are parsed with dateparser, to the day. An order whose dates cannot be read is kept without
them, and the renewal module says what it could not measure.

Every order after the earliest one is a detention extension on the timeline: the event is built here
from the order's own caption, so the Procedural Clock sees it without a model call.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable

from ratio.extraction.dates import find_written_dates, parse_date_text
from ratio.extraction.segment import header_value, sentence_spans
from ratio.schema import DetentionOrder, Document, Event, SourceSpan, stable_id

DATE_LABELS = ("Date", "Order date", "Dated")
DAY_PRECISION = frozenset({"date", "datetime"})
_UNTIL = re.compile(r"\b(?:detention|detained|custody|remand(?:ed)?)\b[^.]{0,160}?\buntil\s+(?:and\s+including\s+)?", re.IGNORECASE)


def _span(doc: Document, start: int, end: int) -> SourceSpan:
    return SourceSpan(doc_id=doc.id, start=start, end=end, text=doc.text[start:end])


def _day(text: str) -> dt.date | None:
    parsed = parse_date_text(text)
    return parsed.value.date() if parsed is not None and parsed.precision in DAY_PRECISION else None


def _order_date(doc: Document) -> tuple[dt.date | None, SourceSpan | None]:
    for label in DATE_LABELS:
        found = header_value(doc.text, label)
        if found is not None and (day := _day(doc.text[found[0] : found[1]])) is not None:
            return day, _span(doc, *found)
    return None, None


def _until(doc: Document) -> tuple[dt.date | None, SourceSpan | None]:
    for start, end in reversed(sentence_spans(doc.text)):
        for match in _UNTIL.finditer(doc.text, start, end):
            written = [d for d in find_written_dates(doc.text, match.end(), end) if d[0] == match.end()]
            if written and (day := _day(doc.text[written[0][0] : written[0][1]])) is not None:
                return day, _span(doc, start, end)
    return None, None


def read_order(doc: Document) -> DetentionOrder:
    date, date_span = _order_date(doc)
    until, until_span = _until(doc)
    return DetentionOrder(doc_id=doc.id, date=date, date_span=date_span, until=until, until_span=until_span)


def read_orders(documents: Iterable[Document]) -> tuple[DetentionOrder, ...]:
    return tuple(read_order(doc) for doc in documents if doc.type == "detention_order")


def extension_events(orders: Iterable[DetentionOrder], documents: Iterable[Document]) -> tuple[Event, ...]:
    """A detention_extension event for every dated order after the earliest one."""
    texts = {doc.id: doc for doc in documents}
    dated = sorted((o for o in orders if o.date is not None), key=lambda o: (o.date, o.doc_id))
    events = []
    for order in dated[1:]:
        doc = texts[order.doc_id]
        line_start = doc.text.rfind("\n", 0, order.date_span.start) + 1
        line_end = doc.text.find("\n", order.date_span.end)
        line = _span(doc, line_start, len(doc.text) if line_end == -1 else line_end)
        events.append(
            Event(
                id=stable_id(doc.id, line.start, "order"),
                type="detention_extension",
                span=line,
                quote_span=line,
                date_span=order.date_span,
                date_text=order.date_span.text,
                parsed_date=dt.datetime.combine(order.date, dt.time()),
                precision="date",
            )
        )
    return tuple(events)
