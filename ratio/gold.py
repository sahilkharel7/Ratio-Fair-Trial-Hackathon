"""Build the demo's ground-truth files from the hand-written annotations (gold/annotations.yaml).

    mock_record.json     the record an ideal extractor would produce; modules are built against it
    expected_flags.json  what the eval expects the modules to find, and must not find
    gold_timeline.json   one hand-checked date per real event, for the date-accuracy metric

Run ``python -m ratio.gold`` after editing the demo case or its annotations.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from ratio.expected import Anchor, ExpectedFlags, ExpectedItem, GoldEvent, GoldTimeline
from ratio.extraction.align import find_all, sentence_bounds
from ratio.extraction.build import load_case
from ratio.extraction.dates import parse_date_text
from ratio.schema import (
    Argument,
    CaseRecord,
    DatePrecision,
    Document,
    Event,
    EventType,
    Frozen,
    Party,
    SourceSpan,
    stable_id,
)
from ratio.paths import DEMO_CASE_DIR, GOLD_DIR


class GoldError(ValueError):
    """The annotations do not match the case documents."""


class GoldEventEntry(Frozen):
    id: str
    type: EventType
    doc: str
    quote: str
    date_text: str
    date: dt.date
    precision: DatePrecision


class GoldArgumentEntry(Frozen):
    id: str
    party: Party
    doc: str
    quote: str


class Annotations(Frozen):
    synthetic: bool
    case_id: str
    events: tuple[GoldEventEntry, ...]
    arguments: tuple[GoldArgumentEntry, ...] = ()
    expected: tuple[ExpectedItem, ...]
    must_not_flag: tuple[ExpectedItem, ...] = ()


def load_annotations(path: Path) -> Annotations:
    try:
        return Annotations.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise GoldError(f"{path} is invalid: {exc}") from exc


def _document(record: CaseRecord, relative_path: str) -> Document:
    try:
        return record.document(f"{record.case_id}/{relative_path}")
    except KeyError as exc:
        raise GoldError(f"no document {relative_path!r} in case {record.case_id}") from exc


def resolve_anchor(record: CaseRecord, anchor: Anchor) -> SourceSpan:
    """The unique span of an anchor quote; raises GoldError if it is missing or ambiguous."""
    doc = _document(record, anchor.doc)
    positions = find_all(doc.text, anchor.quote)
    if len(positions) != 1:
        raise GoldError(f"{anchor.doc}: quote found {len(positions)} times: {anchor.quote!r}")
    start = positions[0]
    return SourceSpan(doc_id=doc.id, start=start, end=start + len(anchor.quote), text=anchor.quote)


def _sentence(doc: Document, span: SourceSpan) -> SourceSpan:
    start, end = sentence_bounds(doc.text, span.start, span.end)
    return SourceSpan(doc_id=doc.id, start=start, end=end, text=doc.text[start:end])


def gold_event(record: CaseRecord, entry: GoldEventEntry) -> Event:
    doc = _document(record, entry.doc)
    quote = resolve_anchor(record, Anchor(doc=entry.doc, quote=entry.quote))
    date_offset = doc.text.find(entry.date_text, quote.start, quote.end)
    if date_offset == -1:
        raise GoldError(f"{entry.id}: date text {entry.date_text!r} is not inside its quote")
    date_span = SourceSpan(
        doc_id=doc.id, start=date_offset, end=date_offset + len(entry.date_text), text=entry.date_text
    )
    parsed = parse_date_text(entry.date_text, base=dt.datetime.combine(entry.date, dt.time()))
    if parsed is None or parsed.value.date() != entry.date:
        raise GoldError(f"{entry.id}: date text {entry.date_text!r} does not parse to {entry.date}")
    return Event(
        id=stable_id(doc.id, quote.start, entry.type, date_offset, "event"),
        type=entry.type,
        span=_sentence(doc, quote),
        quote_span=quote,
        date_span=date_span,
        date_text=entry.date_text,
        model_date=entry.date.isoformat(),
        parsed_date=parsed.value,
        precision=entry.precision,
    )


def gold_argument(record: CaseRecord, entry: GoldArgumentEntry) -> Argument:
    doc = _document(record, entry.doc)
    quote = resolve_anchor(record, Anchor(doc=entry.doc, quote=entry.quote))
    sentence = _sentence(doc, quote)
    return Argument(
        id=stable_id(doc.id, quote.start, entry.party, "argument"),
        party=entry.party,
        text=sentence.text,
        hearing_date=doc.date,
        span=sentence,
        quote_span=quote,
    )


def build_mock_record(case_dir: Path, annotations: Annotations) -> CaseRecord:
    base = load_case(case_dir)
    if base.case_id != annotations.case_id:
        raise GoldError(f"annotations are for {annotations.case_id}, case folder is {base.case_id}")
    events = tuple(gold_event(base, entry) for entry in annotations.events)
    arguments = tuple(gold_argument(base, entry) for entry in annotations.arguments)
    for item in annotations.expected + annotations.must_not_flag:
        for anchor in item.anchors:
            resolve_anchor(base, anchor)
    return CaseRecord(
        meta=base.meta,
        documents=base.documents,
        events=base.events + events,
        observations=base.observations,
        arguments=arguments,
        citations=base.citations,
        passages=base.passages,
        rulings=base.rulings,
    )


def build_expected(annotations: Annotations) -> ExpectedFlags:
    return ExpectedFlags(
        case_id=annotations.case_id,
        synthetic=annotations.synthetic,
        expected=annotations.expected,
        must_not_flag=annotations.must_not_flag,
    )


def build_gold_timeline(annotations: Annotations) -> GoldTimeline:
    """One gold event per (type, date): the first annotation of that event is its anchor."""
    seen: set[tuple[str, dt.date]] = set()
    events = []
    for entry in annotations.events:
        key = (entry.type, entry.date)
        if key in seen:
            continue
        seen.add(key)
        events.append(
            GoldEvent(
                id=entry.id,
                type=entry.type,
                date=entry.date,
                precision=entry.precision,
                anchor=Anchor(doc=entry.doc, quote=entry.quote),
            )
        )
    return GoldTimeline(case_id=annotations.case_id, synthetic=annotations.synthetic, events=tuple(events))


def _dump(model: Frozen) -> str:
    return json.dumps(json.loads(model.model_dump_json()), indent=2, ensure_ascii=False) + "\n"


def gold_files(case_dir: Path = DEMO_CASE_DIR, gold_dir: Path = GOLD_DIR) -> dict[str, str]:
    """File name -> JSON text for every generated gold file (nothing is written)."""
    annotations = load_annotations(gold_dir / "annotations.yaml")
    return {
        "mock_record.json": _dump(build_mock_record(case_dir, annotations)),
        "expected_flags.json": _dump(build_expected(annotations)),
        "gold_timeline.json": _dump(build_gold_timeline(annotations)),
    }


def write_gold_files(case_dir: Path = DEMO_CASE_DIR, gold_dir: Path = GOLD_DIR) -> list[Path]:
    written = []
    for name, text in gold_files(case_dir, gold_dir).items():
        path = gold_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for path in write_gold_files():
        print(f"wrote {path}")
