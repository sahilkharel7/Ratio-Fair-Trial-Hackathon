"""Case record contract shared by the extraction layer, the four modules, the app and the eval.

All models are frozen and use tuples, so a record cannot change after it is built.
Character offsets index ``Document.text``: UTF-8 text with LF line endings and the
SYNTHETIC marker line removed. Document ids are ``"<case_id>/<relative path>"`` so any span
can be resolved across cases (the judge profile links to rulings in other cases).
"""

from __future__ import annotations

import hashlib
import datetime as dt
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

DocType = Literal["monitoring_note", "indictment", "judgment", "detention_order", "transcript"]
EventType = Literal[
    "arrest",
    "first_appearance",
    "counsel_access",
    "charge",
    "detention_extension",
    "hearing",
    "trial_start",
    "verdict",
]
EVENT_TYPES: tuple[str, ...] = get_args(EventType)
DatePrecision = Literal["datetime", "date", "month", "year", "unknown"]
MatchKind = Literal["exact", "normalized", "fuzzy"]
Party = Literal["defense", "prosecution"]
DataProvenance = Literal["synthetic", "public"]
PassageKind = Literal["header", "heading", "body", "signature"]
ModuleName = Literal["absence", "clock", "reuse", "renewal", "judges"]
ReviewStatus = Literal["confirmed", "needs_legal_review"]
EvidenceRole = Literal[
    "supporting",
    "contradicting",
    "context",
    "from_event",
    "to_event",
    "mention",
    "judgment",
    "indictment",
    "argument",
    "ruling",
    "later_order",
    "earlier_order",
    "order_expiry",
    "next_order",
]

# Allowed Flag.status values per module. "No evidence" is deliberately absent: it is a FollowUp.
FLAG_STATUSES: dict[str, frozenset[str]] = {
    "absence": frozenset({"evidence_of_compliance", "evidence_of_violation"}),
    "clock": frozenset({"exceeds_benchmark", "needs_review"}),
    "reuse": frozenset({"verbatim_reuse", "paraphrase_reuse", "charge_wording", "unaddressed_argument"}),
    "renewal": frozenset({"repeated_grounds", "order_gap"}),
    "judges": frozenset({"pattern_warrants_review"}),
}


def stable_id(*parts: object) -> str:
    """Deterministic 16-hex-char id: the same parts give the same id in every process."""
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceSpan(Frozen):
    """Exact text located at [start, end) in one document."""

    doc_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str

    @model_validator(mode="after")
    def _consistent(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("span text length must equal end - start")
        if not self.text.strip():
            raise ValueError("span text must not be blank")
        return self

    @property
    def case_id(self) -> str:
        return self.doc_id.split("/", 1)[0]

    def overlaps(self, other: SourceSpan) -> bool:
        return self.doc_id == other.doc_id and self.start < other.end and other.start < self.end

    def contains(self, other: SourceSpan) -> bool:
        return self.doc_id == other.doc_id and self.start <= other.start and other.end <= self.end


class Evidence(Frozen):
    """A span plus the part it plays in a flag (e.g. from_event vs to_event on the timeline)."""

    role: EvidenceRole
    span: SourceSpan


class Document(Frozen):
    id: str
    case_id: str
    path: str
    type: DocType
    title: str
    text: str
    synthetic: bool
    sha256: str
    date: dt.date | None = None
    date_span: SourceSpan | None = None

    @model_validator(mode="after")
    def _id_matches_case_and_path(self) -> Document:
        if self.id != f"{self.case_id}/{self.path}":
            raise ValueError("document id must be '<case_id>/<path>'")
        return self


class Event(Frozen):
    """A dated procedural event. The Clock only ever uses ``parsed_date`` (parsed in code)."""

    id: str
    type: EventType
    span: SourceSpan
    quote_span: SourceSpan
    date_span: SourceSpan | None = None
    date_text: str | None = None
    model_date: str | None = None
    parsed_date: dt.datetime | None = None
    precision: DatePrecision = "unknown"
    actors: tuple[str, ...] = ()
    match: MatchKind = "exact"
    needs_review: bool = False
    review_reasons: tuple[str, ...] = ()


class Observation(Frozen):
    """One sentence of a monitoring note. ``text`` is always the exact source text."""

    id: str
    text: str
    hearing_date: dt.date | None = None
    span: SourceSpan

    @model_validator(mode="after")
    def _text_is_source(self) -> Observation:
        if self.text != self.span.text:
            raise ValueError("observation text must equal its span text")
        return self


class Argument(Frozen):
    """A party's argument recorded in the notes, stored as its full source sentence."""

    id: str
    party: Party
    text: str
    hearing_date: dt.date | None = None
    span: SourceSpan
    quote_span: SourceSpan | None = None
    match: MatchKind = "exact"
    needs_review: bool = False

    @model_validator(mode="after")
    def _text_is_source(self) -> Argument:
        if self.text != self.span.text:
            raise ValueError("argument text must equal its span text")
        return self


class Citation(Frozen):
    id: str
    text: str
    span: SourceSpan


class Passage(Frozen):
    """A sentence of a judgment, indictment or order, with the heading of its section."""

    id: str
    doc_id: str
    index: int = Field(ge=0)
    section: str | None = None
    chapter: str | None = None  # the top-level heading the section belongs to
    kind: PassageKind = "body"
    span: SourceSpan


class Ruling(Frozen):
    """A coded ruling (ruling_codes.yaml) attributed to the judge named in the source."""

    id: str
    case_id: str
    code: str
    judge_name: str
    date: dt.date | None = None
    value: float | None = None
    span: SourceSpan


class DetentionOrder(Frozen):
    """A detention order read in code (never by the model): the date it was made and the date it
    authorises detention until, each with the exact text it was parsed from."""

    doc_id: str
    date: dt.date | None = None
    date_span: SourceSpan | None = None
    until: dt.date | None = None
    until_span: SourceSpan | None = None  # the sentence of the order that sets the end date

    @model_validator(mode="after")
    def _dated_from_source(self) -> DetentionOrder:
        if (self.date is None) != (self.date_span is None) or (self.until is None) != (self.until_span is None):
            raise ValueError("an order date needs the text it was parsed from")
        return self


class AliasDecision(Frozen):
    """A person's decision on a name from the manual confirmation list (alias_decisions.yaml)."""

    case_id: str = Field(min_length=1)
    name: str = Field(min_length=1)  # exactly as written in that case's rulings
    same_as: str | None = None  # the registered judge at the same court it belongs to; None: a different judge
    synthetic: bool | None = None  # the kind of data of the file it was recorded in (set by the loader)


class CaseMeta(Frozen):
    case_id: str
    title: str
    court: str
    charge_type: str
    presiding_judge: str | None = None
    presiding_judge_span: SourceSpan | None = None
    data_provenance: DataProvenance
    source_note: str | None = None
    synthetic: bool

    @model_validator(mode="after")
    def _provenance_declared(self) -> CaseMeta:
        if self.synthetic != (self.data_provenance == "synthetic"):
            raise ValueError("synthetic flag must match data_provenance")
        if self.data_provenance == "public" and not (self.source_note or "").strip():
            raise ValueError("public material needs a source_note citing where it was published")
        return self


class CaseRecord(Frozen):
    meta: CaseMeta
    documents: tuple[Document, ...]
    events: tuple[Event, ...] = ()
    observations: tuple[Observation, ...] = ()
    arguments: tuple[Argument, ...] = ()
    citations: tuple[Citation, ...] = ()
    passages: tuple[Passage, ...] = ()
    rulings: tuple[Ruling, ...] = ()
    orders: tuple[DetentionOrder, ...] = ()

    @model_validator(mode="after")
    def _documents_belong_to_case(self) -> CaseRecord:
        ids = [doc.id for doc in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate document ids")
        if any(doc.case_id != self.meta.case_id for doc in self.documents):
            raise ValueError("every document must belong to the record's case")
        return self

    @property
    def case_id(self) -> str:
        return self.meta.case_id

    def document(self, doc_id: str) -> Document:
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        raise KeyError(doc_id)

    def documents_of_type(self, doc_type: str) -> tuple[Document, ...]:
        return tuple(doc for doc in self.documents if doc.type == doc_type)

    def passages_of(self, doc_id: str) -> tuple[Passage, ...]:
        return tuple(passage for passage in self.passages if passage.doc_id == doc_id)


class Flag(Frozen):
    """A module finding. It always carries at least one evidence span."""

    id: str
    case_id: str | None
    module: ModuleName
    standard_id: str
    standard_label: str
    status: str
    message: str
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    citation: str | None = None
    review_status: ReviewStatus = "needs_legal_review"
    model_note: str | None = None

    @model_validator(mode="after")
    def _status_in_vocabulary(self) -> Flag:
        if self.status not in FLAG_STATUSES[self.module]:
            raise ValueError(f"status {self.status!r} is not valid for module {self.module!r}")
        if self.status == "exceeds_benchmark" and self.review_status != "confirmed":
            raise ValueError("only a confirmed benchmark can be exceeded (hard rule 5)")
        return self

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(item.span for item in self.evidence)


class FollowUp(Frozen):
    """A question for the monitor where the record holds no evidence. Not a finding."""

    id: str
    case_id: str
    rubric_id: str
    question: str
    uncovered: tuple[str, ...] = ()
    context: tuple[Evidence, ...] = ()
