"""Ground truth for the eval: expected outputs and the hand-checked timeline for a case."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import Field

from ratio.schema import DatePrecision, EventType, Frozen, ModuleName

ExpectedKind = Literal["status", "flag", "follow_up", "event"]


class Anchor(Frozen):
    """Exact text in one case document (path relative to the case folder); it must occur once."""

    doc: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ExpectedItem(Frozen):
    """One expected output, matched on module, standard, kind and status, plus anchor overlap.

    ``match="all"`` needs every anchor overlapped by the output's spans; ``"any"`` needs one.
    In must_not_flag, ``status="any"`` forbids every status of that module and standard.
    """

    id: str
    module: ModuleName
    standard_id: str
    kind: ExpectedKind
    status: str
    match: Literal["all", "any"] = "all"
    anchors: tuple[Anchor, ...] = ()
    note: str = ""


class ExpectedFlags(Frozen):
    case_id: str
    synthetic: bool
    expected: tuple[ExpectedItem, ...]
    must_not_flag: tuple[ExpectedItem, ...] = ()


class GoldEvent(Frozen):
    id: str
    type: EventType
    date: dt.date
    precision: DatePrecision
    anchor: Anchor


class GoldTimeline(Frozen):
    case_id: str
    synthetic: bool
    events: tuple[GoldEvent, ...]
