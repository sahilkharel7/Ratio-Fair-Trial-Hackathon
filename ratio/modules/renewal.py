"""Module 5: Detention renewals (ICCPR Art. 9(1), 9(3)).

Detention pending trial must be re-examined each time it is extended. The case's detention orders
are put in date order (their dates were read in code when the record was built), and for each order:

1. Grounds: the court's own reasoning in the order, found as for a judgment (reuse_exclusions): the
   caption, quoted law, the prosecutor's request and the operative part are left out.
2. Repetition: every grounds passage of a later order is compared with the grounds of all the orders
   before it, with the Reasoning Reuse Detector's matching (5-gram containment and MinHash for copies,
   embeddings for close paraphrases). The share of the order's grounds traceable to earlier orders is
   measured in code, and an order at or above settings.renewal.repeated_share is flagged.
3. Gaps: when the next order is dated more than a day after an order's end date, the whole days in
   between are covered by no order in the record, and are flagged.

The model is never asked anything here. The share threshold is an engineering heuristic, not a legal
standard: a flag says where to look, and whether detention was properly re-examined is for the lawyer.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence

from ratio.context import AnalysisContext
from ratio.messages import render
from ratio.modules.reuse import Source, find_pairs, merge_ranges
from ratio.modules.reuse_exclusions import PassageRules, classify, compile_rules
from ratio.results import ExcludedPassage, OrderGap, OrderSummary, RenewalPair, RenewalResult
from ratio.schema import CaseRecord, DetentionOrder, Evidence, Flag, Passage, stable_id

REVIEW_STANDARD = "renewal_review"
GAP_STANDARD = "renewal_gap"
_WORD = re.compile(r"\w+")


def day_text(day: dt.date) -> str:
    return f"{day.day} {day:%B %Y}"


def _grounds(record: CaseRecord, doc_id: str, rules: PassageRules) -> tuple[list[Passage], tuple[ExcludedPassage, ...]]:
    document = record.document(doc_id)
    passages = record.passages_of(doc_id)
    reasons = classify(passages, document.text, record.citations, rules)
    grounds = [p for p in passages if p.id in reasons and reasons[p.id] is None]
    excluded = tuple(
        ExcludedPassage(passage_id=p.id, span=p.span, reason=reason) for p in passages if (reason := reasons.get(p.id)) is not None
    )
    return grounds, excluded


def _pairs(record: CaseRecord, grounds: Sequence[Passage], earlier: Sequence[Passage], ctx: AnalysisContext) -> tuple[RenewalPair, ...]:
    if not grounds or not earlier:
        return ()
    found = find_pairs(record, grounds, [Source(passage, charge=False) for passage in earlier], ctx)
    return tuple(
        RenewalPair(
            id=stable_id(record.case_id, "renewal", pair.kind, pair.judgment.doc_id, pair.judgment.start, pair.indictment.doc_id, pair.indictment.start),
            kind=pair.kind,
            later=pair.judgment,
            earlier=pair.indictment,
            later_ranges=pair.judgment_ranges,
            earlier_ranges=pair.indictment_ranges,
            containment=pair.passage_containment,
            cosine=pair.cosine,
        )
        for pair in found
    )


def repeated_chars(pairs: Sequence[RenewalPair]) -> int:
    """Characters of the later order traceable to earlier ones: the matched 5-gram text of copies,
    the whole passage of a paraphrase."""
    copied = merge_ranges((r.start, r.end) for pair in pairs if pair.kind == "verbatim" for r in pair.later_ranges)
    copied_passages = {pair.later for pair in pairs if pair.kind == "verbatim"}
    paraphrased = {pair.later for pair in pairs if pair.kind == "paraphrase" and pair.later not in copied_passages}
    return sum(r.end - r.start for r in copied) + sum(len(span.text) for span in paraphrased)


def _repeated_flag(record: CaseRecord, summary: OrderSummary, ctx: AnalysisContext) -> Flag:
    config = ctx.config
    standard = config.standard(REVIEW_STANDARD)
    sources = ", ".join(dict.fromkeys(record.document(pair.earlier.doc_id).title for pair in summary.pairs))
    message = render(
        config.messages, "renewal_repeated",
        percent=round(100 * summary.share_repeated), sources=sources, new=summary.passages_new, compared=summary.passages_compared,
    )  # fmt: skip
    later = [Evidence(role="later_order", span=span) for span in dict.fromkeys(pair.later for pair in summary.pairs)]
    earlier = [Evidence(role="earlier_order", span=span) for span in dict.fromkeys(pair.earlier for pair in summary.pairs)]
    return Flag(
        id=stable_id(record.case_id, "renewal", summary.doc_id, "repeated"),
        case_id=record.case_id,
        module="renewal",
        standard_id=REVIEW_STANDARD,
        standard_label=standard.label,
        status="repeated_grounds",
        message=message,
        evidence=tuple(later + earlier),
        citation=standard.citation,
        review_status=standard.review_status,
    )


def _summary(record: CaseRecord, order: DetentionOrder, previous: DetentionOrder | None, earlier: list[Passage], ctx: AnalysisContext, rules: PassageRules) -> OrderSummary:
    grounds, excluded = _grounds(record, order.doc_id, rules)
    pairs = _pairs(record, grounds, earlier, ctx) if previous is not None else ()
    minimum = ctx.config.settings.reuse.min_passage_words
    compared = [p for p in grounds if len(_WORD.findall(p.span.text)) >= minimum]
    matched = {pair.later for pair in pairs}
    chars = sum(len(p.span.text) for p in grounds)
    share = None
    if previous is not None and chars:
        share = min(1.0, repeated_chars(pairs) / chars)
    return OrderSummary(
        doc_id=order.doc_id,
        title=record.document(order.doc_id).title,
        date=order.date,
        until=order.until,
        date_span=order.date_span,
        until_span=order.until_span,
        days_since_previous=(order.date - previous.date).days if previous is not None else None,
        grounds_passage_ids=tuple(p.id for p in grounds),
        grounds_chars=chars,
        repeated_chars=repeated_chars(pairs),
        share_repeated=round(share, 4) if share is not None else None,
        passages_compared=len(compared) if previous is not None else 0,
        passages_new=sum(p.span not in matched for p in compared) if previous is not None else 0,
        pairs=pairs,
        excluded=excluded,
    )


def _gap(record: CaseRecord, earlier: DetentionOrder, later: DetentionOrder, ctx: AnalysisContext) -> tuple[OrderGap, Flag] | None:
    if earlier.until is None or later.date is None:
        return None
    days = (later.date - earlier.until).days - 1  # whole days strictly between the end date and the next order
    if days < 1:
        return None
    standard = ctx.config.standard(GAP_STANDARD)
    evidence = (Evidence(role="order_expiry", span=earlier.until_span), Evidence(role="next_order", span=later.date_span))
    message = render(
        ctx.config.messages, "renewal_gap",
        earlier=day_text(earlier.date) if earlier.date else record.document(earlier.doc_id).title,
        until=day_text(earlier.until), next=day_text(later.date), days=days,
    )  # fmt: skip
    flag = Flag(
        id=stable_id(record.case_id, "renewal", earlier.doc_id, later.doc_id, "gap"),
        case_id=record.case_id,
        module="renewal",
        standard_id=GAP_STANDARD,
        standard_label=standard.label,
        status="order_gap",
        message=message,
        evidence=evidence,
        citation=standard.citation,
        review_status=standard.review_status,
    )
    gap = OrderGap(
        id=stable_id(record.case_id, "renewal-gap", earlier.doc_id, later.doc_id),
        earlier_doc_id=earlier.doc_id, later_doc_id=later.doc_id, until=earlier.until, next_date=later.date,
        days=days, evidence=evidence, flag_id=flag.id,
    )  # fmt: skip
    return gap, flag


def run(record: CaseRecord, ctx: AnalysisContext) -> RenewalResult:
    if not record.orders:
        return RenewalResult()
    rules = compile_rules(ctx.config.settings.reuse)
    titles = {order.doc_id: record.document(order.doc_id).title for order in record.orders}
    notes = [f"{titles[o.doc_id]}: no date was read from its caption, so it is not compared." for o in record.orders if o.date is None]
    dated = sorted((o for o in record.orders if o.date is not None), key=lambda o: (o.date, o.doc_id))
    summaries: list[OrderSummary] = []
    gaps: list[OrderGap] = []
    flags: list[Flag] = []
    earlier: list[Passage] = []
    previous: DetentionOrder | None = None
    for order in dated:
        summary = _summary(record, order, previous, earlier, ctx, rules)
        if summary.share_repeated is not None and summary.pairs and summary.share_repeated >= ctx.config.settings.renewal.repeated_share:
            flag = _repeated_flag(record, summary, ctx)
            summary = summary.model_copy(update={"flag_id": flag.id})
            flags.append(flag)
        if previous is not None:
            if previous.until is None:
                notes.append(f"{titles[previous.doc_id]}: no end date was read, so the time until the next order is not measured.")
            elif (found := _gap(record, previous, order, ctx)) is not None:
                gaps.append(found[0])
                flags.append(found[1])
        summaries.append(summary)
        grounds = set(summary.grounds_passage_ids)
        earlier += [p for p in record.passages_of(order.doc_id) if p.id in grounds]
        previous = order
    if len(dated) < 2:
        notes.append("Fewer than two dated orders: there is nothing to compare.")
    return RenewalResult(orders=tuple(summaries), gaps=tuple(gaps), notes=tuple(notes), flags=tuple(flags))
