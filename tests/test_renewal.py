"""Detention renewals: orders read in code (dates and end dates with their exact source text), grounds
compared with every earlier order, the share of repeated grounds, gaps between orders, the provenance
check, and the planted demo orders (one copied renewal, one gap, and the traps around them)."""

import datetime as dt

import pytest

from ratio.config import load_config
from ratio.context import AnalysisContext
from ratio.extraction.build import load_case
from ratio.extraction.orders import extension_events, read_order
from ratio.modules import renewal
from ratio.paths import DEMO_CASE_DIR, MINILM_DIR
from ratio.provenance import check_renewal, resolver_for, span_is_valid
from ratio.schema import Evidence, SourceSpan
from ratio.testing import FakeEmbedder, FakeLLM, make_record

CONFIG = load_config()
needs_model = pytest.mark.skipif(not (MINILM_DIR / "modules.json").exists(), reason="run scripts/fetch_models.py once")

GROUNDS = (
    "The accused has relatives abroad and travelled there twice last year, so there is a risk that he will flee.",
    "The examination of the seized telephone has not yet been completed and its results are still awaited.",
)
NEW_GROUND = "Two witnesses have reported being approached by acquaintances of the accused since the last order was made."


def order_text(date: str, until: str | None, grounds: tuple[str, ...], *, label: str = "Date") -> str:
    caption = f"DETENTION ORDER\nCase no. T-1\n{label}: {date}\n\n" if date else "DETENTION ORDER\nCase no. T-1\n\n"
    law = 'II. APPLICABLE LAW\n\nArticle 167(1) of the Code of Criminal Procedure provides: "Detention may be ordered where there is a risk of flight."\n\n'
    decision = f"IV. DECISION\n\nThe detention of the accused is extended until {until}.\n" if until else "IV. DECISION\n\nThe detention of the accused is extended.\n"
    request = "I. REQUEST\n\nThe Public Prosecutor requests that the detention of the accused be extended.\n\n"
    return caption + request + law + "III. GROUNDS\n\n" + " ".join(grounds) + "\n\n" + decision


def record_of(*orders: tuple[str, str]):
    return make_record([(path, "detention_order", text) for path, text in orders])


def run(record, config=CONFIG):
    return renewal.run(record, AnalysisContext(llm=FakeLLM(lambda **_: {}), embedder=FakeEmbedder(), config=config))


# --- reading orders in code -----------------------------------------------------------------


def test_an_order_date_comes_from_its_caption_and_its_end_date_from_the_operative_part():
    record = record_of(("o1.txt", order_text("19 February 2025", "19 March 2025", GROUNDS)))
    [order] = record.orders
    assert (order.date, order.until) == (dt.date(2025, 2, 19), dt.date(2025, 3, 19))
    assert order.date_span.text == "19 February 2025"
    assert order.until_span.text == "The detention of the accused is extended until 19 March 2025."
    document = record.documents[0]
    for span in (order.date_span, order.until_span):
        assert document.text[span.start : span.end] == span.text


@pytest.mark.parametrize("label", ["Date", "Order date", "Dated"])
def test_every_date_label_is_read(label):
    [order] = record_of(("o1.txt", order_text("3 March 2025", None, GROUNDS, label=label))).orders
    assert order.date == dt.date(2025, 3, 3) and order.until is None


def test_until_counts_only_when_the_sentence_speaks_of_detention():
    text = order_text("3 March 2025", None, GROUNDS) + "\nThe hearing is adjourned until 9 April 2025.\n"
    [order] = record_of(("o1.txt", text)).orders
    assert order.until is None


def test_an_order_without_a_caption_date_is_kept_undated():
    [order] = record_of(("o1.txt", order_text("", "19 March 2025", GROUNDS))).orders
    assert order.date is None and order.date_span is None and order.until == dt.date(2025, 3, 19)


def test_every_order_after_the_earliest_is_an_extension_event_built_in_code():
    record = record_of(
        ("b.txt", order_text("19 March 2025", "12 May 2025", GROUNDS)),
        ("a.txt", order_text("19 February 2025", "19 March 2025", GROUNDS)),
    )
    events = [e for e in record.events if e.type == "detention_extension"]
    assert [(e.parsed_date.date(), e.span.text) for e in events] == [(dt.date(2025, 3, 19), "Date: 19 March 2025")]
    assert extension_events(record.orders, record.documents) == tuple(events)


def test_read_order_ignores_other_document_types():
    record = make_record([("judgment.txt", "judgment", "JUDGMENT\nDate: 1 June 2025\n\nThe accused is detained until 2 June 2025.")])
    assert record.orders == ()


# --- the module -----------------------------------------------------------------------------


def test_a_renewal_that_copies_the_earlier_grounds_is_flagged_and_one_with_new_grounds_is_not():
    record = record_of(
        ("o1.txt", order_text("19 February 2025", "19 March 2025", GROUNDS)),
        ("o2.txt", order_text("19 March 2025", "19 May 2025", GROUNDS)),
        ("o3.txt", order_text("19 May 2025", "19 July 2025", (GROUNDS[0], NEW_GROUND))),
    )
    result = run(record)
    first, copied, fresh = result.orders
    assert first.share_repeated is None and first.flag_id is None  # nothing before it
    assert copied.share_repeated >= 0.95 and (copied.passages_new, copied.passages_compared) == (0, 2)
    assert copied.flag_id is not None and fresh.flag_id is None
    assert fresh.passages_new == 1 and fresh.share_repeated < CONFIG.settings.renewal.repeated_share
    assert [f.status for f in result.flags] == ["repeated_grounds"]
    assert copied.days_since_previous == 28 and result.gaps == ()
    excluded = {e.reason for e in copied.excluded}
    assert {"party_position", "statute_quote", "non_reasoning_section"} <= excluded  # request, law, decision


def test_the_threshold_is_a_setting():
    record = record_of(
        ("o1.txt", order_text("19 February 2025", "19 March 2025", GROUNDS)),
        ("o2.txt", order_text("19 March 2025", "19 May 2025", (*GROUNDS, NEW_GROUND))),
    )
    share = run(record).orders[1].share_repeated
    assert CONFIG.settings.renewal.repeated_share <= share < 0.9  # two of three grounds repeated
    strict = CONFIG.model_copy(update={"settings": CONFIG.settings.model_copy(update={"renewal": CONFIG.settings.renewal.model_copy(update={"repeated_share": 0.9})})})
    assert len(run(record).flags) == 1 and run(record, strict).flags == ()


def test_whole_days_with_no_order_are_a_gap_and_a_same_day_renewal_is_not():
    record = record_of(
        ("o1.txt", order_text("1 March 2025", "31 March 2025", GROUNDS)),
        ("o2.txt", order_text("1 April 2025", "30 April 2025", (NEW_GROUND,))),  # the day after: continuous
        ("o3.txt", order_text("4 May 2025", "4 June 2025", (GROUNDS[1] + " Its results arrived on 2 May 2025.",))),
    )
    result = run(record)
    [gap] = result.gaps
    assert (gap.until, gap.next_date, gap.days) == (dt.date(2025, 4, 30), dt.date(2025, 5, 4), 3)
    flag = next(f for f in result.flags if f.id == gap.flag_id)
    assert flag.status == "order_gap" and [e.role for e in flag.evidence] == ["order_expiry", "next_order"]
    assert "Whole days with no order in the record: 3." in flag.message


def test_what_cannot_be_measured_is_said():
    result = run(record_of(("o1.txt", order_text("", "19 March 2025", GROUNDS)), ("o2.txt", order_text("19 March 2025", None, GROUNDS))))
    assert any("no date was read" in note for note in result.notes)
    assert any("Fewer than two dated orders" in note for note in result.notes)
    no_end = run(record_of(("o1.txt", order_text("1 March 2025", None, GROUNDS)), ("o2.txt", order_text("9 April 2025", None, GROUNDS))))
    assert any("no end date was read" in note for note in no_end.notes) and no_end.gaps == ()
    assert run(make_record([("judgment.txt", "judgment", "JUDGMENT\n\nThe Court finds.")])).orders == ()


def test_a_flag_whose_source_text_is_not_found_is_dropped_with_its_measure():
    record = record_of(
        ("o1.txt", order_text("19 February 2025", "19 March 2025", GROUNDS)),
        ("o2.txt", order_text("19 March 2025", "19 May 2025", GROUNDS)),
    )
    result = run(record)
    [flag] = result.flags
    forged = SourceSpan(doc_id=record.documents[1].id, start=0, end=5, text="FORGE")
    tampered = result.model_copy(update={"flags": (flag.model_copy(update={"evidence": (Evidence(role="later_order", span=forged),)}),)})
    dropped: list[str] = []
    checked = check_renewal(tampered, resolver_for([record]), dropped)
    assert checked.flags == () and len(dropped) == 1
    assert checked.orders[1].flag_id is None and checked.orders[1].share_repeated is None and checked.orders[1].pairs == ()
    assert check_renewal(result, resolver_for([record]), []) == result


# --- the planted demo orders ----------------------------------------------------------------


@pytest.fixture(scope="module")
def demo():
    from ratio.embeddings import MiniLMEmbedder

    record = load_case(DEMO_CASE_DIR)
    ctx = AnalysisContext(llm=FakeLLM(lambda **_: {}), embedder=MiniLMEmbedder(), config=CONFIG)
    return record, renewal.run(record, ctx)


@needs_model
@pytest.mark.embed
def test_the_demo_third_order_copies_the_second_and_follows_a_one_day_gap(demo):
    record, result = demo
    first, second, third = result.orders
    assert [o.date for o in result.orders] == [dt.date(2025, 2, 19), dt.date(2025, 3, 19), dt.date(2025, 5, 14)]
    assert second.flag_id is None and second.share_repeated < 0.3 and (second.passages_new, second.passages_compared) == (3, 4)
    assert third.flag_id is not None and third.share_repeated > 0.95 and (third.passages_new, third.passages_compared) == (0, 3)
    assert {pair.earlier.doc_id for pair in third.pairs} == {"venn-2025/orders/detention_order_2.txt"}
    [gap] = result.gaps
    assert (gap.until, gap.next_date, gap.days) == (dt.date(2025, 5, 12), dt.date(2025, 5, 14), 1)
    resolve = resolver_for([record])
    assert all(span_is_valid(span, resolve) for flag in result.flags for span in flag.spans)
    assert [f.id for f in result.flags] == [f.id for f in renewal.run(record, AnalysisContext(llm=FakeLLM(lambda **_: {}), embedder=FakeEmbedder(), config=CONFIG)).flags]  # stable ids
