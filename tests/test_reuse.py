"""Reuse Detector: verbatim and paraphrase pairs, the score, and unaddressed defence arguments.

Demo-level tests use the real MiniLM model (the paraphrase threshold was measured with it). A
rule-based "careful reader" stands in for the LLM; the real model's replies are tested on the
replayed demo cache (test_demo_story).
"""

import json
import re

import pytest

from ratio.config import load_config
from ratio.context import AnalysisContext
from ratio.embeddings import MiniLMEmbedder, cosine_matrix
from ratio.expected import ExpectedFlags
from ratio.gold import resolve_anchor
from ratio.messages import REMOVED, contains_blocked_term
from ratio.modules import reuse
from ratio.modules.reuse_prompts import ArgumentReply
from ratio.paths import GOLD_DIR, MINILM_DIR
from ratio.provenance import resolver_for, span_is_valid
from ratio.schema import CaseRecord
from ratio.testing import FakeEmbedder, FakeLLM, make_record, with_argument

CONFIG = load_config()
SETTINGS = CONFIG.settings.reuse
MOCK = CaseRecord.model_validate_json((GOLD_DIR / "mock_record.json").read_text(encoding="utf-8"))
EXPECTED = ExpectedFlags.model_validate(json.loads((GOLD_DIR / "expected_flags.json").read_text(encoding="utf-8")))
INDEX = {passage.id: passage.index for passage in MOCK.passages}
needs_model = pytest.mark.skipif(not (MINILM_DIR / "modules.json").exists(), reason="run scripts/fetch_models.py once")

INDICTMENT = "INDICTMENT\nCase no. T-1\n\n"
JUDGMENT = "JUDGMENT\nCase no. T-1\n\n"
COPIED = "The accused moved the funds to a private account in Zurich."
INDEPENDENT = "The witness was credible and consistent throughout."


def careful_reader(system, user, schema, purpose):
    """Answers as a careful lawyer would: only the retroactivity argument is dealt with."""
    assert schema is ArgumentReply and purpose == "argument_check"
    argument = re.search(r'^Defence argument \(trial monitoring note\): "(.*)"$', user, re.MULTILINE).group(1)
    passages = dict(re.findall(r"^(P\d+): (.*)$", user, re.MULTILINE))
    answers = ("defence argument that the first article", "argument is therefore rejected")
    responding = [pid for pid, text in passages.items() if "amendment" in argument and any(a in text for a in answers)]
    return {"note": "Checked every passage.", "responding": responding}


def run(record=MOCK, responder=careful_reader, embedder=None, config=CONFIG):
    llm = FakeLLM(responder)
    return reuse.run(record, AnalysisContext(llm=llm, embedder=embedder or FakeEmbedder(), config=config)), llm


def pairs_by_index(result) -> set[tuple[str, int, int]]:
    def index(span):
        return next(p.index for p in MOCK.passages if p.span == span)

    return {(pair.kind, index(pair.judgment), index(pair.indictment)) for pair in result.pairs}


def small_case(judgment_body: str, indictment_body: str | None = COPIED + "\n") -> CaseRecord:
    documents = [("judgment.txt", "judgment", JUDGMENT + judgment_body)]
    if indictment_body is not None:
        documents.append(("indictment.txt", "indictment", INDICTMENT + indictment_body))
    return make_record(documents)


@pytest.fixture(scope="module")
def demo():
    if not (MINILM_DIR / "modules.json").exists():
        pytest.skip("run scripts/fetch_models.py once")
    return run(embedder=MiniLMEmbedder())


# --- the planted demo issues ------------------------------------------------------------------


@pytest.mark.embed
@needs_model
def test_planted_copies_and_the_paraphrase_are_the_only_pairs(demo):
    result, _ = demo
    assert pairs_by_index(result) == {
        ("verbatim", 21, 13),
        ("verbatim", 22, 14),
        ("verbatim", 23, 15),
        ("paraphrase", 24, 16),
        ("verbatim", 29, 17),  # "The Court finds that" + a copied indictment sentence
    }
    partial = next(pair for pair in result.pairs if pair.judgment.text.startswith("The Court finds"))
    assert (partial.jaccard, partial.containment) == (0.8, 0.8)


@pytest.mark.embed
@needs_model
def test_must_not_flag_passages_are_never_flagged(demo):
    result, _ = demo
    forbidden = [item for item in EXPECTED.must_not_flag if item.module == "reuse" and item.standard_id == "reasoning_reuse"]
    assert len(forbidden) == 5
    for item in forbidden:
        anchors = [resolve_anchor(MOCK, anchor) for anchor in item.anchors]
        hits = [flag for flag in result.flags for span in flag.spans for anchor in anchors if span.overlaps(anchor)]
        assert not hits, item.id


@pytest.mark.embed
@needs_model
def test_demo_score_is_in_the_designed_band(demo):
    result, _ = demo
    assert result.reasoning_chars == 1528  # passages 21-30 of the judgment
    assert result.paraphrase_chars == 191  # the whole paraphrased passage 24
    assert 0.55 <= result.score <= 0.65
    assert result.score == pytest.approx((result.verbatim_chars + result.paraphrase_chars) / result.reasoning_chars)


@pytest.mark.embed
@needs_model
def test_paraphrase_threshold_has_a_margin_on_both_sides():
    embedder = MiniLMEmbedder()
    passages = {passage.index: passage for passage in MOCK.passages if passage.doc_id.endswith("judgment.txt")}
    sources = [p for p in MOCK.passages if p.doc_id.endswith("indictment.txt") and p.kind == "body" and p.index != 21]
    source_vectors = embedder.encode([p.span.text for p in sources])

    def best(index: int) -> float:
        return float(cosine_matrix(embedder.encode([passages[index].span.text]), source_vectors).max())

    threshold = SETTINGS.paraphrase_cosine
    assert best(24) >= threshold + 0.05  # the planted paraphrase (0.756)
    hard_negatives = [best(index) for index in (25, 26, 27, 28, 30)]  # reasoning passages that copy nothing
    assert max(hard_negatives) <= threshold - 0.15  # best is 0.523


@pytest.mark.embed
@needs_model
def test_lsh_proposes_every_pair_the_exact_check_would_confirm(demo):
    result, _ = demo
    reasoning = [p for p in MOCK.passages if p.id in result.reasoning_passage_ids]
    sources = [p for p in MOCK.passages if p.doc_id.endswith("indictment.txt") and p.kind == "body"]
    items = [reuse.shingle(p, SETTINGS.shingle_size) for p in reasoning]
    targets = [reuse.shingle(p, SETTINGS.shingle_size) for p in sources]
    proposed = reuse.lsh_candidates(items, targets, SETTINGS)
    confirmed = 0
    for item in items:
        for target in targets:
            shared = item.shingles.keys() & target.shingles.keys()
            if not shared:
                continue
            jaccard = len(shared) / len(item.shingles.keys() | target.shingles.keys())
            containment = len(shared) / len(item.shingles)
            if jaccard >= SETTINGS.verbatim_jaccard or containment >= SETTINGS.verbatim_containment:
                confirmed += 1
                assert target.passage.id in proposed[item.passage.id]
    assert confirmed == 4


@pytest.mark.embed
@needs_model
def test_reuse_flags_are_templated_reviewable_and_exactly_sourced(demo):
    result, _ = demo
    resolve = resolver_for([MOCK])
    assert len(result.flags) == 6  # 4 verbatim, 1 paraphrase, 1 unaddressed argument
    for flag in result.flags:
        assert flag.module == "reuse" and flag.review_status == "needs_legal_review" and flag.citation
        assert not contains_blocked_term(flag.message, CONFIG.messages.block_list)
        assert all(span_is_valid(span, resolve) for span in flag.spans)
    pair_flags = [flag for flag in result.flags if flag.standard_id == "reasoning_reuse"]
    assert all([e.role for e in flag.evidence] == ["judgment", "indictment"] for flag in pair_flags)
    assert {pair.flag_id for pair in result.pairs} == {flag.id for flag in pair_flags}


# --- verbatim matching and the score on small cases ------------------------------------------


def test_hand_computed_score():
    result, _ = run(small_case(f"{COPIED} {INDEPENDENT}\n"))
    assert (len(COPIED), len(INDEPENDENT)) == (59, 51)
    # The copied sentence's 5-grams cover "The accused ... Zurich" (58 chars, not the final period).
    assert (result.verbatim_chars, result.paraphrase_chars, result.reasoning_chars) == (58, 0, 110)
    assert result.score == pytest.approx(58 / 110)
    (pair,) = result.pairs
    assert (pair.kind, pair.jaccard, pair.containment) == ("verbatim", 1.0, 1.0)
    assert [(r.end - r.start) for r in pair.judgment_ranges] == [58]


def test_only_the_copied_words_of_a_partial_copy_count():
    result, _ = run(small_case("The Court finds that the accused moved the funds to a private account in Zurich.\n"))
    (pair,) = result.pairs
    assert pair.containment == pytest.approx(7 / 11, abs=1e-4)  # 7 of the sentence's 11 five-word sequences
    (covered,) = pair.judgment_ranges
    offset = pair.judgment.start
    assert pair.judgment.text[covered.start - offset : covered.end - offset] == "the accused moved the funds to a private account in Zurich"
    assert "64% of its 5-word sequences" in result.flags[0].message


def test_short_sentence_lifted_from_a_long_one_is_found_through_containment():
    long_sentence = (
        "On 4 May 2024 and again on 9 May 2024, acting on instructions from persons abroad, "
        "the accused transferred the money to an account abroad and then destroyed the records "
        "of every transfer he had made that month."
    )
    record = small_case("The accused transferred the money to an account abroad.\n", long_sentence + "\n")
    result, _ = run(record)
    (pair,) = result.pairs
    assert pair.containment == 1.0 and pair.jaccard < SETTINGS.lsh_threshold  # LSH on Jaccard alone would miss it


def test_legitimate_quotation_is_not_paired_even_when_identical():
    record = small_case(
        "V. ASSESSMENT\n\nThe prosecution argues that the accused moved the funds to a private account in Zurich.\n",
        "The prosecution argues that the accused moved the funds to a private account in Zurich.\n",
    )
    result, _ = run(record)
    assert result.pairs == () and result.flags == ()
    assert [e.reason for e in result.excluded if "prosecution" in e.span.text] == ["party_position"]
    assert result.score is None  # no reasoning left to measure


def test_no_indictment_means_no_score_but_arguments_are_still_checked():
    record = make_record(
        [
            ("judgment.txt", "judgment", JUDGMENT + f"{INDEPENDENT}\n"),
            ("note.txt", "monitoring_note", "Hearing date: 2 May 2024\n\nDefence counsel argued that the search was unlawful.\n"),
        ]
    )
    record = with_argument(record, "note.txt", "search was unlawful")
    result, llm = run(record)
    assert (result.indictment_doc_id, result.score, result.pairs) == (None, None, ())
    assert len(llm.calls) == 1 and [check.addressed for check in result.arguments] == [False]


def test_no_judgment_means_nothing_to_check():
    record = make_record([("indictment.txt", "indictment", INDICTMENT + COPIED + "\n")])
    result, llm = run(record)
    assert (result.judgment_doc_id, result.score, result.flags, llm.calls) == (None, None, (), ())


# --- unaddressed defence arguments ----------------------------------------------------------


def test_unanswered_argument_is_flagged_and_answered_one_is_not():
    result, llm = run()
    checks = {check.argument.text.split(" that ")[1][:20]: check for check in result.arguments}
    warrant, retroactivity = checks["the messages present"], checks["the first article wa"]
    assert not warrant.addressed and warrant.flag_id
    assert retroactivity.addressed and retroactivity.flag_id is None
    assert [INDEX[p.id] for p in MOCK.passages if p.span in retroactivity.responding] == [27, 28]
    assert (warrant.passages_checked, len(llm.calls)) == (10, 2)  # the prosecution's reply is not checked
    (flag,) = [flag for flag in result.flags if flag.status == "unaddressed_argument"]
    assert flag.id == warrant.flag_id and [e.role for e in flag.evidence] == ["argument"]
    assert "10 passages checked" in flag.message


def test_only_the_courts_reasoning_is_shown_to_the_model():
    _, llm = run()
    prompt = llm.calls[0].user
    assert "The Court finds that the conduct of the accused" in prompt
    assert "The defence contends that the messages relied on" not in prompt  # the restated argument
    assert "is charged with" not in prompt and "(signed)" not in prompt


def test_passage_ids_are_read_leniently_and_unknown_ids_ignored():
    def loose(system, user, schema, purpose):
        return {"note": "", "responding": ["7", "P8: That argument is therefore rejected.", "P99", "none"]}

    result, _ = run(responder=loose)
    assert all(check.addressed for check in result.arguments)
    assert [INDEX[p.id] for p in MOCK.passages if p.span in result.arguments[0].responding] == [27, 28]


def test_model_note_is_filtered_and_never_the_message():
    def judgemental(system, user, schema, purpose):
        return {"note": "The judge is biased and ignored it.", "responding": []}

    result, _ = run(responder=judgemental)
    flag = next(flag for flag in result.flags if flag.status == "unaddressed_argument")
    assert REMOVED in flag.model_note and "biased" not in flag.model_note
    assert "biased" not in flag.message


def test_long_reasoning_is_narrowed_to_the_most_similar_passages():
    narrow = CONFIG.model_copy(
        update={"settings": CONFIG.settings.model_copy(update={"reuse": SETTINGS.model_copy(update={"argument_max_passages": 3})})}
    )
    result, llm = run(config=narrow)
    assert {check.passages_checked for check in result.arguments} == {3}
    assert all(len(re.findall(r"^P\d+: ", call.user, re.MULTILINE)) == 3 for call in llm.calls)
    retroactivity = next(call.user for call in llm.calls if "amendment" in call.user.split("\n")[0])
    assert "the Court notes that the amendment entered into force" in retroactivity
