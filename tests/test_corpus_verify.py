"""Corpus builder, verification: a model quote is kept only when it is in the precedent's text,
exactly or after whitespace and quote normalisation. Never fuzzy (hard rule 2)."""

import json

import pytest

from corpus_builder.extract import PROMPT_SHA, cache_key, extract_all, schema_sha
from corpus_builder.store import BuildStore
from corpus_builder.verify import (
    FINDING_CHARS,
    VerifyReport,
    drops_negation,
    finding_bounds,
    negation_counts,
    negation_words,
    verify_all,
)
from fixtures.corpus_b.synthetic import REPORT, VIEWS, FakeLLM, facet, fixture_text, make_doc
from ratio.config import default_config
from ratio.extraction.align import locate_quote
from ratio.provenance import span_is_valid

TAXONOMY = default_config().fact_patterns

# A quote is kept only with every negation of its sentence ("without", "not"), so these quote whole clauses.
EXACT = "She was held in police custody for five days before she was first brought before a judge, who ordered her pretrial detention without hearing her lawyer"
CLAUSE = "She was held in police custody for five days before she was first brought before a judge"  # leaves out "without"
WRAPPED = "Her lawyer was not allowed to attend the remand hearing, and she met him for the first time two weeks after her arrest"  # a line break
WRAPPED_TEXT = "Her lawyer was not allowed to attend the remand hearing, and she met him for the first time two\nweeks after her arrest"
CURLY = 'the court refused to hear the two defence witnesses named by the author, stating that their evidence was "not relevant to the charge"'
PARAPHRASE = "The author was held for five days without ever seeing a judge"
NEAR_MISS = "She was held in police custody for five days before she was brought before a judge"
FINDING = "The Committee finds that the refusal to hear the defence witnesses violated article 14(3)(e) of the Covenant"
TWICE = "The Committee considers that the facts before it disclose a violation of article 9(3) of the Covenant"
MEDIA = "The author is a journalist and the editor of an independent news portal in the capital"


@pytest.fixture
def store(tmp_path):
    return BuildStore(tmp_path / "build.db")


@pytest.fixture
def doc(store):
    views = make_doc("ccpr-9999-2099", fixture_text(VIEWS))
    store.put_document(views, views.url)
    return views


def put_answer(store, doc, facets, *, model="fake-model", created_at="2099-01-01T00:00:00+00:00", text_sha256=None):
    key = cache_key(
        text_sha256=text_sha256 or doc.text_sha256, prompt_sha=PROMPT_SHA, schema_sha=schema_sha(TAXONOMY), model=model
    )
    store.put_extraction(key, doc.id, model, json.dumps({"facets": facets}), created_at)
    return key


def verified(store, doc, *facets):
    put_answer(store, doc, list(facets))
    report = verify_all(store, TAXONOMY)
    kept, dropped = store.verified().get(doc.id, ((), 0))
    return report, {f.facet_id: f for f in kept}, dropped


def resolver(doc):
    return lambda doc_id: doc.text if doc_id == doc.doc_id else None


def test_exact_and_normalised_quotes_are_kept_as_spans_of_the_text(store, doc):
    _, kept, dropped = verified(
        store,
        doc,
        facet("gc35_48h", [EXACT]),
        facet("iccpr_14_3_b", [WRAPPED]),
        facet("iccpr_14_3_e", [CURLY], finding=FINDING),
    )
    assert dropped == 0
    assert [q.match for q in kept["gc35_48h"].facts] == ["exact"]
    assert [q.match for q in kept["iccpr_14_3_b"].facts] == ["normalized"]
    assert [q.match for q in kept["iccpr_14_3_e"].facts] == ["normalized"]
    assert kept["iccpr_14_3_e"].finding.match == "exact"
    for item in kept.values():
        for quote in (*item.facts, item.finding):
            if quote is not None:
                assert quote.span.doc_id == "precedent-ccpr-9999-2099/text"
                assert quote.span.case_id == "precedent-ccpr-9999-2099"
                assert span_is_valid(quote.span, resolver(doc))
    assert kept["gc35_48h"].facts[0].span.text == EXACT
    assert kept["iccpr_14_3_b"].facts[0].span.text == WRAPPED_TEXT


def test_non_verbatim_quotes_are_dropped_and_counted(store, doc):
    _, kept, dropped = verified(store, doc, facet("iccpr_14_3_b", [WRAPPED, PARAPHRASE]))
    assert [q.span.text for q in kept["iccpr_14_3_b"].facts] == [WRAPPED_TEXT]
    assert dropped == 1


def test_a_quote_fuzzy_matching_would_accept_is_dropped(store, doc):
    assert locate_quote(doc.text, NEAR_MISS).match == "fuzzy"  # the default aligner would accept it
    _, kept, dropped = verified(store, doc, facet("gc35_48h", [NEAR_MISS]))
    assert kept == {} and dropped == 1


def test_a_repeated_passage_is_ambiguous_and_dropped(store, doc):
    _, kept, dropped = verified(store, doc, facet("gc35_48h", [TWICE]))
    assert kept == {} and dropped == 1


def test_unknown_facet_ids_are_dropped_with_their_quotes(store, doc):
    _, kept, dropped = verified(store, doc, facet("judge_bias", [EXACT], finding=FINDING), facet("gc35_48h", [EXACT]))
    assert list(kept) == ["gc35_48h"]
    assert dropped == 2


def test_a_finding_not_found_leaves_the_facet_without_one(store, doc):
    _, kept, dropped = verified(store, doc, facet("gc35_48h", [EXACT], finding=TWICE))
    assert kept["gc35_48h"].finding is None
    assert kept["gc35_48h"].finding_kind == "violation_found"
    assert dropped == 1


def test_a_facet_left_without_facts_is_dropped_with_its_finding(store, doc):
    _, kept, dropped = verified(store, doc, facet("iccpr_14_3_e", [PARAPHRASE], finding=FINDING))
    assert kept == {} and dropped == 2


def test_facts_are_deduplicated_by_span(store, doc):
    _, kept, dropped = verified(store, doc, facet("gc35_48h", [EXACT, EXACT + ".", f"“{EXACT}”"]), facet("gc35_48h", [EXACT]))
    assert len(kept["gc35_48h"].facts) == 1
    assert dropped == 0


def test_facets_follow_the_taxonomy_order_and_facts_the_text_order(store, doc):
    put_answer(store, doc, [facet("media_defendant", [MEDIA]), facet("gc35_48h", [EXACT]), facet("iccpr_14_3_b", [WRAPPED, EXACT])])
    verify_all(store, TAXONOMY)
    kept, _ = store.verified()[doc.id]
    assert [f.facet_id for f in kept] == ["iccpr_14_3_b", "gc35_48h", "media_defendant"]
    starts = [q.span.start for q in kept[0].facts]
    assert starts == sorted(starts)


def test_an_extraction_of_older_text_is_not_verified(store, doc):
    put_answer(store, doc, [facet("gc35_48h", [EXACT])], text_sha256="0" * 64)
    report = verify_all(store, TAXONOMY)
    assert report.missing == (doc.id,)
    assert store.verified() == {}


def test_the_latest_current_extraction_is_used(store, doc):
    put_answer(store, doc, [facet("gc35_48h", [EXACT])], model="old-model", created_at="2099-01-01T00:00:00+00:00")
    put_answer(store, doc, [facet("media_defendant", [MEDIA])], model="new-model", created_at="2099-02-01T00:00:00+00:00")
    verify_all(store, TAXONOMY)
    kept, _ = store.verified()[doc.id]
    assert [f.facet_id for f in kept] == ["media_defendant"]


def test_the_report_counts_what_was_kept_and_dropped(store, doc):
    report, _, _ = verified(store, doc, facet("gc35_48h", [EXACT, PARAPHRASE], finding=FINDING), facet("media_defendant", [MEDIA]))
    assert report == VerifyReport(documents=1, facets=2, quotes=3, dropped=1, missing=())


def test_verify_reads_what_extract_stored(store, doc):
    extract_all(store, TAXONOMY, FakeLLM(answer={"facets": [facet("gc35_48h", [EXACT], finding=FINDING)]}))
    report = verify_all(store, TAXONOMY)
    kept, dropped = store.verified()[doc.id]
    assert report.documents == 1 and dropped == 0
    assert kept[0].facts[0].span.text == EXACT


# --- negations, short quotes and whole-sentence findings ----------------------------------------

NOT_FOUND = "7.5 Accordingly, the Committee does not find a violation of article 9(1) of the Covenant in respect of the arrest itself."
WITHOUT_NOT = "a violation of article 9(1) of the Covenant in respect of the arrest itself"  # the negation left out


@pytest.fixture
def negated(store):
    views = make_doc("ccpr-9998-2099", fixture_text(VIEWS) + "\n" + NOT_FOUND + "\n")
    store.put_document(views, views.url)
    return views


def test_a_finding_that_drops_the_negation_of_its_sentence_is_dropped(store, negated):
    assert locate_quote(negated.text, WITHOUT_NOT, fuzzy_min_chars=10**9).match == "exact"  # it is in the text
    _, kept, dropped = verified(store, negated, facet("gc35_48h", [EXACT], "violation_found", finding=WITHOUT_NOT))
    assert kept["gc35_48h"].finding is None  # never shown under "Committee found a violation"
    assert dropped == 1


def test_a_fact_that_drops_the_negation_of_its_sentence_is_dropped(store, negated):
    _, kept, dropped = verified(store, negated, facet("gc35_48h", [WITHOUT_NOT, MEDIA]))
    assert [q.span.text for q in kept["gc35_48h"].facts] == [MEDIA]
    assert dropped == 1


def test_a_clause_that_leaves_out_its_sentences_negation_is_dropped(store, doc):
    _, kept, dropped = verified(store, doc, facet("gc35_48h", [CLAUSE]))
    assert kept == {} and dropped == 1


# The sentence holds "not" twice; the quote keeps the second and drops the first, which reverses the finding.
TWO_NOTS = "The Committee does not find a violation of article 14(3)(b), since the author was not deprived of counsel at any stage."
ONE_NOT = "a violation of article 14(3)(b), since the author was not deprived of counsel at any stage"
BOTH_NOTS = "The Committee does not find a violation of article 14(3)(b), since the author was not deprived of counsel at any stage"


@pytest.fixture
def two_nots(store):
    views = make_doc("ccpr-9996-2099", fixture_text(VIEWS) + "\n" + TWO_NOTS + "\n")
    store.put_document(views, views.url)
    return views


def test_a_quote_with_fewer_negations_than_its_sentence_is_dropped(store, two_nots):
    assert locate_quote(two_nots.text, ONE_NOT, fuzzy_min_chars=10**9).match == "exact"  # it is in the text
    _, kept, dropped = verified(store, two_nots, facet("iccpr_14_3_b", [ONE_NOT, WRAPPED], "violation_found", finding=ONE_NOT))
    assert [q.span.text for q in kept["iccpr_14_3_b"].facts] == [WRAPPED_TEXT]
    assert kept["iccpr_14_3_b"].finding is None  # never shown under "Committee found a violation"
    assert dropped == 2


def test_a_quote_with_every_negation_of_its_sentence_is_kept(store, two_nots):
    _, kept, dropped = verified(store, two_nots, facet("iccpr_14_3_b", [BOTH_NOTS], "no_violation", finding=BOTH_NOTS))
    assert [q.span.text for q in kept["iccpr_14_3_b"].facts] == [BOTH_NOTS]
    assert kept["iccpr_14_3_b"].finding.span.text == TWO_NOTS
    assert dropped == 0


@pytest.mark.parametrize(
    ("sentence", "quote"),
    [
        (TWO_NOTS, ONE_NOT),
        ("He was neither charged nor released for nine days.", "charged nor released for nine days"),
        ("He cannot appeal and was not told why.", "and was not told why"),
        ("She was never told the charge and had no lawyer.", "She was never told the charge and had"),
        ("The defendant didn’t have an interpreter and wasn't told why.", "have an interpreter and wasn't told why"),
        ("He was held without charge and without a lawyer.", "He was held without charge and"),
    ],
)
def test_a_quote_that_loses_a_negation_of_its_sentence_drops_one(sentence: str, quote: str):
    assert drops_negation(sentence, quote)


@pytest.mark.parametrize(
    ("sentence", "quote"),
    [
        (TWO_NOTS, BOTH_NOTS),
        ("In its opinion No. 27/2017 the Working Group found the detention arbitrary.", "the Working Group found the detention arbitrary"),
        ("Under Law No 12 he was detained.", "he was detained"),
        ("He was not brought before a judge.", "He was not brought before a judge"),
        ("The court heard the two defence witnesses.", "heard the two defence witnesses"),
    ],
)
def test_a_quote_that_keeps_every_negation_of_its_sentence_drops_none(sentence: str, quote: str):
    assert not drops_negation(sentence, quote)


HELD_48H = "the author was held for more than 48 hours before being brought before a judge"
DENIES = f"4.2 The State party denies that {HELD_48H}."
REJECTS = f"9.1 The Committee rejects the claim that {HELD_48H}."
CONTESTS = f"4.3 The State party contests that {HELD_48H}."
DISPUTES = f"4.4 The State party disputes that {HELD_48H}."
UNSUBSTANTIATED = f"9.2 The Committee considers unsubstantiated the claim that {HELD_48H}."
DENYING_SENTENCES = (DENIES, REJECTS, CONTESTS, DISPUTES, UNSUBSTANTIATED)


@pytest.mark.parametrize(
    ("sentence", "quote"),
    [
        (DENIES, HELD_48H),
        (REJECTS, HELD_48H),
        ("The officers deny that the author was beaten in custody.", "that the author was beaten in custody"),
        ("The officers denied that the author was beaten in custody.", "that the author was beaten in custody"),
        ("Denying any arrest, the police kept him in a cell for days.", "the police kept him in a cell for days"),
        ("Refusing him a lawyer, the police questioned him for hours.", "the police questioned him for hours"),
        ("Failing to review the detention, the court kept him in custody for months.", "the court kept him in custody for months"),
        ("The court refuses to hear the defence witnesses at all.", "to hear the defence witnesses at all"),
        ("The court refuse to hear the defence witnesses at all.", "to hear the defence witnesses at all"),
        ("The State party rejected the claim that he had been beaten.", "he had been beaten"),
        ("The Committee will reject the claim that he had been beaten.", "he had been beaten"),
        ("The court dismissed the claim that his confession was coerced.", "his confession was coerced"),
        ("The courts dismiss the claim that his confession was coerced.", "his confession was coerced"),
        ("The police failed to inform his family of the arrest.", "to inform his family of the arrest"),
        ("The police fail to inform families of arrests.", "to inform families of arrests"),
        ("The State party fails to show that the detention was necessary.", "that the detention was necessary"),
        ("The tribunal lacked the independence required by article 14.", "the independence required by article 14"),
        ("The tribunal lacks the independence required by article 14.", "the independence required by article 14"),
        ("Courts that lack the independence required by article 14.", "the independence required by article 14"),
        (CONTESTS, HELD_48H),
        (DISPUTES, HELD_48H),
        (UNSUBSTANTIATED, HELD_48H),
        ("The authorities contest the claim that he had been beaten.", "he had been beaten"),
        ("The State party contested the claim that he had been beaten.", "he had been beaten"),
        ("Contesting the claim, the State party says he was held for a day.", "the State party says he was held for a day"),
        ("The authorities dispute the claim that he had been beaten.", "he had been beaten"),
        ("The State party disputed the claim that he had been beaten.", "he had been beaten"),
        ("The officers refute the claim that the author was beaten in custody.", "the author was beaten in custody"),
        ("The State party refutes the claim that the author was beaten in custody.", "the author was beaten in custody"),
        ("The State party refuted the claim that the author was beaten in custody.", "the author was beaten in custody"),
        ("The State party calls untrue the claim that the author was beaten in custody.", "the author was beaten in custody"),
        ("The State party calls unfounded the claim that the author was beaten in custody.", "the author was beaten in custody"),
        ("The judges decline to hear the defence witnesses at all.", "to hear the defence witnesses at all"),
        ("The court declines to hear the defence witnesses at all.", "to hear the defence witnesses at all"),
        ("The court declined to hear the defence witnesses at all.", "to hear the defence witnesses at all"),
    ],
)
def test_a_quote_that_leaves_out_a_denying_or_rejecting_verb_drops_a_negation(sentence: str, quote: str):
    assert drops_negation(sentence, quote)


@pytest.mark.parametrize("sentence", DENYING_SENTENCES)
def test_a_fact_or_finding_that_drops_the_denying_verb_is_never_kept(store, sentence):
    views = make_doc("ccpr-9995-2099", fixture_text(VIEWS) + "\n" + sentence + "\n")
    store.put_document(views, views.url)
    _, kept, dropped = verified(store, views, facet("gc35_48h", [HELD_48H, MEDIA], "violation_found", finding=HELD_48H))
    assert [q.span.text for q in kept["gc35_48h"].facts] == [MEDIA]  # never "the author was held ..." alone
    assert kept["gc35_48h"].finding is None
    assert dropped == 2


@pytest.mark.parametrize("sentence", DENYING_SENTENCES)
def test_a_quote_that_keeps_the_denying_verb_is_kept(store, sentence):
    views = make_doc("ccpr-9995-2099", fixture_text(VIEWS) + "\n" + sentence + "\n")
    store.put_document(views, views.url)
    whole = sentence.split(" ", 1)[1].removesuffix(".")  # without its paragraph number
    _, kept, dropped = verified(store, views, facet("gc35_48h", [whole]))
    assert [q.span.text for q in kept["gc35_48h"].facts] == [whole] and dropped == 0


def test_a_finding_is_kept_as_its_whole_sentence_with_its_negation(store, negated):
    quote = "the Committee does not find a violation of article 9(1) of the Covenant"
    _, kept, dropped = verified(store, negated, facet("gc35_48h", [EXACT], "no_violation", finding=quote))
    finding = kept["gc35_48h"].finding
    assert dropped == 0 and finding.match == "exact"
    assert finding.span.text == NOT_FOUND
    assert span_is_valid(finding.span, resolver(negated))
    assert kept["gc35_48h"].facts[0].span.text == EXACT  # facts stay the quoted passage


def test_a_quote_under_eight_words_is_dropped(store, doc):
    short = "false information about the regional water authority"  # seven words, in the text
    assert locate_quote(doc.text, short).match == "exact"
    _, kept, dropped = verified(store, doc, facet("media_defendant", [short, MEDIA], finding=short))
    assert [q.span.text for q in kept["media_defendant"].facts] == [MEDIA]
    assert kept["media_defendant"].finding is None
    assert dropped == 2


def test_a_long_finding_sentence_is_cut_around_the_quote_without_losing_it():
    text = "SYNTHETIC: a long sentence.\n\n" + "The Committee, " + "having regard to the material, " * 40 + (
        "does not find a violation of article 9 of the Covenant" + ", having regard to the record" * 40 + "."
    )
    start = text.index("does not find")
    end = start + len("does not find a violation of article 9 of the Covenant")
    low, high = finding_bounds(text, start, end)
    assert low <= start and end <= high and high - low <= FINDING_CHARS
    assert text[low].isalnum() and not text[low - 1].isalnum()  # whole words at both ends
    assert text[high - 1].isalnum() and not text[high].isalnum()


def test_a_finding_longer_than_the_cap_is_kept_whole():
    text = "SYNTHETIC: " + "word " * 300 + "end."
    assert finding_bounds(text, 11, len(text) - 1) == (11, len(text) - 1)


# A 574-character sentence (under FINDING_CHARS) whose negation, "does not show", comes near its end.
LONG_574 = (
    "9.6 Having regard to the State party's account of the investigation, to the author's comments on that account, "
    "to the record of the remand hearing held in the capital, to the medical certificate issued on the day of his release, "
    "to the letters his lawyer sent to the prosecutors, "
    "to the statements of the two officers who questioned him and to the decision of the court of appeal that upheld his conviction, "
    "the Committee considers that the material before it does not show that the author was held in police custody for five days "
    "before he was first brought before a judge."
)
LONG_574_QUOTE = (
    "the material before it does not show that the author was held in police custody for five days before he was first brought before a judge"
)
FACT_5_DAYS = "2.2 The author was held in police custody for five days before he was first brought before a judge, the record shows."


def test_a_574_character_finding_sentence_is_kept_whole_with_its_negation_near_the_end(store):
    assert len(LONG_574) == 574 and len(LONG_574) - LONG_574.index("does not") < 120
    views = make_doc("ccpr-9994-2099", f"SYNTHETIC: a long finding.\n\n{FACT_5_DAYS}\n\n{LONG_574}\n")
    store.put_document(views, views.url)
    fact = "The author was held in police custody for five days before he was first brought before a judge, the record shows"
    _, kept, dropped = verified(store, views, facet("gc35_48h", [fact], "no_violation", finding=LONG_574_QUOTE))
    finding = kept["gc35_48h"].finding
    assert dropped == 0 and finding.span.text == LONG_574  # the whole sentence: no window under FINDING_CHARS
    assert "does not show" in finding.span.text and span_is_valid(finding.span, resolver(views))


def long_sentence(before_negation: int, between: int, after: int) -> tuple[str, int, int]:
    """A SYNTHETIC text holding one long sentence: filler, "does not", filler, then the quote and filler.
    Returns the text and the quote's offsets; the negation precedes the quote, outside it."""
    quote = "the author was held for six days before being brought before a judge"
    sentence = (
        "The Committee, " + "having regard to the material, " * before_negation + "does not, "
        + "on the record of the case, " * between + f"consider that {quote}" + ", having regard to the record" * after + "."
    )  # fmt: skip
    text = "SYNTHETIC: a long sentence.\n\n" + sentence
    start = text.index(quote)
    return text, start, start + len(quote)


def test_a_long_finding_window_keeps_a_negation_that_precedes_the_quote():
    text, start, end = long_sentence(before_negation=10, between=12, after=12)
    negation = text.index("not, on the record")
    assert start - negation > FINDING_CHARS // 2  # a window centred on the quote would cut it
    low, high = finding_bounds(text, start, end)
    assert low <= negation and end <= high and high - low <= FINDING_CHARS
    assert negation_counts(text[low:high]) == {"not": 1}
    assert text[low].isalnum() and not text[low - 1].isalnum()  # whole words at both ends
    assert text[high - 1].isalnum() and not text[high].isalnum()


def test_negation_words_gives_each_negation_and_where_it_starts():
    text = "In Opinion No. 27/2017 the court didn’t find that he was not held; Law No 12 was never applied."
    assert list(negation_words(text)) == [(text.index("didn"), "n't"), (text.index("not held"), "not"), (text.index("never"), "never")]


def test_a_long_finding_whose_negation_cannot_fit_in_the_window_is_dropped():
    text, start, end = long_sentence(before_negation=2, between=22, after=4)
    assert end - text.index("not, on the record") > FINDING_CHARS
    assert finding_bounds(text, start, end) is None


# --- finding kinds by document kind -------------------------------------------------------------

KEPT_4_DAYS = "The monitor observed that the defendant was kept in a police station for four days before any judge saw him"
ASSESSMENT = "In the monitor's assessment, the delay before the first appearance breached the requirement of prompt judicial control"


@pytest.fixture
def report(store):
    melk = make_doc("tw-synthetic-melk", fixture_text(REPORT), kind="trialwatch_report")
    store.put_document(melk, melk.url)
    return melk


@pytest.mark.parametrize("kind", ["monitor_assessment", "not_examined"])
def test_a_trialwatch_report_keeps_a_monitor_assessment_or_not_examined(store, report, kind):
    report_, kept, dropped = verified(store, report, facet("gc35_48h", [KEPT_4_DAYS], kind, finding=ASSESSMENT))
    assert kept["gc35_48h"].finding_kind == kind and kept["gc35_48h"].finding is not None
    assert dropped == 0 and report_.wrong_kind == 0


@pytest.mark.parametrize("kind", ["violation_found", "no_violation"])
def test_a_trialwatch_report_never_holds_a_violation_finding(store, report, kind):
    report_, kept, dropped = verified(store, report, facet("gc35_48h", [KEPT_4_DAYS], kind, finding=ASSESSMENT))
    assert [q.span.text for q in kept["gc35_48h"].facts] == [KEPT_4_DAYS]  # the facts stay
    assert kept["gc35_48h"].finding is None and kept["gc35_48h"].finding_kind == "not_examined"
    assert dropped == 1 and report_.wrong_kind == 1


def test_a_wrong_kind_without_a_quote_is_counted_but_drops_no_quote(store, report):
    report_, kept, dropped = verified(store, report, facet("gc35_48h", [KEPT_4_DAYS], "violation_found"))
    assert kept["gc35_48h"].finding_kind == "not_examined"
    assert dropped == 0 and report_.wrong_kind == 1


NOT_FOUND_QUOTE = "the Committee does not find a violation of article 9(1) of the Covenant"


@pytest.mark.parametrize(("kind", "finding"), [("violation_found", FINDING), ("no_violation", NOT_FOUND_QUOTE), ("not_examined", FINDING)])
@pytest.mark.parametrize("doc_kind", ["ccpr_views", "wgad_opinion"])
def test_views_and_opinions_keep_their_finding_kinds(store, kind, finding, doc_kind):
    views = make_doc("ccpr-9997-2099", fixture_text(VIEWS) + "\n" + NOT_FOUND + "\n", kind=doc_kind)
    store.put_document(views, views.url)
    report_, kept, dropped = verified(store, views, facet("iccpr_14_3_e", [CURLY], kind, finding=finding))
    assert kept["iccpr_14_3_e"].finding_kind == kind and kept["iccpr_14_3_e"].finding is not None
    assert dropped == 0 and report_.wrong_kind == 0


@pytest.mark.parametrize("doc_kind", ["ccpr_views", "wgad_opinion"])
def test_views_and_opinions_never_hold_a_monitor_assessment(store, doc_kind):
    views = make_doc("ccpr-9997-2099", fixture_text(VIEWS), kind=doc_kind)
    store.put_document(views, views.url)
    report_, kept, dropped = verified(store, views, facet("iccpr_14_3_e", [CURLY], "monitor_assessment", finding=FINDING))
    assert kept["iccpr_14_3_e"].finding is None and kept["iccpr_14_3_e"].finding_kind == "not_examined"
    assert len(kept["iccpr_14_3_e"].facts) == 1  # the facts stay
    assert dropped == 1 and report_.wrong_kind == 1


# --- a finding's kind against its own words -----------------------------------------------------

DOES_NOT_FIND_9_3 = "The Committee does not find a violation of article 9 (3) of the Covenant."
FINDS_14_3_B = "The Committee finds a violation of article 14 (3) (b)."
DISCLOSES_9_3 = "The Committee finds that the facts before it disclose a violation of article 9 (3)."


def views_with(store, sentence, *, kind="ccpr_views"):
    """The SYNTHETIC Views with one more sentence, stored."""
    views = make_doc("ccpr-9993-2099", fixture_text(VIEWS) + "\n" + sentence + "\n", kind=kind)
    store.put_document(views, views.url)
    return views


@pytest.mark.parametrize("doc_kind", ["ccpr_views", "wgad_opinion"])
@pytest.mark.parametrize(("kind", "sentence"), [("violation_found", DOES_NOT_FIND_9_3), ("no_violation", FINDS_14_3_B)])
def test_a_finding_whose_words_contradict_its_kind_is_dropped(store, doc_kind, kind, sentence):
    views = views_with(store, sentence, kind=doc_kind)
    report_, kept, dropped = verified(store, views, facet("gc35_48h", [EXACT], kind, finding=sentence.removesuffix(".")))
    assert kept["gc35_48h"].finding is None  # never "found a violation" over "does not find", nor the reverse
    assert kept["gc35_48h"].finding_kind == "not_examined"  # a kind its own words contradict is not stored
    assert [q.span.text for q in kept["gc35_48h"].facts] == [EXACT]  # the facts stay
    assert dropped == 1 and report_.wrong_kind == 0


@pytest.mark.parametrize(
    ("kind", "sentence"),
    [
        ("violation_found", DISCLOSES_9_3),
        ("no_violation", DOES_NOT_FIND_9_3),
        ("not_examined", DOES_NOT_FIND_9_3),  # only the two kinds that state an outcome are checked
        ("not_examined", DISCLOSES_9_3),
    ],
)
def test_a_finding_whose_words_fit_its_kind_is_kept(store, kind, sentence):
    views = views_with(store, sentence)
    _, kept, dropped = verified(store, views, facet("gc35_48h", [EXACT], kind, finding=sentence.removesuffix(".")))
    assert kept["gc35_48h"].finding.span.text == sentence and kept["gc35_48h"].finding_kind == kind
    assert dropped == 0


@pytest.mark.parametrize(
    ("kind", "sentence", "contradicts"),
    [
        ("violation_found", "The Committee finds a violation of article 14 (3) (b), since the author was not afforded counsel.", False),
        ("violation_found", "The deprivation of liberty of Mr. Venn is arbitrary and falls within categories I and III.", False),
        ("violation_found", "The deprivation of liberty of Mr. Venn is not arbitrary.", True),
        ("violation_found", "The Committee is not in a position to find a violation of article 14 (3) (g).", True),
        ("violation_found", "The facts before the Committee do not disclose a violation of article 9 (3).", True),
        ("no_violation", "There has been no violation of article 19 of the Covenant.", False),
        ("no_violation", "The author was not afforded counsel, in violation of article 14 (3) (b).", True),
    ],
)
def test_only_a_negated_outcome_contradicts_a_finding_kind(kind, sentence, contradicts):
    from corpus_builder.verify import contradicts_kind

    assert contradicts_kind(kind, sentence) is contradicts
