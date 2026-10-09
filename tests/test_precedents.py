"""Similar cases (ratio/precedents.py): a case's fact patterns come only from its stored findings and
exact keyword matches, and a precedent is linked only when it shares enough of them, one about
procedure, ranked by how rare the shared ones are, with every precedent quote re-checked against its
stored text. No model is called: the embedder only pairs quotes and breaks ties."""

import math
from collections import Counter

import pytest

from ratio.config import PrecedentSettings, load_config
from ratio.paths import MINILM_DIR
from ratio.precedent_schema import (
    EMBED_MODEL_TAG,
    CaseProfile,
    CorpusMeta,
    PrecedentDoc,
    PrecedentFacet,
    PrecedentQuote,
    precedent_doc_id,
)
from ratio.precedent_store import SqlitePrecedentIndex
from ratio.precedents import COSINE_DECIMALS, _defendant_surname, _sentence_bounds, link, profile
from ratio.provenance import resolver_for, span_is_valid
from ratio.results import AbsenceResult, CaseAnalysis, ClockResult, Interval, RenewalResult, ReuseResult
from ratio.schema import Evidence, Flag, FollowUp, SourceSpan, stable_id
from ratio.testing import make_record
from tests.precedent_fixtures import DEMO_LIKE, LINKED, NOT_LINKED, TAMPERED, FakeEmbedder, build_fixture_corpus

SETTINGS = PrecedentSettings(backend="sqlite")


def needs_minilm(test):
    """The demo replay embeds passages with the local MiniLM model."""
    skip = pytest.mark.skipif(not (MINILM_DIR / "modules.json").exists(), reason="run scripts/fetch_models.py once")
    return pytest.mark.embed(skip(test))


@pytest.fixture(scope="module")
def taxonomy():
    return load_config().fact_patterns


@pytest.fixture(scope="module")
def corpus_index(tmp_path_factory, taxonomy):
    path = build_fixture_corpus(tmp_path_factory.mktemp("corpus") / "precedents.db", FakeEmbedder(), taxonomy)
    return SqlitePrecedentIndex(path, taxonomy)


# --- helpers ----------------------------------------------------------------------------------

NOTE = (
    "Three journalists attended the hearing.\n\n"
    "Defence counsel was not present when the court ordered the accused's detention. "
    "The court refused to hear the two witnesses named by the defence."
)
INDICTMENT = (
    "I. THE ACCUSED\n\n"
    "Dana Orel, born in 1990, an editor of the news portal Testland Daily, residing in Testville.\n\n"
    "II. CHARGE\n\n"
    "The accused is charged with spreading false news likely to cause alarm. "
    "The editorial board met twice."
)


def titled(record, title: str):
    """The record under another case title (make_record names every defendant "A. Example")."""
    return record.model_copy(update={"meta": record.meta.model_copy(update={"title": title})})


@pytest.fixture(scope="module")
def record():
    built = make_record([("notes/hearing_1.txt", "monitoring_note", NOTE), ("indictment.txt", "indictment", INDICTMENT)])
    return titled(built, "Republic of Testland v. Dana Orel")


def span_of(record, path: str, phrase: str) -> SourceSpan:
    doc = record.document(f"{record.case_id}/{path}")
    start = doc.text.index(phrase)
    return SourceSpan(doc_id=doc.id, start=start, end=start + len(phrase), text=phrase)


def make_flag(record, module: str, standard: str, status: str, phrase: str) -> Flag:
    span = span_of(record, "notes/hearing_1.txt", phrase)
    return Flag(
        id=stable_id(module, standard, status, phrase),
        case_id=record.case_id,
        module=module,
        standard_id=standard,
        standard_label=standard,
        status=status,
        message="test finding",
        evidence=(Evidence(role="supporting", span=span),),
        review_status="confirmed" if status == "exceeds_benchmark" else "needs_legal_review",
    )


def analysis_with(record, *, absence=(), clock=(), reuse=(), renewal=(), follow_ups=(), intervals=()) -> CaseAnalysis:
    return CaseAnalysis(
        case_id=record.case_id,
        absence=AbsenceResult(assessments=(), flags=tuple(absence), follow_ups=tuple(follow_ups)),
        clock=ClockResult(timeline=(), intervals=tuple(intervals), flags=tuple(clock)),
        reuse=ReuseResult(judgment_doc_id=None, indictment_doc_id=None, flags=tuple(reuse)),
        renewal=RenewalResult(flags=tuple(renewal)),
    )


def idf(df: int, documents: int) -> float:
    return math.log((documents + 1) / (df + 1)) + 1


class MemoryIndex:
    """An in-memory PrecedentIndex: facts are paired by ``nearest`` only (no vectors)."""

    def __init__(self, entries, *, texts=None, nearest=None, doc_freq=None, documents=None):
        self.entries = tuple(entries)
        self.texts = texts if texts is not None else {doc.doc_id: doc.text for doc, _ in self.entries}
        self.nearest = nearest or {}
        self.doc_freq = doc_freq
        self.documents = documents if documents is not None else len(self.entries)

    def meta(self) -> CorpusMeta:
        facets = sum(len(facets) for _, facets in self.entries)
        return CorpusMeta(schema_version=1, taxonomy_sha="t", embed_model=EMBED_MODEL_TAG, extraction_model="none",
                          prompt_sha="0", built_at="x", documents=self.documents, facets=facets, dropped_quotes=0)  # fmt: skip

    def facet_doc_freq(self):
        if self.doc_freq is not None:
            return self.doc_freq
        return Counter(facet.facet_id for _, facets in self.entries for facet in facets)

    def candidates(self, facet_ids):
        wanted = set(facet_ids)
        return [(doc, facets) for doc, facets in self.entries if any(f.facet_id in wanted for f in facets)]

    def nearest_facts(self, facet_id, query, precedent_ids):
        return {pid: found for (fid, pid), found in self.nearest.items() if fid == facet_id and pid in precedent_ids}

    def text(self, doc_id):
        return self.texts.get(doc_id)


def make_precedent(pid: str, facts: dict[str, str], findings: dict[str, str] | None = None):
    """A precedent whose text is its fact (and finding) sentences, one facet per entry of ``facts``."""
    findings = findings or {}
    text = "SYNTHETIC: test precedent.\n\n" + "\n\n".join([*facts.values(), *findings.values()])
    doc = PrecedentDoc(id=pid, kind="ccpr_views", title=f"Synthetic {pid}", body="Test Committee", url="https://example.invalid/x",
                       retrieved_at="2026-10-09", raw_sha256="0" * 64, text_sha256="1" * 64, attribution="SYNTHETIC", text=text)  # fmt: skip

    def quote(sentence: str) -> PrecedentQuote:
        start = text.index(sentence)
        return PrecedentQuote(span=SourceSpan(doc_id=doc.doc_id, start=start, end=start + len(sentence), text=sentence), match="exact")

    facets = tuple(
        PrecedentFacet(
            precedent_id=pid,
            facet_id=fid,
            facts=(quote(sentence),),
            finding_kind="violation_found",
            finding=quote(findings[fid]) if fid in findings else None,
        )
        for fid, sentence in facts.items()
    )
    return doc, facets


# --- profile: where a case's fact patterns come from -----------------------------------------


def test_facets_come_only_from_findings_that_match_a_signal(record, taxonomy):
    counsel = make_flag(record, "absence", "iccpr_14_3_b", "evidence_of_violation", "Defence counsel was not present")
    complied = make_flag(record, "absence", "iccpr_14_3_e", "evidence_of_compliance", "The court refused to hear")
    late = make_flag(record, "clock", "gc35_48h", "exceeds_benchmark", "ordered the accused's detention")
    reuse = make_flag(record, "reuse", "reasoning_reuse", "paraphrase_reuse", "Three journalists attended")
    gap = make_flag(record, "renewal", "renewal_gap", "order_gap", "the two witnesses")
    found = profile(record, analysis_with(record, absence=(counsel, complied), clock=(late,), reuse=(reuse,), renewal=(gap,)), taxonomy)
    findings = [facet for facet in found.facets if facet.origin == "finding"]
    assert [(f.facet_id, f.flag_id) for f in findings] == [("iccpr_14_3_b", counsel.id), ("gc35_48h", late.id), ("reasoning_reuse", reuse.id)]
    assert all(f.evidence == flag.evidence for f, flag in zip(findings, (counsel, late, reuse), strict=True))


def test_no_facet_from_a_follow_up_an_amber_interval_or_a_benchmark_needing_legal_review(taxonomy):
    plain = make_record([("notes/hearing_1.txt", "monitoring_note", "The hearing opened late.\n\nNo interpreter was present.")])
    follow_up = FollowUp(id="u1", case_id=plain.case_id, rubric_id="iccpr_14_3_f", question="Did the accused understand the language?")
    interval = Interval(id="i1", benchmark_id="counsel_access", benchmark_name="Access to counsel", from_event_id=None, to_event_id=None,
                        min_hours=100, max_hours=120, threshold_hours=None, status="measured", citation="x", review_status="needs_legal_review")  # fmt: skip
    amber = make_flag(plain, "clock", "gc35_48h", "needs_review", "The hearing opened late.")
    found = profile(plain, analysis_with(plain, clock=(amber,), follow_ups=(follow_up,), intervals=(interval,)), taxonomy)
    assert found == CaseProfile(case_id=plain.case_id, facets=())


def test_a_finding_the_lawyer_rejected_makes_no_facet(record, taxonomy):
    counsel = make_flag(record, "absence", "iccpr_14_3_b", "evidence_of_violation", "Defence counsel was not present")
    late = make_flag(record, "clock", "gc35_48h", "exceeds_benchmark", "ordered the accused's detention")
    analysis = analysis_with(record, absence=(counsel,), clock=(late,))
    found = profile(record, analysis, taxonomy, rejected_flag_ids={counsel.id})
    assert [(f.facet_id, f.flag_id) for f in found.facets if f.origin == "finding"] == [("gc35_48h", late.id)]
    assert profile(record, analysis, taxonomy, rejected_flag_ids=()) == profile(record, analysis, taxonomy)


def test_a_facet_rests_on_another_matching_finding_when_one_is_rejected(record, taxonomy):
    first = make_flag(record, "absence", "iccpr_14_3_b", "evidence_of_violation", "Defence counsel was not present")
    second = make_flag(record, "absence", "iccpr_14_3_b", "evidence_of_violation", "ordered the accused's detention")
    found = profile(record, analysis_with(record, absence=(first, second)), taxonomy, rejected_flag_ids=[first.id])
    [counsel] = [f for f in found.facets if f.origin == "finding"]
    assert (counsel.facet_id, counsel.flag_id, counsel.evidence) == ("iccpr_14_3_b", second.id, second.evidence)


def test_keyword_facets_carry_the_exact_sentence_from_the_indictment_first(record, taxonomy):
    found = {facet.facet_id: facet for facet in profile(record, analysis_with(record), taxonomy).facets}
    assert set(found) == {"media_defendant", "charge_false_information"}
    resolve = resolver_for([record])
    media, charge = found["media_defendant"], found["charge_false_information"]
    for facet in (media, charge):
        [evidence] = facet.evidence
        assert (facet.origin, facet.flag_id, evidence.role) == ("keyword", None, "mention")
        assert span_is_valid(evidence.span, resolve)
        assert evidence.span.doc_id == f"{record.case_id}/indictment.txt"  # not the note that comes first
    assert media.evidence[0].span.text == "Dana Orel, born in 1990, an editor of the news portal Testland Daily, residing in Testville."
    assert charge.evidence[0].span.text == "The accused is charged with spreading false news likely to cause alarm."


def test_keywords_match_whole_words_only(taxonomy):
    plain = make_record([("indictment.txt", "indictment", "The editorial line of the paper and the accused's reportage were discussed.")])
    assert profile(plain, analysis_with(plain), taxonomy).facets == ()


THEFT_NOTE = (
    "The accused's arrest was reported in a local newspaper.\n\n"
    "Defence counsel said the whole case was fake news invented by the police.\n\n"
    "A witness said the accused insulted her when she stopped him at the door."
)
THEFT_INDICTMENT = (
    "I. THE ACCUSED\n\n"
    "Tomas Brenn, born in 1988, a warehouse worker, residing in Testville.\n\n"
    "II. CHARGE\n\n"
    "The accused is charged with the theft of goods worth 300 euros from a shop."
)


def test_a_theft_case_gets_no_media_or_charge_facet_from_mere_mentions(taxonomy):
    """Press coverage, a defence claim and a witness's words in a note describe neither the accused nor the charge."""
    theft = titled(
        make_record([("notes/hearing_1.txt", "monitoring_note", THEFT_NOTE), ("indictment.txt", "indictment", THEFT_INDICTMENT)]),
        "Republic of Testland v. Tomas Brenn",
    )
    assert profile(theft, analysis_with(theft), taxonomy).facets == ()


def test_a_charge_facet_comes_only_from_an_indictment(taxonomy):
    """The record marks no charge recital in a judgment, so without an indictment there is no charge facet."""
    charge = "The accused is charged with spreading false news and with insulting the President."
    for doc_type in ("judgment", "monitoring_note", "detention_order", "transcript"):
        plain = make_record([(f"{doc_type}.txt", doc_type, charge)])
        assert profile(plain, analysis_with(plain), taxonomy).facets == (), doc_type
    indicted = make_record([("indictment.txt", "indictment", charge)])
    found = [facet.facet_id for facet in profile(indicted, analysis_with(indicted), taxonomy).facets]
    assert found == ["charge_false_information", "charge_defamation"]


@pytest.mark.parametrize(
    ("sentence", "described"),
    [
        ("The accused is a journalist.", True),
        ("The defendant works as a reporter for a weekly.", True),
        ("The author, a blogger, was arrested in 2020.", True),
        ("Ana Example, an editor, was arrested at dawn.", True),  # the surname from "... v. A. Example"
        ("Three journalists attended the hearing.", False),
        ("The editor of the weekly declined to comment.", False),
        ("The examples were given by a reporter.", False),  # the surname matches as a whole word, case and all
        ("The accused's arrest was reported in a local newspaper.", False),  # an outlet is not a media worker
        ("The accused gave an interview to a news portal.", False),
        ("The accused, a freelance journalist, was arrested.", True),
        ("The accused journalist was detained for six days.", True),
        ("The accused, born in 1988, is a reporter.", True),
        ("The accused works as an editor at a weekly.", True),
        ("Ana Example, born in 1990 in Testville, journalist, residing in Testville.", True),
        ("The accused is not a journalist.", False),
        ("The accused's brother is a journalist.", False),
        ("The accused, a lawyer, was attacked by a journalist.", False),
        ("The accused was interviewed by a reporter.", False),
        ("The accused was threatening a journalist.", False),
        ("Ana Example, who sued a reporter, was arrested.", False),
        ("The accused, a publisher's assistant, was arrested.", False),  # a possessive: the media word describes another
        ("The accused, born in 1988, attacked journalists.", False),
    ],
)
def test_the_media_facet_needs_a_sentence_about_the_accused(taxonomy, sentence, described):
    plain = make_record([("notes/hearing_1.txt", "monitoring_note", sentence)])
    found = {facet.facet_id: facet for facet in profile(plain, analysis_with(plain), taxonomy).facets}
    assert ("media_defendant" in found) is described
    if described:
        assert found["media_defendant"].evidence[0].span.text == sentence


def test_the_media_facet_skips_mentions_until_one_describes_the_accused(taxonomy):
    text = "Two journalists attended.\n\nThe accused, a reporter, did not attend."
    plain = make_record([("notes/hearing_1.txt", "monitoring_note", text)])
    [facet] = profile(plain, analysis_with(plain), taxonomy).facets
    assert facet.evidence[0].span.text == "The accused, a reporter, did not attend."


# --- incidental mentions in an indictment: a media word or an offence word that is not the charge ---

INCIDENTAL = (
    "The accused is charged with stealing a camera from a journalist.",
    "The accused stole a laptop belonging to a reporter.",
    "The accused broke into a publisher's warehouse.",
    "The accused terrorised the shop assistant.",
    "The accused insulted and pushed the shop assistant.",
    "The accused made false statements to the police.",
    "The accused called the complaint fake news.",
)
CHARGE_SECTION = "I. THE ACCUSED\n\nTomas Brenn, born in 1988, a warehouse worker, residing in Testville.\n\nII. CHARGE\n\n{}"


def indictment(text: str, title: str = "Republic of Testland v. Tomas Brenn"):
    return titled(make_record([("indictment.txt", "indictment", text)]), title)


def keyword_facets(record, taxonomy) -> dict[str, str]:
    """Keyword facet id -> the sentence it rests on."""
    return {f.facet_id: f.evidence[0].span.text for f in profile(record, analysis_with(record), taxonomy).facets if f.origin == "keyword"}


@pytest.mark.parametrize("layout", ["{}", CHARGE_SECTION], ids=["bare", "charge-section"])
@pytest.mark.parametrize("sentence", INCIDENTAL)
def test_an_object_a_victim_or_an_aside_in_an_indictment_gives_no_facet(taxonomy, sentence, layout):
    assert keyword_facets(indictment(layout.format(sentence)), taxonomy) == {}


def test_a_surname_that_is_also_a_common_word_is_not_the_accused_at_a_sentence_start(taxonomy):
    assert keyword_facets(indictment("Long queues of reporters waited outside the court.", "State v. Ana Long"), taxonomy) == {}
    worded = "The hearing was long.\n\nLong, a reporter, waited outside the court."  # the record writes "long" as a word
    assert keyword_facets(indictment(worded, "State v. Ana Long"), taxonomy) == {}
    full = "The hearing was long.\n\nAna Long, a reporter, waited outside the court."
    assert keyword_facets(indictment(full, "State v. Ana Long"), taxonomy) == {"media_defendant": "Ana Long, a reporter, waited outside the court."}
    rare = "The hearing was long.\n\nVenn, a reporter, waited outside the court."
    assert keyword_facets(indictment(rare, "State v. Ana Venn"), taxonomy) == {"media_defendant": "Venn, a reporter, waited outside the court."}


@pytest.mark.parametrize(
    ("text", "facet"),
    [
        ("The accused is charged with insulting the President.", "charge_defamation"),
        ("The accused is charged with having insulted a public official.", "charge_defamation"),
        ("The accused is charged with criminal defamation of a judge.", "charge_defamation"),
        ("The accused is charged with membership of a terrorist organisation.", "charge_extremism"),
        ("The accused is accused of separatism under Article 302 of the Penal Code.", "charge_extremism"),
        ("The accused is charged with the offence of spreading false news.", "charge_false_information"),
        ("II. CHARGE\n\nCriminal defamation of a public official, under Article 185 of the Penal Code.", "charge_defamation"),
        ("II. Charge\n\nThe accused committed criminal defamation, an offence under Article 185.", "charge_defamation"),
    ],
)
def test_a_charge_facet_names_the_offence_in_the_charge_wording(taxonomy, text, facet):
    assert set(keyword_facets(indictment(text), taxonomy)) == {facet}


@pytest.mark.parametrize(
    "text",
    [
        "The accused is charged with assault after he insulted a public official.",
        "The accused is charged with theft from a man who had spread false news.",
        "The accused is charged with the murder of a man accused of terrorism.",
        "The accused is charged with theft. He later called the case fake news.",
        "V. LEGAL QUALIFICATION\n\nArticle 185 of the Penal Code punishes the defamation of a public official.",
        "II. CHARGE\n\nThe accused is charged with theft under Article 205. He was acquitted of spreading false news under Article 214 in 2019.",
    ],
    ids=["after-clause", "relative-clause", "reduced-relative", "next-sentence", "law-recital", "charge-section-with-clause"],
)
def test_an_offence_word_outside_the_charge_itself_gives_no_facet(taxonomy, text):
    assert keyword_facets(indictment(text), taxonomy) == {}


# --- a charge the accused does not face: negated, withdrawn, past, or the object of another noun ---

NOT_FACED = {
    "negated": "The accused is not charged with defamation; he is charged with theft.",
    "no-longer": "The accused is no longer charged with separatism.",
    "never": "The accused has never been charged with defamation.",
    "contracted": "The accused isn't charged with libel.",
    "no-charge": "The accused faces no charges of extremism.",
    "withdrawn": "The charge of defamation was withdrawn on 2 May.",
    "dismissed": "The count of libel was dismissed by the court.",
    "dropped-aside": "The accused is charged with theft, the charges of extremism having been dropped.",
    "dropped-before": "The prosecution dropped the charges of extremism.",
    "previously-convicted": "The accused was previously convicted of the offence of spreading false news in 2015.",
    "previously-charged": "The accused was previously charged with defamation in 2012.",
    "acquitted-of-offence": "The accused was acquitted of the offence of spreading false news in 2019.",
    "used": "The accused is charged with theft of a phone used to spread false news.",
    "belonging": "The accused is charged with theft of a camera belonging to a journalist.",
    "containing": "The accused is charged with theft of a bag containing defamatory leaflets.",
    "relating": "The accused is charged with fraud relating to a terrorism investigation.",
    "concerning": "The accused is charged with forgery concerning a defamation settlement.",
    "about": "The accused is charged with fraud about false news insurance.",
    "against": "The accused is charged with assault against a separatist leader.",
    "of-a": "The accused is charged with the murder of a separatist leader.",
    "later-dropped": "The accused was charged with defamation, but the charge was later dropped.",
    "semicolon-withdrawn": "The accused was charged with defamation; the charge was withdrawn.",
    "acquitted-in-a-relative": "The accused, who was acquitted of false news charges in 2019, is charged with theft.",
    "and-not": "The accused is charged with theft and not with extremism.",
    "dropped-then-charged": "The prosecution dropped the extremism charge and charged the accused with fraud.",
    "year-before-the-wording": "In 2019 the accused was charged with defamation.",
}


@pytest.mark.parametrize("layout", ["{}", CHARGE_SECTION], ids=["bare", "charge-section"])
@pytest.mark.parametrize("sentence", NOT_FACED.values(), ids=NOT_FACED.keys())
def test_a_charge_the_accused_does_not_face_gives_no_facet(taxonomy, sentence, layout):
    """Not charged, a charge withdrawn or dropped, a past conviction or charge, or an offence word that is
    the object of another noun ("a phone used to spread false news"): none is the charge the accused faces."""
    assert keyword_facets(indictment(layout.format(sentence)), taxonomy) == {}


NOT_FACED_CUES = {
    "cannot": "The accused cannot be charged with defamation.",
    "neither-nor": "The accused is charged with neither defamation nor libel.",
    "withdrew": "The accused was charged with defamation, but the prosecutor withdrew it.",
    "quashed": "The accused was charged with defamation, but the indictment was quashed.",
    "struck-out": "The accused was charged with defamation, but the count was struck out.",
    "abandoned": "The accused was charged with defamation, a charge since abandoned.",
    "cleared": "The accused was charged with defamation and later cleared.",
    "discontinued": "The accused was charged with defamation until the case was discontinued.",
    "formerly": "The accused was formerly charged with defamation.",
    "prior": "The accused is charged with defamation in a prior case.",
    "past": "The accused was charged with defamation in the past.",
    "earlier": "The accused was charged with defamation earlier.",
}


@pytest.mark.parametrize("layout", ["{}", CHARGE_SECTION], ids=["bare", "charge-section"])
@pytest.mark.parametrize("sentence", NOT_FACED_CUES.values(), ids=NOT_FACED_CUES.keys())
def test_every_not_faced_cue_vetoes_the_charge(taxonomy, sentence, layout):
    assert keyword_facets(indictment(layout.format(sentence)), taxonomy) == {}


@pytest.mark.parametrize(
    ("text", "facets"),
    [
        ("The accused is charged with spreading false news. The charge of defamation was withdrawn.", {"charge_false_information"}),
        ("The accused is charged with defamation. The charges of extremism were dropped.", {"charge_defamation"}),
        ("The accused, previously convicted of theft, is now charged with defamation.", {"charge_defamation"}),
        ("The accused is not charged with theft; he is charged with separatism.", {"charge_extremism"}),
        ("The accused is charged with the publication of a defamatory article.", {"charge_defamation"}),
        ("The accused is charged with offences relating to terrorism.", {"charge_extremism"}),
        ("The accused is charged with several offences, including separatism.", {"charge_extremism"}),
        ("The accused is charged with libel against a public official.", {"charge_defamation"}),
        ("The accused is charged with insulting of the President.", {"charge_defamation"}),
        ("The accused is charged with the creation of an extremist community.", {"charge_extremism"}),
        ("The accused is charged with the spreading of a false news story.", {"charge_false_information"}),
        ("The accused is charged with possession of the extremist materials listed in Annex 2.", {"charge_extremism"}),
        ("The accused is charged with spreading false news in 2024.", {"charge_false_information"}),
        ("The accused is charged with membership of a terrorist organisation under Article 7(2) of Law No. 3713.", {"charge_extremism"}),
    ],
    ids=["withdrawn-next-sentence", "dropped-next-sentence", "past-then-present", "negated-then-present", "publication-of-a",
         "offences-relating", "offences-including", "against-in-keyword-tail", "of-the-in-keyword", "creation-of-an",
         "spreading-of-a", "possession-of-the", "year-after-the-wording", "law-number"],  # fmt: skip
)
def test_the_charge_faced_is_kept_beside_one_that_is_not(taxonomy, text, facets):
    """A cue in another sentence or another clause before the wording does not veto it; nor does the year of
    the offence after the wording, or "No." before a number ("Law No. 3713")."""
    assert set(keyword_facets(indictment(text), taxonomy)) == facets


@pytest.mark.parametrize("layout", ["{}", CHARGE_SECTION], ids=["bare", "charge-section"])
@pytest.mark.parametrize(
    "sentence",
    [
        "The accused is charged with spreading false news; the charge of defamation was withdrawn.",
        "The accused is charged with defamation, the charges of extremism having been dropped.",
    ],
    ids=["withdrawn-after-a-semicolon", "dropped-in-an-aside"],
)
def test_a_cue_later_in_the_sentence_vetoes_even_a_charge_the_accused_faces(taxonomy, sentence, layout):
    """Conservative on purpose: a missed charge facet costs little, a false one misstates the case. So a
    negation, a withdrawal or a past marker anywhere after the offence word in its sentence vetoes it."""
    assert keyword_facets(indictment(layout.format(sentence)), taxonomy) == {}


def test_the_demo_indictment_names_a_journalist_charged_with_false_information(taxonomy):
    """The demo case's indictment on its own (no model): the accused is a journalist, and the charge is
    disseminating false information; the theft-like probes above share none of its wording, and its charge
    sentence holds no not-faced cue, so the conservative veto keeps both facets."""
    from ratio.paths import DEMO_CASE_DIR

    text = (DEMO_CASE_DIR / "indictment.txt").read_text(encoding="utf-8")
    assert keyword_facets(indictment(text, "Republic of Calderra v. Daro Venn"), taxonomy) == {
        "media_defendant": "Daro Venn, born on 11 May 1987 in Halvar, journalist, residing in Mirevo.",
        "charge_false_information": "The accused is charged with disseminating false information likely to cause public alarm, "
        "an offence under Article 214(2) of the Penal Code.",
    }


@pytest.mark.parametrize(
    ("title", "surname"),
    [
        ("Republic of Calderra v. Daro Venn", "Venn"),
        ("Republic of Testland v. A. Example", "Example"),
        ("State v. Ana Lind and others", "Lind"),
        ("State v. Ana Lind, Petar Lund", "Lind"),
        ("State v. Ana Lind (appeal)", "Lind"),
        ("Fairness report on the trial of Ana Lind", None),
    ],
)
def test_the_defendant_surname_is_read_from_the_case_title(title, surname):
    assert _defendant_surname(title) == surname


def test_profile_follows_the_taxonomy_order(record, taxonomy):
    late = make_flag(record, "clock", "gc35_48h", "exceeds_benchmark", "ordered the accused's detention")
    counsel = make_flag(record, "absence", "iccpr_14_3_b", "evidence_of_violation", "Defence counsel was not present")
    found = profile(record, analysis_with(record, absence=(counsel,), clock=(late,)), taxonomy)
    order = [facet.id for facet in taxonomy.facets]
    ids = [facet.facet_id for facet in found.facets]
    assert ids == sorted(ids, key=order.index) == ["iccpr_14_3_b", "gc35_48h", "media_defendant", "charge_false_information"]


@pytest.mark.parametrize(
    ("text", "phrase", "sentence"),
    [
        ("First one. The accused is a journalist here. Last one.", "journalist", "The accused is a journalist here."),
        ("A heading\n\nThe reporter wrote it", "reporter", "The reporter wrote it"),
        ("Alpha. Beta gamma.\n\nDelta editor epsilon. Zeta.", "editor", "Delta editor epsilon."),
    ],
)
def test_sentence_bounds_without_the_extraction_layer(text, phrase, sentence):
    start = text.index(phrase)
    low, high = _sentence_bounds(text, start, start + len(phrase))
    assert text[low:high] == sentence


# --- link: which precedents, in which order ----------------------------------------------------


def test_links_are_ranked_by_the_rarity_of_shared_facets(corpus_index, taxonomy):
    links = link(DEMO_LIKE, corpus_index, FakeEmbedder(), SETTINGS, taxonomy)
    assert [found.precedent.id for found in links] == list(LINKED)
    freq, n = corpus_index.facet_doc_freq(), corpus_index.meta().documents
    for found in links:
        shared = [s.facet_id for s in found.shared]
        assert found.score.shared == len(shared)
        assert found.score.idf_sum == pytest.approx(sum(idf(freq[f], n) for f in shared))
    assert [s.facet_id for s in links[0].shared] == ["iccpr_14_3_b", "gc35_48h", "media_defendant", "charge_false_information"]
    assert not set(NOT_LINKED) & {found.precedent.id for found in links}


def test_every_precedent_span_shown_is_its_stored_text(corpus_index, taxonomy):
    for found in link(DEMO_LIKE, corpus_index, FakeEmbedder(), SETTINGS, taxonomy):
        for shared in found.shared:
            assert shared.label == taxonomy.facet(shared.facet_id).label
            assert span_is_valid(shared.precedent_fact.span, corpus_index.text)
            assert shared.precedent_fact.span.case_id == f"precedent-{found.precedent.id}"
            if shared.precedent_finding is not None:
                assert span_is_valid(shared.precedent_finding.span, corpus_index.text)


def test_the_closest_fact_is_paired_with_its_cosine(corpus_index, taxonomy):
    top = link(DEMO_LIKE, corpus_index, FakeEmbedder(), SETTINGS, taxonomy)[0]
    gc35 = next(s for s in top.shared if s.facet_id == "gc35_48h")
    assert gc35.precedent_fact.span.text == "She was arrested at her home and was brought before a judge six days later."
    assert 0.0 < gc35.pair_cosine <= 1.0
    cosines = [s.pair_cosine for s in top.shared]
    assert top.score.mean_pair_cosine == round(sum(cosines) / len(cosines), COSINE_DECIMALS)
    assert gc35.finding_kind == "monitor_assessment" and gc35.precedent_finding is not None


def test_cosines_are_kept_to_the_precision_the_two_backends_agree_on(corpus_index, taxonomy):
    assert COSINE_DECIMALS == 4
    for found in link(DEMO_LIKE, corpus_index, FakeEmbedder(), SETTINGS, taxonomy):
        assert found.score.mean_pair_cosine == round(found.score.mean_pair_cosine, COSINE_DECIMALS)
        assert all(s.pair_cosine == round(s.pair_cosine, COSINE_DECIMALS) for s in found.shared)


def test_float32_noise_in_a_cosine_does_not_reorder_links(taxonomy):
    """The backends' cosines differ by about 1e-7 (float32). Here that noise straddles a sixth decimal:
    kept to four, the cosines tie and the precedent id decides, whichever backend answered."""
    entries = [make_precedent(pid, {"iccpr_14_3_b": "No lawyer.", "gc35_48h": "Late before a judge."}) for pid in ("ccpr-tie-a", "ccpr-tie-b")]

    def ranked(cosines: dict[str, float]) -> list[str]:
        nearest = {(facet.facet_id, doc.id): (facet.facts[0], cosines[doc.id]) for doc, facets in entries for facet in facets}
        return [found.precedent.id for found in link(DEMO_LIKE, MemoryIndex(entries, nearest=nearest), FakeEmbedder(), SETTINGS, taxonomy)]

    assert ranked({"ccpr-tie-a": 0.7123406, "ccpr-tie-b": 0.7123404}) == ranked({"ccpr-tie-a": 0.7123404, "ccpr-tie-b": 0.7123406})


def test_profile_and_charge_facets_alone_never_link(corpus_index, taxonomy):
    loose = PrecedentSettings(backend="sqlite", min_shared=1)
    profile_only = CaseProfile(case_id="test-case", facets=tuple(f for f in DEMO_LIKE.facets if f.origin == "keyword"))
    assert link(profile_only, corpus_index, FakeEmbedder(), loose, taxonomy) == ()
    ids = {found.precedent.id for found in link(DEMO_LIKE, corpus_index, FakeEmbedder(), loose, taxonomy)}
    assert "tw-synthetic-d" not in ids  # it shares two facets, neither about procedure
    assert "ccpr-synthetic-f" in ids  # one shared procedural facet is enough when min_shared is 1


def test_min_shared_is_applied(corpus_index, taxonomy):
    strict = PrecedentSettings(backend="sqlite", min_shared=4)
    assert [found.precedent.id for found in link(DEMO_LIKE, corpus_index, FakeEmbedder(), strict, taxonomy)] == ["tw-synthetic-a"]


def test_a_precedent_whose_text_changed_is_dropped(corpus_index, taxonomy):
    """tw-synthetic-e shares five facets but its stored text no longer matches its verified quotes."""
    assert TAMPERED not in {found.precedent.id for found in link(DEMO_LIKE, corpus_index, FakeEmbedder(), SETTINGS, taxonomy)}


def test_a_tampered_fact_drops_its_facet_and_a_tampered_finding_becomes_none(taxonomy):
    doc, facets = make_precedent(
        "ccpr-test-1",
        {"iccpr_14_3_b": "No lawyer was present at the remand hearing.", "gc35_48h": "He was brought before a judge after six days.",
         "media_defendant": "He is a journalist."},
        {"gc35_48h": "The Committee found a violation of article 9(3)."},
    )  # fmt: skip
    texts = {doc.doc_id: doc.text.replace("No lawyer", "An lawyer").replace("found a", "found X")}  # same lengths
    found = link(DEMO_LIKE, MemoryIndex([(doc, facets)], texts=texts), FakeEmbedder(), SETTINGS, taxonomy)
    [only] = found
    assert [s.facet_id for s in only.shared] == ["gc35_48h", "media_defendant"]
    assert only.shared[0].precedent_finding is None
    texts = {doc.doc_id: doc.text.replace("No lawyer", "An lawyer").replace("He is a", "He is X")}
    assert link(DEMO_LIKE, MemoryIndex([(doc, facets)], texts=texts), FakeEmbedder(), SETTINGS, taxonomy) == ()  # one left


def test_without_a_nearest_fact_the_first_one_is_shown_without_a_cosine(taxonomy):
    doc, facets = make_precedent("ccpr-test-1", {"iccpr_14_3_b": "No lawyer was present.", "iccpr_14_3_e": "No witness was heard."})
    [only] = link(DEMO_LIKE, MemoryIndex([(doc, facets)]), FakeEmbedder(), SETTINGS, taxonomy)
    assert [s.precedent_fact for s in only.shared] == [facets[0].facts[0], facets[1].facts[0]]
    assert [s.pair_cosine for s in only.shared] == [None, None]
    assert only.score.mean_pair_cosine is None


def test_a_rare_shared_facet_outranks_common_ones(taxonomy):
    common = make_precedent("ccpr-common", {"gc35_48h": "Late before a judge.", "iccpr_14_3_b": "No lawyer.", "media_defendant": "A journalist."})
    rare = make_precedent("ccpr-rare", {"renewal_review": "Renewed on the same grounds.", "unaddressed_defense_argument": "The defence was not answered."})
    doc_freq = {"gc35_48h": 9, "iccpr_14_3_b": 9, "media_defendant": 9, "renewal_review": 1, "unaddressed_defense_argument": 1}
    index = MemoryIndex([common, rare], doc_freq=doc_freq, documents=10)
    links = link(DEMO_LIKE, index, FakeEmbedder(), SETTINGS, taxonomy)
    assert [found.precedent.id for found in links] == ["ccpr-rare", "ccpr-common"]
    assert links[0].score.idf_sum == pytest.approx(2 * idf(1, 10))
    assert links[1].score.idf_sum == pytest.approx(3 * idf(9, 10))


def test_ties_are_broken_the_same_way_whatever_the_candidate_order(taxonomy):
    entries = [make_precedent(pid, {"iccpr_14_3_b": "No lawyer.", "gc35_48h": "Late before a judge."}) for pid in ("ccpr-tie-a", "ccpr-tie-b", "ccpr-tie-c")]
    forward = link(DEMO_LIKE, MemoryIndex(entries), FakeEmbedder(), SETTINGS, taxonomy)
    backward = link(DEMO_LIKE, MemoryIndex(entries[::-1]), FakeEmbedder(), SETTINGS, taxonomy)
    assert [found.precedent.id for found in forward] == [found.precedent.id for found in backward]
    assert [found.rank_key for found in forward] == sorted((found.rank_key for found in forward), reverse=True)


def test_top_k_limits_the_links(corpus_index, taxonomy):
    one = PrecedentSettings(backend="sqlite", top_k=1)
    assert [found.precedent.id for found in link(DEMO_LIKE, corpus_index, FakeEmbedder(), one, taxonomy)] == [LINKED[0]]


def test_a_case_without_facets_links_nothing(corpus_index, taxonomy):
    assert link(CaseProfile(case_id="test-case"), corpus_index, FakeEmbedder(), SETTINGS, taxonomy) == ()


def test_a_quote_from_another_precedent_is_never_paired(taxonomy):
    first = make_precedent("ccpr-one", {"iccpr_14_3_b": "No lawyer.", "gc35_48h": "Late."})
    second = make_precedent("ccpr-two", {"iccpr_14_3_b": "Counsel absent.", "gc35_48h": "Six days."})
    wrong = {("iccpr_14_3_b", "ccpr-one"): (second[1][0].facts[0], 0.9)}  # an index answering with the wrong precedent's quote
    links = link(DEMO_LIKE, MemoryIndex([first, second], nearest=wrong), FakeEmbedder(), SETTINGS, taxonomy)
    assert [found.precedent.id for found in links] == ["ccpr-two"]  # ccpr-one keeps one facet: below min_shared
    assert all(s.precedent_fact.span.doc_id == precedent_doc_id("ccpr-two") for s in links[0].shared)


# --- the demo case (Calderra), replayed from the committed model cache --------------------------


@pytest.fixture(scope="module")
def demo():
    from ratio.extraction.build import load_case
    from ratio.paths import DEMO_CASE_DIR
    from ratio.pipeline import analysis_context, analyze, demo_llm, ingest

    config = load_config()
    llm = demo_llm(config)
    record, _ = ingest(load_case(DEMO_CASE_DIR), llm, config)
    return record, analyze(record, analysis_context(config, llm))


@needs_minilm
def test_the_demo_profile(demo, taxonomy):
    record, analysis = demo
    found = profile(record, analysis, taxonomy)
    assert [facet.facet_id for facet in found.facets] == [
        "iccpr_14_3_b", "iccpr_14_3_e", "gc35_48h", "reasoning_reuse", "unaddressed_defense_argument", "renewal_review",
        "media_defendant", "charge_false_information",
    ]  # fmt: skip
    keyword = {facet.facet_id: facet.evidence[0].span for facet in found.facets if facet.origin == "keyword"}
    assert keyword["media_defendant"].text == "Daro Venn, born on 11 May 1987 in Halvar, journalist, residing in Mirevo."
    assert keyword["charge_false_information"].text.startswith("The accused is charged with disseminating false information")
    assert {span.doc_id for span in keyword.values()} == {"venn-2025/indictment.txt"}
    resolve = resolver_for([record])
    assert all(span_is_valid(e.span, resolve) for facet in found.facets for e in facet.evidence)
    by_flag = {flag.id: flag for flag in analysis.all_flags()}
    assert all(by_flag[f.flag_id].status not in {"evidence_of_compliance", "needs_review"} for f in found.facets if f.flag_id)


@needs_minilm
def test_the_demo_links_to_the_fixture_corpus(demo, corpus_index, taxonomy):
    record, analysis = demo
    links = link(profile(record, analysis, taxonomy), corpus_index, FakeEmbedder(), SETTINGS, taxonomy)
    assert [found.precedent.id for found in links] == list(LINKED)


# --- in the dashboard, and in the installed package ---------------------------------------------


def test_the_index_opens_from_a_background_thread_without_a_streamlit_warning():
    """The Case page opens the index from a worker thread, which has no ScriptRunContext: a spinner there
    only logs "missing ScriptRunContext" on every open."""
    import logging
    import threading

    from ratio_ui import session

    class Collect(logging.Handler):
        def __init__(self):
            super().__init__(logging.DEBUG)
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    collect = Collect()
    logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
    logger.addHandler(collect)
    session.clear_precedent_index()
    try:
        worker = threading.Thread(target=session.precedent_index)
        worker.start()
        worker.join()
    finally:
        logger.removeHandler(collect)
        session.clear_precedent_index()
    assert not [message for message in collect.messages if "missing ScriptRunContext" in message]


def test_the_builder_sources_file_ships_with_the_package():
    import tomllib

    from ratio.paths import REPO_ROOT

    package_data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]
    assert "sources.yaml" in package_data["corpus_builder"]
    assert (REPO_ROOT / "corpus_builder" / "sources.yaml").is_file()
