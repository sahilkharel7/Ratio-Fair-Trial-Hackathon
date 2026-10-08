"""Absence Detector: label validation and status rules on the demo record.

A rule-based "careful reader" stands in for the model, so the status rules are tested end to end
without Ollama. The real model's labels are tested on the replayed demo cache (test_demo_story).
"""

import datetime as dt
import re

import pytest

from ratio.config import load_config
from ratio.context import AnalysisContext
from ratio.embeddings import MiniLMEmbedder
from ratio.modules import absence
from ratio.modules.absence_prompts import LabelReply
from ratio.paths import GOLD_DIR, MINILM_DIR
from ratio.schema import CaseRecord
from ratio.testing import FakeEmbedder, FakeLLM, make_record

CONFIG = load_config()
# The status-rule tests label every note (a wide shortlist), so they do not depend on how a test
# embedder ranks notes. Shortlist ranking is tested separately with the real MiniLM model.
WIDE = CONFIG.model_copy(
    update={
        "settings": CONFIG.settings.model_copy(
            update={"absence": CONFIG.settings.absence.model_copy(update={"shortlist_top_k": 100, "per_hearing_top_k": 100, "shortlist_min_similarity": -1.0, "max_shortlist": 1000})}
        )
    }
)
MOCK = CaseRecord.model_validate_json((GOLD_DIR / "mock_record.json").read_text(encoding="utf-8"))
EXPECTED = {
    "iccpr_14_3_a": "no_evidence",
    "iccpr_14_3_b": "evidence_of_violation",
    "iccpr_14_3_c": "no_evidence",
    "iccpr_14_3_d": "evidence_of_compliance",
    "iccpr_14_3_e": "evidence_of_violation",
    "iccpr_14_3_f": "no_evidence",
    "iccpr_14_3_g": "evidence_of_compliance",
}

# (pattern in the observation, label, indicator id): how a careful reader labels the planted facts.
CAREFUL_READER = [
    (r"Mr\. Venn was present in the (courtroom|dock)", "supports", "d_defendant_present"),
    (r"Defence counsel Lena Brask.*was present", "supports", "d_counsel_present"),
    (r"read out the charge", "supports", "a_charge_communicated"),
    (r"case file only three days", "contradicts", "b_preparation_inadequate"),
    (r"refused to hear them", "contradicts", "e_defence_witnesses_refused"),
    (r"cross-examined both witnesses", "supports", "e_prosecution_witnesses_examined"),
    (r"right to remain silent", "supports", "g_silence_respected"),
]


def observations_in(user: str) -> dict[str, str]:
    return dict(re.findall(r"^(O\d+) \([^)]*\): (.*)$", user, flags=re.MULTILINE))


def careful_reader(rules=CAREFUL_READER, note="Matches the indicator."):
    def respond(system, user, schema, purpose):
        assert schema is LabelReply and purpose == "labels"
        labels = []
        for obs_id, text in observations_in(user).items():
            match = next(((lab, ind) for pat, lab, ind in rules if re.search(pat, text) and f"{ind}:" in user), None)
            label, indicator = match or ("unrelated", "")
            labels.append({"observation": obs_id, "label": label, "indicator_id": indicator, "note": note})
        return {"labels": labels}

    return respond


def run(record=MOCK, responder=None, embedder=None, config=WIDE):
    llm = FakeLLM(responder or careful_reader())
    ctx = AnalysisContext(llm=llm, embedder=embedder or FakeEmbedder(), config=config)
    return absence.run(record, ctx), llm


def statuses(result) -> dict[str, str]:
    return {a.rubric_id: a.status for a in result.assessments}


def assessment(result, rubric_id):
    return next(a for a in result.assessments if a.rubric_id == rubric_id)


def test_demo_statuses_follow_the_rules():
    result, llm = run()
    assert statuses(result) == EXPECTED
    for item in CONFIG.rubric.items:  # the wide shortlist puts every note in front of the model, 30 at a time
        prompts = [call.user for call in llm.calls if f", {item.name}" in call.user.splitlines()[0]]
        shown = [len(observations_in(prompt)) for prompt in prompts]
        assert sum(shown) == len(MOCK.observations) and max(shown) <= 30, item.id


def test_violation_and_compliance_are_flags_and_no_evidence_is_a_follow_up():
    result, _ = run()
    flagged = {f.standard_id: f.status for f in result.flags}
    assert flagged == {k: v for k, v in EXPECTED.items() if v != "no_evidence"}
    assert {u.rubric_id for u in result.follow_ups} == {k for k, v in EXPECTED.items() if v == "no_evidence"}
    for flag in result.flags:
        assert flag.evidence and flag.review_status == "needs_legal_review"
        assert "ICCPR Art. 14(3)" in flag.citation


def test_violation_flag_points_at_the_contradicting_note():
    result, _ = run()
    (flag,) = [f for f in result.flags if f.standard_id == "iccpr_14_3_b"]
    assert {e.role for e in flag.evidence} == {"contradicting"}
    assert any("case file only three days" in e.span.text for e in flag.evidence)


def test_interpreter_follow_up_lists_every_hearing_without_evidence():
    result, _ = run()
    follow_up = assessment(result, "iccpr_14_3_f").follow_up
    assert follow_up is not None
    assert "interpreter" in follow_up.question.lower()
    assert len([u for u in follow_up.uncovered if re.search(r"\d{4}", u)]) == 4


def test_a_hearing_without_evidence_blocks_per_hearing_compliance():
    rules = [r for r in CAREFUL_READER if r[2] != "d_defendant_present"] + [
        (r"Mr\. Venn was present in the (courtroom|dock)(?! when the hearing opened)", "supports", "d_defendant_present")
    ]
    result, _ = run(responder=careful_reader(rules))
    d = assessment(result, "iccpr_14_3_d")
    assert d.status == "no_evidence"
    (presence,) = [p for p in d.parts if p.part_id == "d_presence"]
    assert presence.hearings_missing == (dt.date(2025, 6, 30),)


def test_contradiction_wins_over_support():
    rules = [(r"Mr\. Venn was present in the dock\.$", "contradicts", "d_defendant_absent"), *CAREFUL_READER]
    result, _ = run(responder=careful_reader(rules))
    assert assessment(result, "iccpr_14_3_d").status == "evidence_of_violation"


def test_labels_citing_the_wrong_kind_of_indicator_are_ignored():
    wrong = [(pat, "supports" if lab == "contradicts" else "contradicts", ind) for pat, lab, ind in CAREFUL_READER]
    result, _ = run(responder=careful_reader(wrong))
    assert set(statuses(result).values()) == {"no_evidence"}
    assert result.flags == ()


def test_a_label_on_a_note_that_never_mentions_the_subject_is_ignored():
    rules = [(r"refused the request and ordered the trial to continue", "contradicts", "f_interpreter_refused"), *CAREFUL_READER]
    result, _ = run(responder=careful_reader(rules))
    assert assessment(result, "iccpr_14_3_f").status == "no_evidence"


def test_unknown_observation_ids_are_ignored():
    def invented(system, user, schema, purpose):
        return {"labels": [{"observation": "O999", "label": "contradicts", "indicator_id": "f_interpreter_missing", "note": ""}]}

    result, _ = run(responder=invented)
    assert assessment(result, "iccpr_14_3_f").status == "no_evidence"


def test_observation_ids_are_read_even_when_the_model_copies_the_whole_line():
    def copies_line(system, user, schema, purpose):
        reply = careful_reader()(system, user, schema, purpose)
        lines = {m.group(1): m.group(0) for m in re.finditer(r"^(O\d+) \(.*$", user, flags=re.MULTILINE)}
        return {"labels": [{**label, "observation": lines[label["observation"]]} for label in reply["labels"]]}

    result, _ = run(responder=copies_line)
    assert statuses(result) == EXPECTED


def test_model_notes_are_filtered_for_characterisations_of_people():
    result, _ = run(responder=careful_reader(note="The judge was biased and refused."))
    notes = [obs.model_note for a in result.assessments for obs in a.supporting + a.contradicting]
    assert notes and all("biased" not in (n or "") for n in notes)
    assert all(f.model_note is None or "biased" not in f.model_note for f in result.flags)


def test_a_record_without_notes_gives_follow_ups_only_and_no_model_calls():
    bare = MOCK.model_copy(update={"observations": ()})
    result, llm = run(record=bare)
    assert set(statuses(result).values()) == {"no_evidence"}
    assert llm.calls == ()


needs_model = pytest.mark.skipif(not (MINILM_DIR / "modules.json").exists(), reason="run scripts/fetch_models.py once")


@pytest.mark.embed
@needs_model
def test_real_shortlists_reach_the_planted_notes_of_every_hearing():
    result, _ = run(embedder=MiniLMEmbedder(), config=CONFIG)
    assert statuses(result) == EXPECTED


@pytest.mark.embed
@needs_model
def test_language_fact_is_shown_as_context_for_the_interpreter_follow_up():
    result, _ = run(embedder=MiniLMEmbedder(), config=CONFIG)
    f = assessment(result, "iccpr_14_3_f")
    assert f.status == "no_evidence"
    assert any("first language is Ostric" in e.span.text for e in f.context)
    assert all(e.role == "context" for e in f.context)


@pytest.mark.embed
@needs_model
def test_charge_follow_up_shows_the_language_fact_not_a_merely_similar_note():
    result, _ = run(embedder=MiniLMEmbedder(), config=CONFIG)
    a = assessment(result, "iccpr_14_3_a")
    assert a.status == "no_evidence"
    assert [e.span.text for e in a.context] == [
        next(o.text for o in MOCK.observations if "first language is Ostric" in o.text)
    ]  # "the court would rule on the request later" scores higher with MiniLM but mentions no language


# --- review fixes: undated notes, the shortlist cap, context facts, skipped notes, grounding ---------


def undated(record: CaseRecord) -> CaseRecord:
    """The demo record with the hearing dates of its notes removed (as with a note lacking the header)."""
    observations = tuple(obs.model_copy(update={"hearing_date": None}) for obs in record.observations)
    return record.model_copy(update={"observations": observations})


def test_undated_notes_are_each_their_own_hearing_for_per_hearing_parts():
    only_first = [(r"Mr\. Venn was present in the courtroom throughout", "supports", "d_defendant_present"),
                  (r"Lena Brask, whom Mr\. Venn had retained", "supports", "d_counsel_present")]  # fmt: skip
    result, _ = run(record=undated(MOCK), responder=careful_reader(only_first))
    d = assessment(result, "iccpr_14_3_d")
    assert d.status == "no_evidence"  # one note's evidence does not cover the other three notes
    presence = next(p for p in d.parts if p.part_id == "d_presence")
    assert len(presence.notes_missing) == 3 and presence.hearings_missing == ()
    assert any("(no hearing date)" in line for line in d.follow_up.uncovered)


def test_the_shortlist_cap_keeps_every_hearings_keyword_note():
    hearings = 12
    documents = [
        (f"note_{n}.txt", "monitoring_note",
         f"Hearing date: {n} March 2025\n\nThe hearing began at 10:{n:02d}. Mr. Venn sat in the dock for the whole session. "
         "Two witnesses testified. Counsel asked questions. The hearing ended at 14:00.")
        for n in range(1, hearings + 1)
    ]  # fmt: skip
    record = make_record(documents)
    capped = CONFIG.model_copy(
        update={"settings": CONFIG.settings.model_copy(update={"absence": CONFIG.settings.absence.model_copy(update={"max_shortlist": 20})})}
    )
    rules = [(r"sat in the dock for the whole session", "supports", "d_defendant_present")]
    _, llm = run(record=record, responder=careful_reader(rules), config=capped)
    d_prompts = [call.user for call in llm.calls if "14(3)(d)" in call.user.splitlines()[0]]
    dock_notes = sum("sat in the dock" in text for prompt in d_prompts for text in observations_in(prompt).values())
    assert dock_notes == hearings


def test_a_contradiction_after_a_support_for_the_same_note_wins():
    def both(system, user, schema, purpose):
        labels = []
        for obs_id, text in observations_in(user).items():
            if "present in the dock when the hearing opened" in text and "d_defendant_absent:" in user:
                labels.append({"observation": obs_id, "label": "supports", "indicator_id": "d_defendant_present", "note": ""})
                labels.append({"observation": obs_id, "label": "contradicts", "indicator_id": "d_defendant_absent", "note": ""})
            else:
                labels.append({"observation": obs_id, "label": "unrelated", "indicator_id": "", "note": ""})
        return {"labels": labels}

    result, _ = run(responder=both)
    assert assessment(result, "iccpr_14_3_d").status == "evidence_of_violation"


def test_notes_the_model_skips_are_asked_about_again():
    asked = []

    def forgetful_once(system, user, schema, purpose):
        asked.append(user)
        reply = careful_reader()(system, user, schema, purpose)
        first_time = sum(user.splitlines()[0] == earlier.splitlines()[0] for earlier in asked) == 1
        return {"labels": reply["labels"][::2]} if first_time else reply  # the first reply skips every other note

    result, _ = run(responder=forgetful_once)
    assert statuses(result) == EXPECTED
    assert all(a.unlabelled_notes == 0 for a in result.assessments)


def test_a_guarantee_is_not_compliant_while_notes_stay_unlabelled():
    def never_finishes(system, user, schema, purpose):
        reply = careful_reader()(system, user, schema, purpose)
        return {"labels": [label for label in reply["labels"] if label["label"] != "unrelated"]}

    result, _ = run(responder=never_finishes)
    g = assessment(result, "iccpr_14_3_g")
    assert g.status == "no_evidence" and g.unlabelled_notes > 0
    assert any("not labelled by the model" in line for line in g.follow_up.uncovered)
    assert assessment(result, "iccpr_14_3_e").status == "evidence_of_violation"  # a contradiction still counts


def test_a_label_is_not_grounded_by_a_sentence_from_another_paragraph():
    # The note before "read out the charge" (another paragraph) says the first language is Ostric.
    rules = [(r"read out the charge", "supports", "a_charge_in_understood_language")]
    result, _ = run(responder=careful_reader(rules))
    a = assessment(result, "iccpr_14_3_a")
    assert not any(label.indicator_id == "a_charge_in_understood_language" for label in a.supporting)


@pytest.mark.embed
@needs_model
def test_a_context_fact_is_never_counted_as_evidence():
    rules = [(r"first language is Ostric", "supports", "f_understands_court_language")]
    result, _ = run(responder=careful_reader(rules), embedder=MiniLMEmbedder(), config=CONFIG)
    f = assessment(result, "iccpr_14_3_f")
    assert f.status == "no_evidence" and f.supporting == ()


def test_a_rubric_item_needs_a_required_part():
    from pydantic import ValidationError

    from ratio.config import RubricItem

    item = CONFIG.rubric_item("iccpr_14_3_g").model_dump()
    item["parts"] = [{**part, "required": False} for part in item["parts"]]
    with pytest.raises(ValidationError, match="required"):
        RubricItem.model_validate(item)
