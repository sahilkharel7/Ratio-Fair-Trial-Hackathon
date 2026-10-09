"""Tests that call the real local model. They are skipped by default; run them with Ollama started:

    pytest -m live

They show that the recorded demo answers are real: the model, asked again now on this computer,
gives the same answer, and every item it returns is found exactly in the source text."""

import pytest

from ratio import preflight
from ratio.config import default_config
from ratio.extraction.build import load_case
from ratio.llm import OllamaClient
from ratio.paths import DEMO_CASE_DIR
from ratio.pipeline import demo_llm, ingest, read_demo_manifest, run_note_live

pytestmark = pytest.mark.live

CONFIG = default_config()
MODEL = read_demo_manifest().model


def test_the_recorded_model_is_pulled_and_served_on_this_computer():
    client = OllamaClient(CONFIG.settings.llm, model=MODEL)
    assert client.version()
    client.check_model()  # raises unless the model is pulled and local (not an Ollama cloud model)
    assert preflight.ollama_server(CONFIG).status == "ok"


def test_a_note_read_again_live_gives_the_recorded_answer_with_exact_sources():
    record, _ = ingest(load_case(DEMO_CASE_DIR), demo_llm(CONFIG), CONFIG)
    note = record.documents_of_type("monitoring_note")[0]
    live = run_note_live(record, note.id, CONFIG, model=MODEL)
    assert live.events, "the model found no dated event in the first hearing note"
    for item in (*live.events, *live.arguments):
        assert note.text[item.span.start : item.span.end] == item.span.text  # hard rule 2: the exact source text
    assert live.same_as_record, f"{MODEL} answered differently from the recorded demo answer (Ollama version or model changed?)"
