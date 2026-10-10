"""Corpus builder, extraction: the answer schema, the prompt, the cache, long documents, and the two
model clients (Gemini mocked, Ollama with a fake transport). No test touches the network."""

import enum
import hashlib
import json
import sys
import types
import urllib.error

import pytest
from pydantic import ValidationError

from corpus_builder import extract
from corpus_builder.extract import (
    PROMPT_SHA,
    SYSTEM_TEMPLATE,
    USER_TEMPLATE,
    ExtractionFailed,
    GeminiLLM,
    GeminiSetupError,
    OllamaLLM,
    answer_schema,
    cache_key,
    extract_all,
    schema_sha,
    split_windows,
    system_prompt,
    templates_sha,
    user_prompt,
)
from corpus_builder.store import BuildStore
from fixtures.corpus_b.synthetic import VIEWS, FakeLLM, facet, failing_llm, fixture_text, make_doc
from ratio.config import default_config

TAXONOMY = default_config().fact_patterns
SECRET = "AIzaSy-test-secret-key-0123456789abcdef"
FACT = "She was held in police custody for five days before she was first brought before a judge"
ANSWER = {"facets": [facet("gc35_48h", [FACT])]}


@pytest.fixture
def store(tmp_path):
    return BuildStore(tmp_path / "build.db")


@pytest.fixture
def views(store):
    doc = make_doc("ccpr-9999-2099", fixture_text(VIEWS))
    store.put_document(doc, doc.url)
    return doc


def key_for(doc, model="fake-model"):
    return cache_key(text_sha256=doc.text_sha256, prompt_sha=PROMPT_SHA, schema_sha=schema_sha(TAXONOMY), model=model)


# --- the answer schema -----------------------------------------------------------------------


def test_answer_schema_restricts_facet_ids_to_the_taxonomy():
    schema = answer_schema(TAXONOMY)
    enum_ids = schema.model_json_schema()["$defs"]["ExtractedFacet"]["properties"]["facet_id"]["enum"]
    assert enum_ids == [f.id for f in TAXONOMY.facets]
    assert schema.model_validate(ANSWER).facets[0].facet_id == "gc35_48h"
    with pytest.raises(ValidationError):
        schema.model_validate({"facets": [facet("judge_bias", [FACT])]})


@pytest.mark.parametrize("count", [0, 4])
def test_answer_schema_wants_one_to_three_facts(count):
    with pytest.raises(ValidationError):
        answer_schema(TAXONOMY).model_validate({"facets": [facet("gc35_48h", [FACT] * count)]})


def test_answer_schema_is_accepted_by_the_gemini_sdk():
    pytest.importorskip("google.genai")
    from google.genai import _transformers

    converted = _transformers.t_schema(None, answer_schema(TAXONOMY))  # offline: only builds the request schema
    facet_schema = converted.properties["facets"].items
    assert facet_schema.properties["facet_id"].enum == [f.id for f in TAXONOMY.facets]


# --- the prompt ------------------------------------------------------------------------------


def test_prompt_lists_every_facet_and_the_quoting_rules():
    system = system_prompt(TAXONOMY)
    for pattern in TAXONOMY.facets:
        assert f"{pattern.id}: {pattern.label}" in system
        assert pattern.description in system
    for rule in ("exactly", "character for character", "8 to 60 words", "contiguous", "person concerned"):
        assert rule in system
    for kind in ("violation_found", "no_violation", "not_examined", "monitor_assessment"):
        assert kind in system
    assert "never infer" in system.lower()
    assert "judges" in system


def test_prompt_asks_for_attributed_facts_with_who_states_them():
    system = system_prompt(TAXONOMY)
    assert "together with who states them" in system
    assert "'The author claims that ...'" in system and "'According to the State party ...'" in system
    assert "never quote an allegation or a party's claim as if it were an established fact" in system.lower()


def test_prompt_sha_covers_both_templates():
    assert PROMPT_SHA == templates_sha(SYSTEM_TEMPLATE, USER_TEMPLATE)
    assert len(PROMPT_SHA) == 64
    assert templates_sha(SYSTEM_TEMPLATE + " ", USER_TEMPLATE) != PROMPT_SHA
    assert templates_sha(SYSTEM_TEMPLATE, USER_TEMPLATE + " ") != PROMPT_SHA


def test_user_prompt_holds_only_the_document_text():
    assert user_prompt("Some public text.", 1, 1) == USER_TEMPLATE.format(part="", text="Some public text.")
    assert "(part 2 of 3)" in user_prompt("Some public text.", 2, 3)


# --- the cache -------------------------------------------------------------------------------


def test_cache_key_changes_with_each_part():
    parts = {"text_sha256": "a" * 64, "prompt_sha": "b" * 64, "schema_sha": "c" * 64, "model": "m"}
    base = cache_key(**parts)
    assert base == cache_key(**parts)
    assert base == hashlib.sha256("|".join(parts.values()).encode()).hexdigest()
    for name in parts:
        assert cache_key(**{**parts, name: parts[name] + "x"}) != base


def test_schema_sha_follows_the_taxonomy():
    changed = TAXONOMY.model_copy(update={"facets": TAXONOMY.facets[:-1]})
    assert schema_sha(TAXONOMY) == schema_sha(TAXONOMY)
    assert schema_sha(changed) != schema_sha(TAXONOMY)


def test_extract_all_stores_the_raw_answer_then_hits_the_cache(store, views):
    llm = FakeLLM(answer=ANSWER)
    first = extract_all(store, TAXONOMY, llm)
    assert first.extracted == (views.id,) and first.cached == () and first.failed == ()
    stored = json.loads(store.get_extraction(key_for(views)))
    assert stored["facets"][0]["facet_id"] == "gc35_48h"
    assert stored["facets"][0]["facts"] == [{"quote": FACT}]

    second = extract_all(store, TAXONOMY, llm)
    assert second.cached == (views.id,) and second.extracted == ()
    assert len(llm.calls) == 1  # the cache hit never calls the model


def test_another_model_is_a_cache_miss(store, views):
    extract_all(store, TAXONOMY, FakeLLM(answer=ANSWER))
    other = FakeLLM(answer=ANSWER, model="other-model")
    assert extract_all(store, TAXONOMY, other).extracted == (views.id,)
    assert store.get_extraction(key_for(views, "other-model")) is not None


def test_only_limits_the_documents(store, views):
    other = make_doc("ccpr-1-2099", fixture_text(VIEWS) + "\nAnother invented paragraph.")
    store.put_document(other, other.url)
    report = extract_all(store, TAXONOMY, FakeLLM(answer=ANSWER), only=other.id)
    assert report.extracted == (other.id,)
    assert store.get_extraction(key_for(views)) is None


def test_the_model_sees_only_the_public_text_and_the_fixed_instructions(store, views):
    llm = FakeLLM(answer=ANSWER)
    extract_all(store, TAXONOMY, llm)
    (call,) = llm.calls
    assert call.system == system_prompt(TAXONOMY)
    assert call.user == USER_TEMPLATE.format(part="", text=views.text)
    assert views.url not in call.user and views.title not in call.user
    assert call.schema.model_json_schema() == answer_schema(TAXONOMY).model_json_schema()


def test_a_failed_extraction_is_recorded_and_skipped(store, views):
    report = extract_all(store, TAXONOMY, failing_llm("the answer stopped early (MAX_TOKENS)"))
    assert report.failed == ((views.id, "the answer stopped early (MAX_TOKENS)"),)
    assert report.extracted == ()
    assert store.get_extraction(key_for(views)) is None
    assert extract_all(store, TAXONOMY, FakeLLM(answer=ANSWER)).extracted == (views.id,)  # retried next run


# --- long documents --------------------------------------------------------------------------


def test_split_windows_cuts_on_paragraph_boundaries():
    text = "\n\n".join(f"Paragraph {n} " + "word " * 10 for n in range(12))
    windows = split_windows(text, 200)
    assert "".join(windows) == text
    assert len(windows) > 1
    assert all(len(window) <= 200 for window in windows)
    assert all(window.startswith("\n\n") for window in windows[1:])
    assert split_windows("short", 200) == ["short"]


def test_split_windows_falls_back_to_spaces_then_hard_cuts():
    assert "".join(split_windows("word " * 100, 64)) == "word " * 100
    assert split_windows("x" * 10, 4) == ["xxxx", "xxxx", "xx"]


def test_long_documents_are_extracted_by_window_and_merged(store):
    first, second = "The first invented paragraph about an arrest.", "The second invented paragraph about a hearing."
    doc = make_doc("tw-long", "SYNTHETIC: long.\n\n" + first + "\n\n" + second)
    store.put_document(doc, doc.url)

    def answer(user):
        if second in user:
            return {"facets": [facet("gc35_48h", [second, first], finding=second)]}
        if first in user:
            return {"facets": [facet("gc35_48h", [first]), facet("iccpr_14_3_b", [first], "not_examined")]}
        return {"facets": []}

    llm = FakeLLM(answer=answer, window_chars=60)
    assert extract_all(store, TAXONOMY, llm).extracted == (doc.id,)
    assert [call.user.count("<document>") for call in llm.calls] == [1, 1, 1]
    assert "(part 3 of 3)" in llm.calls[-1].user
    merged = json.loads(store.get_extraction(key_for(doc)))["facets"]
    assert [f["facet_id"] for f in merged] == ["gc35_48h", "iccpr_14_3_b"]
    assert merged[0]["facts"] == [{"quote": first}, {"quote": second}]
    assert merged[0]["finding"] == {"quote": second}


# --- Gemini (google.genai mocked) ------------------------------------------------------------


class FakeModels:
    def __init__(self):
        self.response = None
        self.error = None
        self.listing = ()
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return self.response

    def list(self):
        if self.error is not None:
            raise self.error
        return iter(self.listing)


class FakeClient:
    instances: list = []

    def __init__(self, *, api_key, http_options=None):
        self.api_key = api_key
        self.http_options = http_options
        self.models = FakeModels()
        FakeClient.instances.append(self)


@pytest.fixture
def genai(monkeypatch, tmp_path):
    module = types.ModuleType("google.genai")
    module.Client = FakeClient
    FakeClient.instances = []
    monkeypatch.setitem(sys.modules, "google.genai", module)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return module


def response(text, finish="STOP", block=None):
    candidate = types.SimpleNamespace(finish_reason=finish)
    feedback = types.SimpleNamespace(block_reason=block) if block else None
    return types.SimpleNamespace(text=text, candidates=[candidate], prompt_feedback=feedback)


def gemini(tmp_path, monkeypatch, key=SECRET):
    monkeypatch.setenv("GEMINI_API_KEY", key)
    llm = GeminiLLM("gemini-test", env_file=tmp_path / "missing.env")
    return llm, FakeClient.instances[-1]


def test_gemini_reads_the_key_from_the_environment(genai, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n")
    monkeypatch.setenv("GEMINI_API_KEY", "from-environment")
    GeminiLLM("gemini-test", env_file=tmp_path / ".env")
    assert FakeClient.instances[-1].api_key == "from-environment"


def test_gemini_reads_the_key_from_the_env_file(genai, tmp_path):
    (tmp_path / ".env").write_text("# a comment\nOTHER=1\nexport GEMINI_API_KEY = \"from-dotenv\"\n")
    llm = GeminiLLM("gemini-test", env_file=tmp_path / ".env")
    assert FakeClient.instances[-1].api_key == "from-dotenv"
    assert llm.model == "gemini-test"


def test_gemini_takes_an_explicit_key(genai, tmp_path):
    GeminiLLM("gemini-test", api_key="explicit", env_file=tmp_path / "missing.env")
    assert FakeClient.instances[-1].api_key == "explicit"


def test_gemini_without_a_key_says_where_to_put_it(genai, tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=\n")
    with pytest.raises(GeminiSetupError, match="GEMINI_API_KEY"):
        GeminiLLM("gemini-test", env_file=tmp_path / ".env")


def test_gemini_without_the_sdk_says_how_to_install_it(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.genai", None)  # makes the import fail
    with pytest.raises(GeminiSetupError, match=r"\[corpus\]"):
        GeminiLLM("gemini-test", api_key="explicit", env_file=tmp_path / "missing.env")


def test_gemini_sends_the_structured_request(genai, tmp_path, monkeypatch):
    llm, client = gemini(tmp_path, monkeypatch)
    schema = answer_schema(TAXONOMY)
    client.models.response = response(json.dumps(ANSWER))
    result = llm.complete_json(system="SYS", user="USER", schema=schema, purpose="precedent_facets")
    assert result.facets[0].facet_id == "gc35_48h"
    (call,) = client.models.calls
    assert call["model"] == "gemini-test" and call["contents"] == "USER"
    assert call["config"] == {
        "system_instruction": "SYS",
        "response_mime_type": "application/json",
        "response_schema": schema,
        "temperature": 0, "thinking_config": {"thinking_level": "LOW"}}


def test_gemini_accepts_the_sdk_enum_for_stop(genai, tmp_path, monkeypatch):
    finish = enum.Enum("FinishReason", {"STOP": "STOP"})
    llm, client = gemini(tmp_path, monkeypatch)
    client.models.response = response(json.dumps(ANSWER), finish=finish.STOP)
    assert llm.complete_json(system="S", user="U", schema=answer_schema(TAXONOMY), purpose="p").facets


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (response('{"facets": [', finish="MAX_TOKENS"), "MAX_TOKENS"),
        (response("", finish="SAFETY"), "SAFETY"),
        (response("", block="PROHIBITED_CONTENT"), "PROHIBITED_CONTENT"),
        (types.SimpleNamespace(text=None, candidates=[], prompt_feedback=None), "no answer"),
        (response("I cannot help with that."), "schema"),
    ],
)
def test_gemini_refusals_and_truncation_fail_the_extraction(genai, tmp_path, monkeypatch, reply, reason):
    llm, client = gemini(tmp_path, monkeypatch)
    client.models.response = reply
    with pytest.raises(ExtractionFailed, match=reason):
        llm.complete_json(system="S", user="U", schema=answer_schema(TAXONOMY), purpose="p")


def test_gemini_never_puts_the_key_in_an_error(genai, tmp_path, monkeypatch):
    llm, client = gemini(tmp_path, monkeypatch)
    client.models.error = RuntimeError(f"401 for https://generativelanguage.googleapis.com/?key={SECRET}")
    with pytest.raises(ExtractionFailed) as caught:
        llm.complete_json(system="S", user="U", schema=answer_schema(TAXONOMY), purpose="p")
    error = caught.value
    assert SECRET not in str(error) and SECRET not in repr(error)
    assert error.__cause__ is None and error.__suppress_context__
    with pytest.raises(GeminiSetupError) as listed:
        llm.list_models()
    assert SECRET not in str(listed.value)


def test_gemini_lists_the_models_that_generate_content(genai, tmp_path, monkeypatch):
    llm, client = gemini(tmp_path, monkeypatch)
    client.models.listing = (
        types.SimpleNamespace(name="models/gemini-b", supported_actions=["generateContent"]),
        types.SimpleNamespace(name="models/embedder", supported_actions=["embedContent"]),
        types.SimpleNamespace(name="models/gemini-a", supported_actions=["countTokens", "generateContent"]),
    )
    assert llm.list_models() == ("gemini-a", "gemini-b")


# --- Ollama (fake transport) -----------------------------------------------------------------


class FakePost:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def __call__(self, url, body, timeout):
        self.requests.append((url, body, timeout))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_ollama_posts_a_deterministic_schema_request_to_loopback():
    post = FakePost({"message": {"content": json.dumps(ANSWER)}, "done_reason": "stop"})
    llm = OllamaLLM(post=post)
    schema = answer_schema(TAXONOMY)
    assert llm.complete_json(system="S", user="U", schema=schema, purpose="p").facets[0].facet_id == "gc35_48h"
    (show, (url, body, _timeout)) = post.requests  # first the server says the model is local, then the question
    assert show[:2] == ("http://127.0.0.1:11434/api/show", {"model": "qwen2.5:7b-instruct"})
    assert url == "http://127.0.0.1:11434/api/chat"
    assert body["model"] == "qwen2.5:7b-instruct" == llm.model
    assert body["format"] == schema.model_json_schema()
    assert body["stream"] is False and body["options"]["temperature"] == 0
    assert body["messages"] == [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]


@pytest.mark.parametrize(
    "reply",
    [
        {"message": {"content": '{"facets": ['}, "done_reason": "length"},
        urllib.error.URLError("connection refused"),
        {"message": {"content": "not json"}, "done_reason": "stop"},
    ],
)
def test_ollama_failures_fail_the_extraction(reply):
    with pytest.raises(ExtractionFailed):
        OllamaLLM(post=FakePost(reply)).complete_json(system="S", user="U", schema=answer_schema(TAXONOMY), purpose="p")


def test_google_genai_is_imported_only_when_gemini_is_used():
    assert "google.genai" not in vars(extract)
    assert not any(name.startswith("google") for name in vars(extract))


class _Clock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(round(seconds, 1))
        self.now += seconds


QUOTA = "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: free_tier_requests, limit: 5. Please retry in 56.6s."


def test_gemini_waits_out_a_per_minute_quota_and_retries(genai, tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    clock = _Clock()
    llm = GeminiLLM("gemini-test", env_file=tmp_path / "missing.env", clock=clock, sleep=clock.sleep)
    client = FakeClient.instances[-1]
    answers = [RuntimeError(QUOTA), response(json.dumps(ANSWER))]

    def generate(**call):
        client.models.calls.append(call)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    client.models.generate_content = generate
    assert llm.complete_json(system="S", user="U", schema=answer_schema(TAXONOMY), purpose="p").facets
    assert len(client.models.calls) == 2 and 57.6 in clock.slept


def test_gemini_spaces_its_requests_and_gives_up_on_a_daily_limit(genai, tmp_path, monkeypatch):
    from corpus_builder.extract import GEMINI_MIN_INTERVAL_S, quota_wait

    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    clock = _Clock()
    llm = GeminiLLM("gemini-test", env_file=tmp_path / "missing.env", clock=clock, sleep=clock.sleep)
    client = FakeClient.instances[-1]
    client.models.response = response(json.dumps(ANSWER))
    for _ in range(2):
        llm.complete_json(system="S", user="U", schema=answer_schema(TAXONOMY), purpose="p")
    assert clock.slept == [GEMINI_MIN_INTERVAL_S]
    assert quota_wait("429 RESOURCE_EXHAUSTED ... Please retry in 3600s.") is None  # a daily limit
    assert quota_wait("503 UNAVAILABLE high demand") == 20.0
    assert quota_wait("429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier retry in 30s") is None
    assert quota_wait("400 INVALID_ARGUMENT") is None
