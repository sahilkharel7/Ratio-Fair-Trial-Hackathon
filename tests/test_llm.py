"""LLM client: loopback only, deterministic structured requests, strict validation, cache, replay."""

import json

import pytest
from pydantic import BaseModel

from ratio.config import load_config
from ratio.llm import (
    CachedLLM,
    CacheMiss,
    HttpError,
    LLMError,
    LLMResponseError,
    ModelNotAvailable,
    OllamaClient,
    OllamaUnavailable,
    ResponseCache,
)

SETTINGS = load_config().settings.llm


class Label(BaseModel):
    label: str
    quote: str


class FakeTransport:
    """Records each request and returns scripted replies (a dict, or an exception to raise)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, url, body, timeout):
        self.requests.append((url, body, timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def chat_reply(content: str, done_reason: str = "stop") -> dict:
    return {"message": {"role": "assistant", "content": content}, "done": True, "done_reason": done_reason}


def client_with(*replies, model="qwen2.5:7b-instruct"):
    transport = FakeTransport(replies)
    return OllamaClient(SETTINGS, model=model, transport=transport), transport


def test_request_is_deterministic_structured_and_not_streamed():
    client, transport = client_with(chat_reply('{"label": "supports", "quote": "x"}'))
    assert client.complete_json(system="sys", user="usr", schema=Label, purpose="labels") == Label(label="supports", quote="x")
    url, body, timeout = transport.requests[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert body["stream"] is False
    assert body["format"] == Label.model_json_schema()
    assert body["options"] == {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": SETTINGS.num_predict.labels}
    assert body["keep_alive"] == "30m"
    assert '"label"' in body["messages"][0]["content"]  # the schema is also given in the prompt
    assert timeout == 90


def test_invalid_reply_is_retried_once_with_the_error_appended():
    client, transport = client_with(chat_reply('{"label": 1}'), chat_reply('{"label": "supports", "quote": "y"}'))
    assert client.complete_json(system="s", user="u", schema=Label, purpose="labels").quote == "y"
    retry = transport.requests[1][1]["messages"]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert "did not match" in retry[-1]["content"]


def test_two_invalid_replies_raise():
    client, _ = client_with(chat_reply("not json"), chat_reply("still not json"))
    with pytest.raises(LLMResponseError, match="Label"):
        client.complete_json(system="s", user="u", schema=Label, purpose="labels")


def test_truncated_reply_is_an_error():
    client, _ = client_with(chat_reply('{"label": "sup', done_reason="length"))
    with pytest.raises(LLMResponseError, match="truncated"):
        client.complete_json(system="s", user="u", schema=Label, purpose="labels")


def test_prompt_too_long_for_the_context_window_is_refused_before_sending():
    client, transport = client_with()
    with pytest.raises(LLMError, match="too long"):
        client.complete_json(system="s", user="x" * 40000, schema=Label, purpose="labels")
    assert transport.requests == []


def test_non_loopback_host_is_refused():
    with pytest.raises(LLMError, match="loopback"):
        OllamaClient(SETTINGS, model="m", host="http://ollama.example.com:11434", transport=FakeTransport([]))


@pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "kimi-k3:cloud"])
def test_cloud_models_are_refused(model):
    with pytest.raises(LLMError, match="cloud"):
        OllamaClient(SETTINGS, model=model, transport=FakeTransport([]))


def test_model_from_environment_variable(monkeypatch):
    monkeypatch.setenv("RATIO_MODEL", "llama3.1:8b")
    assert OllamaClient(SETTINGS, transport=FakeTransport([])).model == "llama3.1:8b"


def test_missing_model_gives_pull_instructions():
    client, _ = client_with(HttpError(404, "model 'm' not found"), model="m")
    with pytest.raises(ModelNotAvailable, match="ollama pull m"):
        client.complete_json(system="s", user="u", schema=Label, purpose="labels")


def test_server_down_is_reported_clearly():
    client, _ = client_with(OllamaUnavailable("Ollama is not reachable; start it with `ollama serve`"))
    with pytest.raises(OllamaUnavailable, match="ollama serve"):
        client.complete_json(system="s", user="u", schema=Label, purpose="labels")


def test_check_model_rejects_remote_models():
    client, _ = client_with({"details": {}, "remote_host": "https://ollama.com:443"})
    with pytest.raises(LLMError, match="cloud"):
        client.check_model()


class TestCache:
    def test_hit_avoids_a_second_call(self, tmp_path):
        client, transport = client_with(chat_reply('{"label": "a", "quote": "q"}'))
        llm = CachedLLM(client, ResponseCache(tmp_path), settings=SETTINGS)
        first = llm.complete_json(system="s", user="u", schema=Label, purpose="labels")
        second = llm.complete_json(system="s", user="u", schema=Label, purpose="labels")
        assert first == second
        assert len(transport.requests) == 1
        assert (llm.stats.hits, llm.stats.misses) == (1, 1)

    def test_replay_only_raises_on_a_miss_without_calling_anything(self, tmp_path):
        llm = CachedLLM(None, ResponseCache(tmp_path), settings=SETTINGS, model="m")
        with pytest.raises(CacheMiss):
            llm.complete_json(system="s", user="u", schema=Label, purpose="labels")

    def test_replay_reads_a_cache_written_live(self, tmp_path):
        client, _ = client_with(chat_reply('{"label": "a", "quote": "q"}'), model="m")
        CachedLLM(client, ResponseCache(tmp_path / "demo"), settings=SETTINGS).complete_json(
            system="s", user="u", schema=Label, purpose="labels"
        )
        replay = CachedLLM(None, ResponseCache(tmp_path / "runtime", read_dirs=[tmp_path / "demo"]), settings=SETTINGS, model="m")
        assert replay.complete_json(system="s", user="u", schema=Label, purpose="labels").label == "a"

    def test_failures_are_never_cached(self, tmp_path):
        client, _ = client_with(chat_reply("{", done_reason="length"))
        llm = CachedLLM(client, ResponseCache(tmp_path), settings=SETTINGS)
        with pytest.raises(LLMResponseError):
            llm.complete_json(system="s", user="u", schema=Label, purpose="labels")
        assert list(tmp_path.glob("*.json")) == []

    def test_key_depends_on_model_prompt_schema_and_purpose(self):
        base = dict(model="m", purpose="labels", system="s", user="u", schema_json={"a": 1}, options={"seed": 1})
        key = ResponseCache.key(**base)
        assert key == ResponseCache.key(**base)
        for change in ({"model": "n"}, {"user": "v"}, {"system": "t"}, {"schema_json": {"a": 2}}, {"purpose": "extraction"}):
            assert ResponseCache.key(**{**base, **change}) != key

    def test_cache_files_hold_the_response_but_not_the_prompt(self, tmp_path):
        client, _ = client_with(chat_reply('{"label": "a", "quote": "q"}'))
        CachedLLM(client, ResponseCache(tmp_path), settings=SETTINGS).complete_json(
            system="SYSTEM PROMPT TEXT", user="CONFIDENTIAL NOTE TEXT", schema=Label, purpose="labels"
        )
        (path,) = tmp_path.glob("*.json")
        stored = path.read_text(encoding="utf-8")
        assert "CONFIDENTIAL NOTE TEXT" not in stored and "SYSTEM PROMPT TEXT" not in stored
        assert json.loads(stored)["response"] == {"label": "a", "quote": "q"}
