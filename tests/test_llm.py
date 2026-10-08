"""LLM client: loopback only, deterministic structured requests, strict validation, cache, replay."""

import json

import pytest
from pydantic import BaseModel

from ratio.config import load_config
from ratio.llm import (
    CacheDamaged,
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
    """Records each request and returns scripted replies (a dict, or an exception to raise).
    /api/show (the "is this model local?" check) is answered by ``show`` and recorded apart."""

    def __init__(self, replies, show=None):
        self.replies = list(replies)
        self.requests = []
        self.shows = []
        self.show = {"details": {}} if show is None else show

    def __call__(self, url, body, timeout):
        if url.endswith("/api/show"):
            self.shows.append(body)
            if isinstance(self.show, Exception):
                raise self.show
            return self.show
        self.requests.append((url, body, timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def chat_reply(content: str, done_reason: str = "stop") -> dict:
    return {"message": {"role": "assistant", "content": content}, "done": True, "done_reason": done_reason}


def client_with(*replies, model="qwen2.5:7b-instruct", show=None):
    transport = FakeTransport(replies, show=show)
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
    assert timeout == 90 + SETTINGS.num_predict.labels / 10  # grows with the reply budget
    assert transport.shows == [{"model": "qwen2.5:7b-instruct"}]  # asked once whether the model is local


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
    client, _ = client_with(model="m", show=HttpError(404, "model 'm' not found"))
    with pytest.raises(ModelNotAvailable, match="ollama pull m"):
        client.complete_json(system="s", user="u", schema=Label, purpose="labels")


def test_a_404_that_is_not_a_missing_model_names_the_host_setting():
    client, _ = client_with(HttpError(404, "404 page not found"))
    with pytest.raises(LLMError, match="RATIO_OLLAMA_HOST") as raised:
        client.complete_json(system="s", user="u", schema=Label, purpose="labels")
    assert not isinstance(raised.value, ModelNotAvailable)


def test_a_local_alias_of_a_cloud_model_is_refused_before_any_prompt_is_sent():
    client, transport = client_with(model="legal-helper", show={"details": {}, "remote_host": "https://ollama.com:443"})
    with pytest.raises(LLMError, match="cloud"):
        client.complete_json(system="s", user="CONFIDENTIAL NOTE", schema=Label, purpose="labels")
    assert transport.requests == []


def test_timeout_can_be_set_by_environment(monkeypatch):
    monkeypatch.setenv("RATIO_OLLAMA_TIMEOUT", "600")
    client, transport = client_with(chat_reply('{"label": "a", "quote": "q"}'))
    client.complete_json(system="s", user="u", schema=Label, purpose="labels")
    assert transport.requests[0][2] == 600


def test_redirects_are_refused():
    from ratio.llm import _PROXYLESS, _NoRedirect

    assert any(isinstance(handler, _NoRedirect) for handler in _PROXYLESS.handlers)


def test_server_down_is_reported_clearly():
    client, _ = client_with(OllamaUnavailable("Ollama is not reachable; start it with `ollama serve`"))
    with pytest.raises(OllamaUnavailable, match="ollama serve"):
        client.complete_json(system="s", user="u", schema=Label, purpose="labels")


def test_check_model_rejects_remote_models():
    client, _ = client_with(show={"details": {}, "remote_host": "https://ollama.com:443"})
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

    def test_key_depends_on_model_prompt_schema_purpose_and_options(self):
        base = dict(model="m", purpose="labels", system="s", user="u", schema_json={"a": 1}, options={"seed": 1})
        key = ResponseCache.key(**base)
        assert key == ResponseCache.key(**base)
        changes = (
            {"model": "n"}, {"user": "v"}, {"system": "t"}, {"schema_json": {"a": 2}}, {"purpose": "extraction"},
            {"options": {"seed": 2}}, {"options": {"seed": 1, "num_ctx": 4096}},
        )  # fmt: skip
        for change in changes:
            assert ResponseCache.key(**{**base, **change}) != key

    def test_damaged_file_is_an_error_when_replaying_and_a_miss_when_live(self, tmp_path):
        settings_key = {"system": "s", "user": "u", "schema": Label, "purpose": "labels"}
        live, _ = client_with(chat_reply('{"label": "a", "quote": "q"}'), model="m")
        CachedLLM(live, ResponseCache(tmp_path), settings=SETTINGS).complete_json(**settings_key)
        (path,) = tmp_path.glob("*.json")
        path.write_text("{ not json", encoding="utf-8")
        with pytest.raises(CacheDamaged, match="cache"):
            CachedLLM(None, ResponseCache(tmp_path), settings=SETTINGS, model="m").complete_json(**settings_key)
        again, transport = client_with(chat_reply('{"label": "b", "quote": "q"}'), model="m")
        assert CachedLLM(again, ResponseCache(tmp_path), settings=SETTINGS).complete_json(**settings_key).label == "b"
        assert len(transport.requests) == 1

    def test_cache_files_hold_the_response_but_not_the_prompt(self, tmp_path):
        client, _ = client_with(chat_reply('{"label": "a", "quote": "q"}'))
        CachedLLM(client, ResponseCache(tmp_path), settings=SETTINGS).complete_json(
            system="SYSTEM PROMPT TEXT", user="CONFIDENTIAL NOTE TEXT", schema=Label, purpose="labels"
        )
        (path,) = tmp_path.glob("*.json")
        stored = path.read_text(encoding="utf-8")
        assert "CONFIDENTIAL NOTE TEXT" not in stored and "SYSTEM PROMPT TEXT" not in stored
        assert json.loads(stored)["response"] == {"label": "a", "quote": "q"}


def test_the_real_transport_reports_a_stopped_server_clearly():
    import socket

    from ratio.llm import urllib_transport

    with socket.socket() as probe:  # a loopback port with nothing listening
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(OllamaUnavailable, match="ollama serve"):
        urllib_transport(f"http://127.0.0.1:{port}/api/version", None, 2)


def test_the_real_transport_ignores_proxy_settings_and_maps_a_missing_model(monkeypatch):
    import http.server
    import threading

    class NotFound(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            self.rfile.read(int(self.headers.get("Content-Length", 0)))  # like a real server: read the request first
            body = b'{"error": "model \'m\' not found"}'
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), NotFound)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("HTTP_PROXY", "http://10.255.255.1:9")  # would hang if a proxy were used
    monkeypatch.setenv("http_proxy", "http://10.255.255.1:9")
    try:
        client = OllamaClient(SETTINGS, model="m", host=f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(ModelNotAvailable, match="ollama pull m"):
            client.check_model()
    finally:
        server.shutdown()


@pytest.mark.parametrize("failure", ["reset", "cut short"])
def test_an_error_body_that_cannot_be_read_still_maps_the_status(monkeypatch, failure):
    import http.client
    import io
    import urllib.error

    from ratio import llm

    class Broken(io.BytesIO):
        def read(self, *args):
            if failure == "reset":
                raise ConnectionResetError(54, "Connection reset by peer")
            raise http.client.IncompleteRead(b'{"error": "mod', 87)

    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, Broken())

    monkeypatch.setattr(llm._PROXYLESS, "open", opener)
    with pytest.raises(llm.HttpError) as caught:
        llm.urllib_transport("http://127.0.0.1:1/api/show", {"model": "m"}, 2)
    assert caught.value.status == 404
