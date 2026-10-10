"""Groq uses bounded requests, schema checks and redacted failure messages."""

import json

import httpx
import pytest

from corpus_builder.extract import ExtractionFailed
from corpus_builder.groq import GroqLLM, GroqSetupError
from ratio.case_briefs import BriefDraft


def answer():
    return {
        "sections": [
            {
                "label": "Charge",
                "summary": "Synthetic example only.",
                "quote": "The charge is a synthetic example.",
                "pinpoint": "Line 1",
            }
        ]
    }


def response(payload=None, *, finish="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {"content": json.dumps(payload or answer())},
                }
            ]
        },
    )


def test_structured_reply_uses_fixed_endpoint_and_provider_cache_identity(tmp_path):
    seen = []

    def transport(request):
        seen.append(request)
        return response()

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        llm = GroqLLM(
            api_key="synthetic-key", env_file=tmp_path / "empty", client=client
        )
        draft = llm.complete_json(
            system="JSON only",
            user="Synthetic source.",
            schema=BriefDraft,
            purpose="fixture",
        )
    assert draft.sections[0].label == "Charge"
    assert llm.model == "groq:openai/gpt-oss-120b"
    assert str(seen[0].url) == "https://api.groq.com/openai/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer synthetic-key"
    body = json.loads(seen[0].content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["max_completion_tokens"] == 2000


@pytest.mark.parametrize("status", [401, 403, 500])
def test_service_errors_never_echo_key_or_response_body(status, tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                status, json={"error": "synthetic-secret and source contents"}
            )
        )
    ) as client:
        llm = GroqLLM(
            api_key="synthetic-secret", env_file=tmp_path / "empty", client=client
        )
        with pytest.raises(ExtractionFailed) as caught:
            llm.complete_json(
                system="", user="fixture", schema=BriefDraft, purpose="fixture"
            )
    assert "synthetic-secret" not in str(caught.value)
    assert "source contents" not in str(caught.value)
    assert str(status) in str(caught.value)


def test_rate_limit_retry_is_bounded(tmp_path):
    replies = iter([httpx.Response(429, headers={"retry-after": "2"}), response()])
    waits = []
    with httpx.Client(transport=httpx.MockTransport(lambda _: next(replies))) as client:
        llm = GroqLLM(
            api_key="fixture",
            env_file=tmp_path / "empty",
            client=client,
            sleep=waits.append,
        )
        assert llm.complete_json(
            system="", user="fixture", schema=BriefDraft, purpose="fixture"
        )
    assert waits == [3]


def test_incomplete_and_oversized_requests_are_refused(tmp_path):
    calls = []
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: calls.append(r) or response(finish="length")
        )
    ) as client:
        llm = GroqLLM(api_key="fixture", env_file=tmp_path / "empty", client=client)
        with pytest.raises(ExtractionFailed, match="incomplete"):
            llm.complete_json(
                system="", user="fixture", schema=BriefDraft, purpose="fixture"
            )
        with pytest.raises(ExtractionFailed, match="input bound"):
            llm.complete_json(
                system="", user="x" * 20_000, schema=BriefDraft, purpose="fixture"
            )
    assert len(calls) == 1


def test_missing_key_and_non_demo_model_fail_without_a_request(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    with pytest.raises(GroqSetupError, match="GROQ_API_KEY"):
        GroqLLM(env_file=tmp_path / "absent")
    with pytest.raises(GroqSetupError, match="supported demo model"):
        GroqLLM("arbitrary-model", api_key="fixture", env_file=tmp_path / "absent")
