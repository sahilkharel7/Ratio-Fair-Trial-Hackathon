"""Explicit online Groq adapter for public-document maintainer and demo workers."""

from __future__ import annotations

import os
import time
from pathlib import Path

from corpus_builder.extract import ENV_FILE, ExtractionFailed, _dotenv_value, _parse

DEFAULT_MODEL = "openai/gpt-oss-120b"
MODELS = frozenset({DEFAULT_MODEL, "openai/gpt-oss-20b"})
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


class GroqSetupError(ValueError):
    """Missing local configuration, without exposing credential contents."""


class GroqLLM:
    window_chars = 9_000

    def __init__(
        self,
        model_id=None,
        api_key=None,
        *,
        env_file: Path = ENV_FILE,
        client=None,
        sleep=time.sleep,
    ):
        self.model_id = (
            model_id
            or os.environ.get("GROQ_MODEL")
            or _dotenv_value(env_file, "GROQ_MODEL")
            or DEFAULT_MODEL
        )
        if self.model_id not in MODELS:
            raise GroqSetupError(
                "Use the supported demo model openai/gpt-oss-120b or openai/gpt-oss-20b."
            )
        self._key = (
            api_key
            or os.environ.get("GROQ_API_KEY")
            or _dotenv_value(env_file, "GROQ_API_KEY")
        )
        if not self._key:
            raise GroqSetupError(
                "Set GROQ_API_KEY in the ignored local .env or environment."
            )
        self._client, self._sleep = client, sleep

    @property
    def model(self):
        return "groq:" + self.model_id

    def complete_json(self, *, system, user, schema, purpose):
        import httpx

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": False,
                    "schema": schema.model_json_schema(),
                },
            },
            "max_completion_tokens": 2_000,
            "temperature": 0,
            "reasoning_effort": "low",
        }
        # Keep individual requests below the documented free-tier token budget.
        import json

        if len(json.dumps(payload, ensure_ascii=False)) > 18_000:
            raise ExtractionFailed(
                f"{purpose}: request exceeds the demo input bound; use smaller source windows."
            )
        client = self._client or httpx.Client(timeout=90, follow_redirects=False)
        try:
            for attempt in range(2):
                try:
                    response = client.post(
                        ENDPOINT,
                        json=payload,
                        headers={
                            "Authorization": "Bearer " + self._key,
                            "User-Agent": "Ratio-Demo/1.0",
                            "Accept": "application/json",
                        },
                    )
                except httpx.HTTPError:
                    raise ExtractionFailed(
                        f"{purpose}: Groq could not be reached; retry when connected."
                    ) from None
                if response.status_code == 429:
                    try:
                        wait = float(response.headers.get("retry-after", "60"))
                    except ValueError:
                        wait = 60
                    if attempt == 0 and 0 <= wait <= 60:
                        self._sleep(min(60, wait + 1))
                        continue
                    raise ExtractionFailed(
                        f"{purpose}: Groq free-tier rate limit reached; retry later."
                    )
                if response.status_code != 200:
                    raise ExtractionFailed(
                        f"{purpose}: Groq returned HTTP {response.status_code}; check the key and model access."
                    )
                try:
                    choice = response.json()["choices"][0]
                    if choice["finish_reason"] != "stop" or choice["message"].get(
                        "refusal"
                    ):
                        raise ExtractionFailed(
                            f"{purpose}: model answer was refused or incomplete; no draft saved."
                        )
                    return _parse(schema, choice["message"]["content"], purpose)
                except (KeyError, IndexError, TypeError, ValueError):
                    raise ExtractionFailed(
                        f"{purpose}: Groq returned an unusable response; no draft saved."
                    ) from None
        finally:
            if self._client is None:
                client.close()
