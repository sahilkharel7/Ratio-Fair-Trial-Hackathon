"""Local LLM access through Ollama, with a response cache and a replay-only mode.

Hard rule 1: the only network peer is the local Ollama server. The client refuses non-loopback
hosts and cloud models and bypasses any HTTP proxy. Requests are deterministic (temperature 0,
fixed seed, one fixed context size) and every reply is validated against a pydantic schema.
Truncated or invalid replies raise errors and are never cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from ratio.config import LLMSettings
from ratio.netguard import is_loopback_host

T = TypeVar("T", bound=BaseModel)
Transport = Callable[[str, dict | None, float], dict]

_CHARS_PER_TOKEN = 3  # conservative estimate, so prompts are refused before Ollama would truncate them
_PROXYLESS = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LLMError(RuntimeError):
    """The local model could not produce a usable answer."""


class OllamaUnavailable(LLMError):
    """The Ollama server is not running or not reachable on loopback."""


class ModelNotAvailable(LLMError):
    """The configured model has not been pulled."""


class LLMResponseError(LLMError):
    """The model's reply was truncated or did not match the schema."""


class CacheMiss(LLMError):
    """Replay-only mode found no cached reply for this request."""


@dataclass(frozen=True)
class HttpError(Exception):
    status: int
    message: str


def model_from_env(settings: LLMSettings) -> str:
    return os.environ.get("RATIO_MODEL") or settings.default_model


def request_options(settings: LLMSettings, purpose: str) -> dict:
    budgets = settings.num_predict
    num_predict = {"extraction": budgets.extraction, "labels": budgets.labels, "argument_check": budgets.argument_check}
    if purpose not in num_predict:
        raise ValueError(f"unknown LLM purpose {purpose!r}")
    return {
        "temperature": settings.temperature,
        "seed": settings.seed,
        "num_ctx": settings.num_ctx,
        "num_predict": num_predict[purpose],
    }


def _require_loopback(host_url: str) -> None:
    parts = urlsplit(host_url)
    if parts.scheme != "http" or not is_loopback_host(parts.hostname):
        raise LLMError(f"Ollama host {host_url!r} is not a loopback address; Ratio only talks to a local server")


def _refuse_cloud_model(model: str) -> None:
    if "cloud" in model.lower():
        raise LLMError(f"model {model!r} looks like an Ollama cloud model; only local models are allowed")


def urllib_transport(url: str, body: dict | None, timeout: float) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST"
    )
    try:
        with _PROXYLESS.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except (ValueError, AttributeError):
            pass
        raise HttpError(exc.code, str(detail)) from exc
    except TimeoutError as exc:
        raise LLMError(f"Ollama did not answer within {timeout:.0f} s") from exc
    except (urllib.error.URLError, ConnectionError) as exc:
        raise OllamaUnavailable(
            f"Ollama is not reachable at {url}. Start it with `ollama serve` or `brew services start ollama`."
        ) from exc


def _short(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()[:5])


class OllamaClient:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        model: str | None = None,
        host: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self._settings = settings
        self._model = model or model_from_env(settings)
        self._host = (host or os.environ.get("RATIO_OLLAMA_HOST") or settings.host).rstrip("/")
        _require_loopback(self._host)
        _refuse_cloud_model(self._model)
        self._transport = transport or urllib_transport

    @property
    def model(self) -> str:
        return self._model

    def _post(self, path: str, body: dict | None) -> dict:
        try:
            return self._transport(f"{self._host}{path}", body, self._settings.timeout_seconds)
        except HttpError as exc:
            if exc.status == 404:
                raise ModelNotAvailable(
                    f"Model {self._model!r} is not available. Run `ollama pull {self._model}` once while online."
                ) from exc
            raise LLMError(f"Ollama returned HTTP {exc.status}: {exc.message}") from exc

    def version(self) -> str:
        return str(self._post("/api/version", None).get("version", "unknown"))

    def check_model(self) -> dict:
        """Confirm the model is pulled and local; returns Ollama's description of it."""
        info = self._post("/api/show", {"model": self._model})
        if info.get("remote_host") or info.get("remote_model"):
            raise LLMError(f"model {self._model!r} is served by an Ollama cloud host; only local models are allowed")
        return info

    def _check_prompt_size(self, messages: Sequence[dict], num_predict: int) -> None:
        estimate = sum(len(m["content"]) for m in messages) // _CHARS_PER_TOKEN
        if estimate + num_predict > self._settings.num_ctx:
            raise LLMError(
                f"prompt too long for the {self._settings.num_ctx}-token context window (about {estimate} tokens); "
                "reduce the chunk size"
            )

    def _chat(self, messages: list[dict], schema_json: dict, options: dict) -> str:
        self._check_prompt_size(messages, options["num_predict"])
        body = {
            "model": self._model,
            "messages": messages,
            "format": schema_json,
            "stream": False,
            "options": options,
            "keep_alive": self._settings.keep_alive,
        }
        response = self._post("/api/chat", body)
        if response.get("error"):
            raise LLMError(f"Ollama error: {response['error']}")
        if response.get("done_reason") != "stop":
            raise LLMResponseError(f"model output was truncated (done_reason={response.get('done_reason')!r})")
        content = (response.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("model returned an empty reply")
        return content

    def complete_json(self, *, system: str, user: str, schema: type[T], purpose: str) -> T:
        options = request_options(self._settings, purpose)
        schema_json = schema.model_json_schema()
        system_text = f"{system}\n\nReply with JSON only, matching this JSON schema:\n{json.dumps(schema_json)}"
        messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user}]
        reply = self._chat(messages, schema_json, options)
        try:
            return schema.model_validate_json(reply)
        except ValidationError as first_error:
            correction = (
                f"Your previous reply did not match the schema ({_short(first_error)}). "
                "Reply again with valid JSON only."
            )
            retry = messages + [{"role": "assistant", "content": reply}, {"role": "user", "content": correction}]
            reply = self._chat(retry, schema_json, options)
            try:
                return schema.model_validate_json(reply)
            except ValidationError as second_error:
                raise LLMResponseError(
                    f"model reply did not match {schema.__name__} after a retry: {_short(second_error)}"
                ) from second_error


class ResponseCache:
    """One JSON file per request key. Files hold the validated reply, never the prompt text."""

    def __init__(self, write_dir: Path, read_dirs: Sequence[Path] = ()) -> None:
        self._write_dir = Path(write_dir)
        self._dirs = (self._write_dir, *(Path(d) for d in read_dirs))

    @staticmethod
    def key(*, model: str, purpose: str, system: str, user: str, schema_json: dict, options: dict) -> str:
        canonical = json.dumps(
            {"model": model, "purpose": purpose, "system": system, "user": user, "schema": schema_json, "options": options},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        for directory in self._dirs:
            path = directory / f"{key}.json"
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        return None

    def put(self, key: str, record: dict) -> None:
        self._write_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self._write_dir, delete=False, suffix=".tmp") as tmp:
            tmp.write(text)
        Path(tmp.name).replace(self._write_dir / f"{key}.json")


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int


class CachedLLM:
    """Wraps a client with a response cache. With ``client=None`` it replays only (no Ollama needed)."""

    def __init__(
        self,
        client: OllamaClient | None,
        cache: ResponseCache,
        *,
        settings: LLMSettings,
        model: str | None = None,
    ) -> None:
        if client is None and model is None:
            raise ValueError("replay-only mode needs the model name the cache was built with")
        self._client = client
        self._cache = cache
        self._settings = settings
        self._model = client.model if client is not None else str(model)
        self._hits = 0
        self._misses = 0
        self._used: set[str] = set()

    @property
    def model(self) -> str:
        return self._model

    @property
    def used_keys(self) -> frozenset[str]:
        """Cache keys consulted so far (lets the demo cache be pruned to exactly what it replays)."""
        return frozenset(self._used)

    @property
    def replay_only(self) -> bool:
        return self._client is None

    @property
    def stats(self) -> CacheStats:
        return CacheStats(hits=self._hits, misses=self._misses)

    def complete_json(self, *, system: str, user: str, schema: type[T], purpose: str) -> T:
        options = request_options(self._settings, purpose)
        key = ResponseCache.key(
            model=self._model, purpose=purpose, system=system, user=user, schema_json=schema.model_json_schema(), options=options
        )
        self._used.add(key)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return schema.model_validate(cached["response"])
        self._misses += 1
        if self._client is None:
            raise CacheMiss(f"no cached {purpose} reply for this request (replay-only mode, model {self._model})")
        result = self._client.complete_json(system=system, user=user, schema=schema, purpose=purpose)
        self._cache.put(
            key,
            {"key": key, "model": self._model, "purpose": purpose, "schema": schema.__name__, "response": result.model_dump(mode="json")},
        )
        return result
