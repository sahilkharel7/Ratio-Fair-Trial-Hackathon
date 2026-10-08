"""Local LLM access through Ollama, with a response cache and a replay-only mode.

Hard rule 1: the only network peer is the local Ollama server. The client refuses non-loopback
hosts and cloud models and bypasses any HTTP proxy. Requests are deterministic (temperature 0,
fixed seed, one fixed context size) and every reply is validated against a pydantic schema.
Truncated or invalid replies raise errors and are never cached.
"""

from __future__ import annotations

import hashlib
import http.client
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
from ratio.context import estimated_tokens, system_with_schema
from ratio.netguard import is_loopback_host

T = TypeVar("T", bound=BaseModel)
Transport = Callable[[str, dict | None, float], dict]

_MIN_TOKENS_PER_SECOND = 10  # the read timeout grows with the reply budget (slow laptops, long label replies)
_PROMPT_FIX = {
    "extraction": "lower extraction.chunk_chars in ratio/config/settings.yaml",
    "labels": "lower absence.max_shortlist or absence.per_hearing_top_k in ratio/config/settings.yaml",
    "argument_check": "lower reuse.argument_max_passages in ratio/config/settings.yaml",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect could point anywhere; the local Ollama server never sends one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201 - urllib signature
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused", headers, fp)


_PROXYLESS = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


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


class CacheDamaged(LLMError):
    """A cache file exists but cannot be read."""


class PromptTooLong(LLMError):
    """The prompt and the reply budget would not fit in the model's context window."""


@dataclass(frozen=True)
class HttpError(Exception):
    status: int
    message: str


def model_from_env(settings: LLMSettings) -> str:
    return os.environ.get("RATIO_MODEL") or settings.default_model


def read_timeout(settings: LLMSettings, num_predict: int) -> float:
    """Seconds to wait for a whole (non-streamed) reply; RATIO_OLLAMA_TIMEOUT overrides it."""
    override = os.environ.get("RATIO_OLLAMA_TIMEOUT")
    if override:
        return float(override)
    return settings.timeout_seconds + num_predict / _MIN_TOKENS_PER_SECOND


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
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except (OSError, http.client.HTTPException):  # reset or cut short: the status still says what went wrong
            detail = str(exc.reason or "")
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
        _refuse_cloud_model(self._model)  # fast check on the name; check_model() asks the server
        self._transport = transport or urllib_transport
        self._verified_local = False

    @property
    def model(self) -> str:
        return self._model

    def _post(self, path: str, body: dict | None, timeout: float | None = None) -> dict:
        url = f"{self._host}{path}"
        try:
            return self._transport(url, body, timeout or self._settings.timeout_seconds)
        except HttpError as exc:
            missing_model = "model" in exc.message.lower() and "not found" in exc.message.lower()  # Ollama's wording
            if exc.status == 404 and path in ("/api/chat", "/api/show") and missing_model:
                raise ModelNotAvailable(
                    f"Model {self._model!r} is not available. Run `ollama pull {self._model}` once while online."
                ) from exc
            if exc.status == 404:
                raise LLMError(f"unexpected 404 from {url}; RATIO_OLLAMA_HOST should look like http://127.0.0.1:11434") from exc
            raise LLMError(f"Ollama returned HTTP {exc.status}: {exc.message}") from exc

    def version(self) -> str:
        return str(self._post("/api/version", None).get("version", "unknown"))

    def check_model(self) -> dict:
        """Confirm the model is pulled and local; returns Ollama's description of it."""
        info = self._post("/api/show", {"model": self._model})
        if info.get("remote_host") or info.get("remote_model"):
            raise LLMError(f"model {self._model!r} is served by an Ollama cloud host; only local models are allowed")
        self._verified_local = True
        return info

    def _check_prompt_size(self, messages: Sequence[dict], num_predict: int, purpose: str) -> None:
        estimate = estimated_tokens(*(m["content"] for m in messages))
        if estimate + num_predict > self._settings.num_ctx:
            raise PromptTooLong(
                f"{purpose} prompt too long for the {self._settings.num_ctx}-token context window "
                f"(about {estimate} tokens); {_PROMPT_FIX.get(purpose, 'shorten the input')}"
            )

    def _chat(self, messages: list[dict], schema_json: dict, options: dict, purpose: str) -> str:
        self._check_prompt_size(messages, options["num_predict"], purpose)
        if not self._verified_local:
            self.check_model()  # a local alias can still be backed by an Ollama cloud model
        body = {
            "model": self._model,
            "messages": messages,
            "format": schema_json,
            "stream": False,
            "options": options,
            "keep_alive": self._settings.keep_alive,
        }
        response = self._post("/api/chat", body, read_timeout(self._settings, options["num_predict"]))
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
        system_text = system_with_schema(system, schema)
        messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user}]
        reply = self._chat(messages, schema_json, options, purpose)
        try:
            return schema.model_validate_json(reply)
        except ValidationError as first_error:
            correction = (
                f"Your previous reply did not match the schema ({_short(first_error)}). "
                "Reply again with valid JSON only."
            )
            retry = messages + [{"role": "assistant", "content": reply}, {"role": "user", "content": correction}]
            reply = self._chat(retry, schema_json, options, purpose)
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
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise CacheDamaged(f"damaged cache file {path}; delete it") from exc
                if not isinstance(record, dict) or "response" not in record:
                    raise CacheDamaged(f"damaged cache file {path}; delete it")
                return record
        return None

    def put(self, key: str, record: dict) -> None:
        self._write_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self._write_dir, delete=False, suffix=".tmp") as tmp:
            try:
                tmp.write(text)
                tmp.flush()
                os.fsync(tmp.fileno())
            except OSError:
                Path(tmp.name).unlink(missing_ok=True)
                raise
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

    def _cached(self, key: str, schema: type[T]) -> T | None:
        """The cached reply; a damaged entry is an error when replaying and a miss when live."""
        try:
            record = self._cache.get(key)
            return None if record is None else schema.model_validate(record["response"])
        except (CacheDamaged, ValidationError) as exc:
            if self._client is None:
                raise CacheDamaged(f"cannot replay cache entry {key}: {exc}") from exc
            return None

    def complete_json(self, *, system: str, user: str, schema: type[T], purpose: str) -> T:
        options = request_options(self._settings, purpose)
        key = ResponseCache.key(
            model=self._model, purpose=purpose, system=system, user=user, schema_json=schema.model_json_schema(), options=options
        )
        self._used.add(key)
        cached = self._cached(key, schema)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        if self._client is None:
            raise CacheMiss(f"no cached {purpose} reply for this request (replay-only mode, model {self._model})")
        result = self._client.complete_json(system=system, user=user, schema=schema, purpose=purpose)
        self._cache.put(
            key,
            {"key": key, "model": self._model, "purpose": purpose, "schema": schema.__name__, "response": result.model_dump(mode="json")},
        )
        return result
