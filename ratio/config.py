"""Load and validate Ratio's YAML configuration: rubric, benchmarks, ruling codes, settings, messages."""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from ratio.paths import CONFIG_DIR
from ratio.schema import EventType, Frozen, ReviewStatus

# Hard rule 5: the only benchmark the team has confirmed (GC35 para 33, 48 hours). A benchmark can
# be marked "confirmed" only if its id is listed here, so confirming another one is a reviewed change.
CONFIRMED_BENCHMARKS: frozenset[str] = frozenset({"gc35_48h"})

IndicatorLabel = Literal["supports", "contradicts"]


class ConfigError(ValueError):
    """A configuration file is missing or invalid."""


class SourceRef(Frozen):
    symbol: str
    title: str
    url: str


class CitationRef(Frozen):
    instrument: str
    paras: str = Field(min_length=1)
    quote: str | None = None


# --- rubric.yaml ----------------------------------------------------------------------------


class RubricIndicator(Frozen):
    id: str
    text: str
    covers_all_hearings: bool = False
    keywords: tuple[str, ...] = ()  # context indicators: a note is shown as context only if it mentions one


class RubricPart(Frozen):
    id: str
    label: str
    required: bool
    scope: Literal["case", "per_hearing"]
    keywords: tuple[str, ...] = ()
    compliance: tuple[RubricIndicator, ...] = Field(min_length=1)
    violation: tuple[RubricIndicator, ...] = Field(min_length=1)


class RubricItem(Frozen):
    id: str
    name: str
    provision: str
    text: str
    citation: CitationRef
    review_status: ReviewStatus
    parts: tuple[RubricPart, ...] = Field(min_length=1)
    context: tuple[RubricIndicator, ...] = ()
    follow_up: str = Field(min_length=1)

    @model_validator(mode="after")
    def _has_a_required_part(self) -> RubricItem:
        if not any(part.required for part in self.parts):
            raise ValueError(f"{self.id}: at least one part must be required, or no status can be reached")
        return self


class Rubric(Frozen):
    version: int
    sources: dict[str, SourceRef]
    items: tuple[RubricItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique_and_sources_known(self) -> Rubric:
        ids: list[str] = []
        for item in self.items:
            if item.citation.instrument not in self.sources:
                raise ValueError(f"{item.id}: unknown citation instrument {item.citation.instrument!r}")
            ids.append(item.id)
            for part in item.parts:
                ids.append(part.id)
                ids.extend(indicator.id for indicator in part.compliance + part.violation)
            ids.extend(indicator.id for indicator in item.context)
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate rubric ids: {duplicates}")
        return self


# --- benchmarks.yaml ------------------------------------------------------------------------


class Benchmark(Frozen):
    id: str
    name: str
    provision: str
    from_event: EventType
    to_event: EventType
    pairing: Literal["first_after"] = "first_after"
    threshold_hours: float | None = Field(default=None, gt=0)
    citation: CitationRef
    review_status: ReviewStatus
    note: str = ""
    flag_note: str = ""

    @model_validator(mode="after")
    def _confirmation_rules(self) -> Benchmark:
        if self.review_status == "confirmed":
            if self.id not in CONFIRMED_BENCHMARKS:
                raise ValueError(f"benchmark {self.id!r} is not confirmed by the team; mark it needs_legal_review")
            if self.threshold_hours is None:
                raise ValueError(f"confirmed benchmark {self.id!r} needs threshold_hours")
        elif self.threshold_hours is not None:
            raise ValueError(f"benchmark {self.id!r} is not confirmed, so it may not carry a threshold (hard rule 5)")
        return self


class Benchmarks(Frozen):
    version: int
    sources: dict[str, SourceRef]
    benchmarks: tuple[Benchmark, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique_and_sources_known(self) -> Benchmarks:
        ids = [benchmark.id for benchmark in self.benchmarks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate benchmark ids")
        for benchmark in self.benchmarks:
            if benchmark.citation.instrument not in self.sources:
                raise ValueError(f"{benchmark.id}: unknown citation instrument {benchmark.citation.instrument!r}")
        return self


# --- ruling_codes.yaml ----------------------------------------------------------------------


class RulingCode(Frozen):
    id: str
    label: str
    numeric: bool = False


class RateDef(Frozen):
    id: str
    label: str
    numerator: tuple[str, ...] = Field(min_length=1)
    denominator: tuple[str, ...] = Field(min_length=1)
    combine: Literal["first", "last", "any", "all"]

    @model_validator(mode="after")
    def _numerator_within_denominator(self) -> RateDef:
        if not set(self.numerator) <= set(self.denominator):
            raise ValueError(f"rate {self.id!r}: numerator codes must also be denominator codes")
        return self


class RulingCodes(Frozen):
    version: int
    codes: tuple[RulingCode, ...] = Field(min_length=1)
    rates: tuple[RateDef, ...] = ()
    descriptive: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _references_known(self) -> RulingCodes:
        known = [code.id for code in self.codes]
        if len(known) != len(set(known)):
            raise ValueError("duplicate ruling codes")
        for rate in self.rates:
            unknown = set(rate.denominator) - set(known)
            if unknown:
                raise ValueError(f"rate {rate.id!r} uses unknown codes {sorted(unknown)}")
        if set(self.descriptive) - set(known):
            raise ValueError("descriptive list uses unknown codes")
        return self


# --- settings.yaml --------------------------------------------------------------------------


class NumPredict(Frozen):
    extraction: int = Field(gt=0)
    labels: int = Field(gt=0)
    argument_check: int = Field(gt=0)


class LLMSettings(Frozen):
    default_model: str
    host: str
    num_ctx: int = Field(gt=0)
    temperature: float = Field(ge=0)
    seed: int
    timeout_seconds: float = Field(gt=0)
    keep_alive: str
    num_predict: NumPredict


class ExtractionSettings(Frozen):
    chunk_chars: int = Field(gt=200)
    chunk_overlap_chars: int = Field(ge=0)
    min_quote_chars: int = Field(gt=0)
    fuzzy_min_quote_chars: int = Field(gt=0)
    fuzzy_threshold: float = Field(gt=0, le=100)
    argument_markers: tuple[str, ...] = Field(min_length=1)


class EmbeddingSettings(Frozen):
    model_dir: str


class AbsenceSettings(Frozen):
    shortlist_top_k: int = Field(gt=0)
    per_hearing_top_k: int = Field(gt=0)
    max_shortlist: int = Field(gt=0)
    shortlist_min_similarity: float
    context_min_similarity: float


class ReuseSettings(Frozen):
    shingle_size: int = Field(gt=1)
    minhash_num_perm: int = Field(gt=0)
    minhash_seed: int
    lsh_threshold: float = Field(gt=0, lt=1)
    lsh_weights: tuple[float, float] = (0.1, 0.9)
    verbatim_jaccard: float = Field(gt=0, le=1)
    verbatim_containment: float = Field(gt=0, le=1)
    min_source_share: float = Field(default=0.1, gt=0, le=1)
    paraphrase_cosine: float = Field(gt=0, le=1)
    paraphrase_min_shared_words: int = Field(ge=0)
    min_passage_words: int = Field(gt=0)
    argument_max_passages: int = Field(gt=0)
    non_reasoning_headings: tuple[str, ...]
    reasoning_heading_words: tuple[str, ...] = ()
    recital_markers: tuple[str, ...] = Field(min_length=1)
    statute_markers: tuple[str, ...] = Field(min_length=1)
    party_terms: tuple[str, ...] = Field(min_length=1)
    not_a_party_after: tuple[str, ...] = ()
    sentence_openers: tuple[str, ...] = ()
    attribution_prefixes: tuple[str, ...] = ()
    attribution_verbs: tuple[str, ...] = Field(min_length=1)
    attribution_possessives: tuple[str, ...] = ()
    endorsement_words: tuple[str, ...] = Field(min_length=1)
    court_voice_markers: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _candidates_below_confirmation(self) -> ReuseSettings:
        # LSH only proposes pairs; it must propose everything the exact check could confirm.
        if self.lsh_threshold > self.verbatim_jaccard:
            raise ValueError("the LSH threshold must not exceed the Jaccard threshold it proposes candidates for")
        if abs(sum(self.lsh_weights) - 1.0) > 1e-9:
            raise ValueError("lsh_weights must add up to 1")
        return self

    @model_validator(mode="after")
    def _patterns_compile(self) -> ReuseSettings:
        lists = ("non_reasoning_headings", "reasoning_heading_words", "recital_markers", "statute_markers", "party_terms",
                 "not_a_party_after", "sentence_openers", "attribution_prefixes", "attribution_verbs",
                 "attribution_possessives", "endorsement_words", "court_voice_markers")  # fmt: skip
        for name in lists:
            for pattern in getattr(self, name):
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"reuse.{name}: {pattern!r} is not a valid pattern ({exc})") from exc
        return self


class JudgeSettings(Frozen):
    min_case_count: int = Field(ge=1)
    alpha: float = Field(gt=0, lt=1)
    name_match_threshold: float = Field(gt=0, le=100)
    ambiguous_floor: float = Field(gt=0, le=100)
    date_tolerance_days: int = Field(gt=0)


class UISettings(Frozen):
    context_chars: int = Field(gt=0)


class Settings(Frozen):
    version: int
    llm: LLMSettings
    extraction: ExtractionSettings
    embeddings: EmbeddingSettings
    absence: AbsenceSettings
    reuse: ReuseSettings
    judges: JudgeSettings
    ui: UISettings


# --- messages.yaml --------------------------------------------------------------------------


class Messages(Frozen):
    version: int
    templates: dict[str, str]
    notes: dict[str, str]
    labels: dict[str, str] = {}
    event_labels: dict[str, str]
    block_list: tuple[str, ...] = Field(min_length=1)

    def label(self, key: str) -> str:
        return self.labels.get(key, key.replace("_", " "))


# --- standards.yaml ------------------------------------------------------------------------


class StandardRef(Frozen):
    label: str
    citation: str
    review_status: ReviewStatus


class Standards(Frozen):
    version: int
    standards: dict[str, StandardRef]


# --- all of it ------------------------------------------------------------------------------


class RatioConfig(Frozen):
    rubric: Rubric
    benchmarks: Benchmarks
    ruling_codes: RulingCodes
    settings: Settings
    messages: Messages
    standards: Standards

    def standard(self, standard_id: str) -> StandardRef:
        return self.standards.standards[standard_id]

    def rubric_item(self, item_id: str) -> RubricItem:
        for item in self.rubric.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def benchmark(self, benchmark_id: str) -> Benchmark:
        for benchmark in self.benchmarks.benchmarks:
            if benchmark.id == benchmark_id:
                return benchmark
        raise KeyError(benchmark_id)

    def rate(self, rate_id: str) -> RateDef:
        for rate in self.ruling_codes.rates:
            if rate.id == rate_id:
                return rate
        raise KeyError(rate_id)

    @property
    def indicator_index(self) -> dict[str, tuple[RubricPart, IndicatorLabel]]:
        """Evidence indicator id -> (rubric part, label it licenses). Context indicators are excluded."""
        index: dict[str, tuple[RubricPart, IndicatorLabel]] = {}
        for item in self.rubric.items:
            for part in item.parts:
                index.update({indicator.id: (part, "supports") for indicator in part.compliance})
                index.update({indicator.id: (part, "contradicts") for indicator in part.violation})
        return index


_FILES: dict[str, tuple[str, type[Frozen]]] = {
    "rubric": ("rubric.yaml", Rubric),
    "benchmarks": ("benchmarks.yaml", Benchmarks),
    "ruling_codes": ("ruling_codes.yaml", RulingCodes),
    "settings": ("settings.yaml", Settings),
    "messages": ("messages.yaml", Messages),
    "standards": ("standards.yaml", Standards),
}


def _read_yaml(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path.name}: {exc}") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc


def load_config(config_dir: Path = CONFIG_DIR) -> RatioConfig:
    """Read and validate every config file; raises ConfigError naming the file at fault."""
    parts: dict[str, Frozen] = {}
    for key, (filename, model) in _FILES.items():
        data = _read_yaml(Path(config_dir) / filename)
        try:
            parts[key] = model.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"{filename} is invalid:\n{exc}") from exc
    return RatioConfig(**parts)


@functools.lru_cache(maxsize=1)
def default_config() -> RatioConfig:
    return load_config(CONFIG_DIR)
