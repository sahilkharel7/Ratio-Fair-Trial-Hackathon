"""Load and validate Ratio's YAML configuration: rubric, benchmarks, ruling codes, settings, messages."""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from ratio.netguard import is_loopback_host
from ratio.paths import CONFIG_DIR
from ratio.schema import FLAG_STATUSES, EventType, Frozen, ReviewStatus

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

_KEYWORD = re.compile(r"[a-z0-9 -]*[a-z0-9][a-z0-9 -]*")


def _plain(keywords: tuple[str, ...]) -> tuple[str, ...]:
    """Keywords are matched in lowercase text whose punctuation is read as spaces (absence.py), so a
    keyword with capitals or punctuation would never match."""
    wrong = [keyword for keyword in keywords if not _KEYWORD.fullmatch(keyword) or "  " in keyword]
    if wrong:
        raise ValueError(f"keywords may hold only lowercase ASCII letters, digits, single spaces and hyphens: {wrong}")
    return keywords


class RubricIndicator(Frozen):
    id: str
    text: str
    covers_all_hearings: bool = False
    keywords: tuple[str, ...] = ()  # context indicators: a note is shown as context only if it mentions one

    @field_validator("keywords")
    @classmethod
    def _plain_keywords(cls, keywords: tuple[str, ...]) -> tuple[str, ...]:
        return _plain(keywords)


class RubricPart(Frozen):
    id: str
    label: str
    required: bool
    scope: Literal["case", "per_hearing"]
    keywords: tuple[str, ...] = ()
    compliance: tuple[RubricIndicator, ...] = Field(min_length=1)
    violation: tuple[RubricIndicator, ...] = Field(min_length=1)

    @field_validator("keywords")
    @classmethod
    def _plain_keywords(cls, keywords: tuple[str, ...]) -> tuple[str, ...]:
        return _plain(keywords)


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
    steelman: int = Field(default=700, gt=0)
    steelman_check: int = Field(default=200, gt=0)


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
    argument_continuations: tuple[str, ...] = ()
    argument_list_nouns: tuple[str, ...] = ()
    event_support: dict[EventType, str] = Field(default_factory=dict)

    @field_validator("event_support")
    @classmethod
    def _patterns_compile(cls, patterns: dict[EventType, str]) -> dict[EventType, str]:
        for event_type, pattern in patterns.items():
            try:
                re.compile(rf"\b(?:{pattern})")  # as extraction compiles it
            except re.error as exc:
                raise ValueError(f"event_support for {event_type} is not a valid pattern: {exc}") from exc
        return patterns


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
    title_words: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _floor_below_threshold(self) -> JudgeSettings:
        if self.ambiguous_floor >= self.name_match_threshold:
            raise ValueError("judges.ambiguous_floor must be below name_match_threshold")
        return self


class RenewalSettings(Frozen):
    repeated_share: float = Field(default=0.6, gt=0, le=1)  # flag an order when this share of its grounds repeats earlier orders


class SteelmanSettings(Frozen):
    top_k_passages: int = Field(default=8, gt=0)  # record passages the model may argue from, per finding
    max_arguments: int = Field(default=3, gt=0)


class UISettings(Frozen):
    context_chars: int = Field(gt=0)


class PrecedentSettings(Frozen):
    """Similar cases (ratio/precedents.py). The corpus file is the source of truth; OpenSearch is an
    optional index of it and must run on this computer."""

    backend: Literal["sqlite", "opensearch"] = "sqlite"
    opensearch_url: str = "http://127.0.0.1:9200"
    opensearch_index: str = Field(default="precedent_passages", pattern=r"^[a-z][a-z0-9_-]*$")
    min_shared: int = Field(default=2, ge=1)  # shared fact patterns a precedent needs, at least one about procedure
    top_k: int = Field(default=8, gt=0)

    @field_validator("opensearch_url")
    @classmethod
    def _on_this_computer(cls, url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not is_loopback_host(parts.hostname or ""):
            raise ValueError(f"opensearch_url must be on this computer (127.0.0.1), not {url!r}")
        return url


class Settings(Frozen):
    version: int
    llm: LLMSettings
    extraction: ExtractionSettings
    absence: AbsenceSettings
    reuse: ReuseSettings
    judges: JudgeSettings
    renewal: RenewalSettings = RenewalSettings()
    steelman: SteelmanSettings = SteelmanSettings()
    precedents: PrecedentSettings = PrecedentSettings()
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


# --- jurisprudence.yaml ---------------------------------------------------------------------

OFFICIAL_SOURCE_PREFIX = "https://documents.un.org/"


class JurisprudenceEntry(Frozen):
    """A General Comment paragraph (its own words), or a Committee decision cited only for what the
    General Comment cites it for (the footnote's own words)."""

    id: str
    kind: Literal["general_comment", "views"]
    source: str
    pinpoint: str = Field(min_length=1)
    standards: tuple[str, ...] = Field(min_length=1)
    statuses: tuple[str, ...] = ()  # empty: every status of a finding on these standards
    quote: str | None = None
    case: str | None = None
    communication: str | None = Field(default=None, pattern=r"^\d+(?:[-–]\d+)?/\d{4}$")
    cited_as: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> JurisprudenceEntry:
        if self.kind == "general_comment":
            if not (self.quote or "").strip() or self.case or self.cited_as:
                raise ValueError(f"{self.id}: a General Comment entry has a quote and no case")
        elif not (self.case and self.communication and self.cited_as and self.note) or self.communication not in self.cited_as:
            raise ValueError(f"{self.id}: a decision needs its case, communication number, the footnote that cites it, and a note")
        return self


class Jurisprudence(Frozen):
    version: int
    checked: str
    sources: dict[str, SourceRef]
    entries: tuple[JurisprudenceEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique_and_sources_official(self) -> Jurisprudence:
        ids = [entry.id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate jurisprudence ids")
        if any(not source.url.startswith(OFFICIAL_SOURCE_PREFIX) for source in self.sources.values()):
            raise ValueError(f"every jurisprudence source must be an official UN document ({OFFICIAL_SOURCE_PREFIX})")
        unknown = sorted({entry.source for entry in self.entries} - self.sources.keys())
        if unknown:
            raise ValueError(f"unknown jurisprudence sources: {unknown}")
        return self


# --- fact_patterns.yaml --------------------------------------------------------------------

FacetGroup = Literal["procedure", "profile", "charge"]


class FacetSignal(Frozen):
    """A stored finding that gives a case this facet: its module, standard and status."""

    module: str
    standard: str
    status: str


class FactPattern(Frozen):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1)
    group: FacetGroup
    provision: str | None = None
    description: str = Field(min_length=1)  # what the facts look like; also the extraction prompt's definition
    signals: tuple[FacetSignal, ...] = ()
    keywords: tuple[str, ...] = ()  # regular expressions, matched as whole words, case-insensitive

    @model_validator(mode="after")
    def _one_source(self) -> FactPattern:
        if self.group == "procedure" and (not self.signals or self.keywords):
            raise ValueError(f"{self.id}: a procedure facet comes only from Ratio's findings (signals)")
        if self.group != "procedure" and (not self.keywords or self.signals):
            raise ValueError(f"{self.id}: a profile or charge facet comes only from keywords")
        for keyword in self.keywords:
            try:
                re.compile(keyword)
            except re.error as exc:
                raise ValueError(f"{self.id}: keyword {keyword!r} is not a valid pattern: {exc}") from exc
        return self


@functools.cache
def keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(keywords) + r")\b", re.IGNORECASE)


class FactPatternTaxonomy(Frozen):
    version: int
    review_status: Literal["needs_legal_review"]
    facets: tuple[FactPattern, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> FactPatternTaxonomy:
        ids = [facet.id for facet in self.facets]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate fact pattern ids")
        return self

    def facet(self, facet_id: str) -> FactPattern:
        for facet in self.facets:
            if facet.id == facet_id:
                return facet
        raise KeyError(facet_id)

    @property
    def sha(self) -> str:
        """Identifies this taxonomy; a corpus built with another one is refused."""
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


# --- steelman.yaml -------------------------------------------------------------------------


class SteelmanGround(Frozen):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1)
    requires: str = Field(min_length=1)  # what a quoted passage must show for the ground to apply
    independent: bool = False  # the quote must come from outside the finding's own evidence
    source: str | None = None  # a jurisprudence.yaml source, when a General Comment recognises the ground
    pinpoint: str | None = None
    quote: str | None = None

    @model_validator(mode="after")
    def _quoted_with_source(self) -> SteelmanGround:
        if (self.quote is None) != (self.source is None) or (self.quote is None) != (self.pinpoint is None):
            raise ValueError(f"ground {self.id}: a quote needs its source and pinpoint, and only a quote has them")
        return self


class SteelmanCatalogue(Frozen):
    version: int
    adverse_statuses: tuple[str, ...] = Field(min_length=1)
    grounds: dict[str, tuple[SteelmanGround, ...]]

    @model_validator(mode="after")
    def _ids_unique(self) -> SteelmanCatalogue:
        shared = {ground.id for ground in self.grounds.get("all", ())}
        for standard_id, grounds in self.grounds.items():
            ids = [ground.id for ground in grounds]
            if len(ids) != len(set(ids)) or (standard_id != "all" and shared & set(ids)):
                raise ValueError(f"duplicate steelman ground ids for {standard_id}")
        return self

    def for_standard(self, standard_id: str) -> tuple[SteelmanGround, ...]:
        """The grounds of one standard, then the ones offered for every finding."""
        return self.grounds.get(standard_id, ()) + self.grounds.get("all", ())


# --- all of it ------------------------------------------------------------------------------


class RatioConfig(Frozen):
    rubric: Rubric
    benchmarks: Benchmarks
    ruling_codes: RulingCodes
    settings: Settings
    messages: Messages
    standards: Standards
    jurisprudence: Jurisprudence
    steelman: SteelmanCatalogue
    fact_patterns: FactPatternTaxonomy

    @model_validator(mode="after")
    def _fact_pattern_signals_known(self) -> RatioConfig:
        """A facet rests only on a finding Ratio can make: never on "no evidence", and never on a
        benchmark that needs legal review (hard rule 5)."""
        rubric_ids = {item.id for item in self.rubric.items}
        confirmed = {b.id for b in self.benchmarks.benchmarks if b.review_status == "confirmed"}
        standards = {"absence": rubric_ids, "clock": confirmed, "reuse": set(self.standards.standards), "renewal": set(self.standards.standards)}
        for facet in self.fact_patterns.facets:
            for signal in facet.signals:
                if signal.module not in standards or signal.status not in FLAG_STATUSES.get(signal.module, ()):
                    raise ValueError(f"fact_patterns.yaml: {facet.id} rests on an unknown finding {signal.module}/{signal.status}")
                if signal.standard not in standards[signal.module]:
                    raise ValueError(f"fact_patterns.yaml: {facet.id} rests on {signal.standard!r}, not a rubric item, confirmed benchmark or standard")
        return self

    @model_validator(mode="after")
    def _steelman_sources_known(self) -> RatioConfig:
        unknown = {g.source for grounds in self.steelman.grounds.values() for g in grounds if g.source} - self.jurisprudence.sources.keys()
        if unknown:
            raise ValueError(f"steelman.yaml cites unknown sources: {sorted(unknown)}")
        return self

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
    "jurisprudence": ("jurisprudence.yaml", Jurisprudence),
    "steelman": ("steelman.yaml", SteelmanCatalogue),
    "fact_patterns": ("fact_patterns.yaml", FactPatternTaxonomy),
}


_UNCHECKED_KEY_TAGS = frozenset({"tag:yaml.org,2002:merge", "tag:yaml.org,2002:value"})  # "<<" and "="


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a key written twice in one mapping (PyYAML would keep the last one).

    Each mapping is checked once, with its keys as written: PyYAML flattens a mapping before it
    builds it, and a merge ("<<:") rewrites the keys of the mapping it reads from."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._checked: set[yaml.Node] = set()

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        if node not in self._checked:
            self._checked.add(node)
            seen: set[object] = set()
            for key_node, _ in node.value:
                if not isinstance(key_node, yaml.ScalarNode) or key_node.tag in _UNCHECKED_KEY_TAGS:
                    continue  # merges, "=", and list or mapping keys, which PyYAML itself refuses
                key = self.construct_object(key_node)
                if key in seen:
                    raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
                seen.add(key)
        super().flatten_mapping(node)


def _read_yaml(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path.name}: {exc}") from exc
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - a SafeLoader subclass
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
