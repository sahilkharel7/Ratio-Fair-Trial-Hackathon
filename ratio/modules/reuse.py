"""Module 3: Reasoning Reuse Detector.

1. Split the judgment into the court's own reasoning and legitimate quotation (reuse_exclusions).
2. Verbatim reuse: word 5-gram shingles. MinHash LSH (Jaccard) and LSH Ensemble (containment)
   propose indictment passages for each reasoning passage; exact Jaccard or containment confirms.
3. Paraphrase: a reasoning passage with no verbatim match whose embedding cosine with an
   indictment passage reaches the threshold, and that shares content words with it.
4. Score: share of the reasoning's characters traceable to the indictment (the matched 5-gram
   text of verbatim pairs, the whole passage for paraphrases).
5. Unaddressed defence arguments: for each defence argument in the notes, the model lists the
   reasoning passages that respond to it; restating the argument is not a response.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from datasketch import MinHash, MinHashLSH, MinHashLSHEnsemble
from unidecode import unidecode

from ratio.config import RatioConfig, ReuseSettings, StandardRef
from ratio.context import AnalysisContext
from ratio.embeddings import cosine_matrix
from ratio.messages import filter_model_note, render
from ratio.modules.reuse_exclusions import PassageRules, classify, compile_rules, is_statute_quote
from ratio.modules.reuse_prompts import ARGUMENT_SYSTEM, ArgumentReply, argument_user_prompt
from ratio.results import ArgumentCheck, CharRange, ExcludedPassage, ReusePair, ReuseResult
from ratio.schema import Argument, CaseRecord, Document, Evidence, Flag, Passage, stable_id

REUSE_STANDARD = "reasoning_reuse"
ARGUMENT_STANDARD = "unaddressed_defense_argument"
MIN_CONTENT_WORD_CHARS = 3
_WORD = re.compile(r"\w+")
_NUMBER = re.compile(r"\d+")
_STOPWORDS = frozenset(
    """about above after again against all also and any are because been before being below between
    both but can could did does doing down during each either for from further had has have having her
    here hers him his how into its itself may might more most must nor not now off once only other our
    ours out over own same shall she should some such than that the their theirs them then there these
    they this those through too under until upon very was were what when where which while who whom
    whose why will with within without would you your""".split()
)

Shingle = tuple[str, ...]


@dataclass(frozen=True)
class _Shingled:
    passage: Passage
    words: int
    shingles: dict[Shingle, tuple[tuple[int, int], ...]]  # 5-gram -> absolute offsets of each occurrence


def _tokens(passage: Passage) -> list[tuple[str, int, int]]:
    offset = passage.span.start
    return [(unidecode(m.group(0)).lower(), offset + m.start(), offset + m.end()) for m in _WORD.finditer(passage.span.text)]


def shingle(passage: Passage, size: int) -> _Shingled:
    tokens = _tokens(passage)
    found: dict[Shingle, list[tuple[int, int]]] = {}
    for i in range(len(tokens) - size + 1):
        window = tokens[i : i + size]
        found.setdefault(tuple(token for token, _, _ in window), []).append((window[0][1], window[-1][2]))
    return _Shingled(passage, len(tokens), {key: tuple(value) for key, value in found.items()})


def content_words(passage: Passage) -> frozenset[str]:
    return frozenset(
        word
        for word, _, _ in _tokens(passage)
        if len(word) >= MIN_CONTENT_WORD_CHARS and word not in _STOPWORDS and not word.isdigit()
    )


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> tuple[CharRange, ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(CharRange(start=start, end=end) for start, end in merged)


def _minhash(item: _Shingled, settings: ReuseSettings) -> MinHash:
    minhash = MinHash(num_perm=settings.minhash_num_perm, seed=settings.minhash_seed)
    for key in item.shingles:
        minhash.update(" ".join(key).encode("utf-8"))
    return minhash


def lsh_candidates(reasoning: Sequence[_Shingled], sources: Sequence[_Shingled], settings: ReuseSettings) -> dict[str, set[str]]:
    """Reasoning passage id -> ids of the indictment passages MinHash LSH proposes for it."""
    indexed = [source for source in sources if source.shingles]
    if not indexed:
        return {}
    hashes = {source.passage.id: _minhash(source, settings) for source in indexed}
    jaccard_index = MinHashLSH(threshold=settings.lsh_threshold, num_perm=settings.minhash_num_perm)
    for passage_id, minhash in hashes.items():
        jaccard_index.insert(passage_id, minhash)
    containment_index = MinHashLSHEnsemble(threshold=settings.lsh_containment_threshold, num_perm=settings.minhash_num_perm)
    containment_index.index([(s.passage.id, hashes[s.passage.id], len(s.shingles)) for s in indexed])
    proposed: dict[str, set[str]] = {}
    for item in reasoning:
        if item.shingles:
            query = _minhash(item, settings)
            proposed[item.passage.id] = set(jaccard_index.query(query)) | set(containment_index.query(query, len(item.shingles)))
    return proposed


def _pair_ids(record: CaseRecord, kind: str, judgment: Passage, indictment: Passage) -> dict[str, str]:
    pair_id = stable_id(record.case_id, "reuse", kind, judgment.id, indictment.id)
    return {"id": pair_id, "flag_id": stable_id(pair_id, "flag")}


def _verbatim(record: CaseRecord, item: _Shingled, source: _Shingled, rules: PassageRules, settings: ReuseSettings) -> ReusePair | None:
    shared = item.shingles.keys() & source.shingles.keys()
    if not shared:
        return None
    jaccard = len(shared) / len(item.shingles.keys() | source.shingles.keys())
    containment = len(shared) / len(item.shingles)
    if jaccard < settings.verbatim_jaccard and containment < settings.verbatim_containment:
        return None
    return ReusePair(
        **_pair_ids(record, "verbatim", item.passage, source.passage),
        kind="verbatim",
        judgment=item.passage.span,
        indictment=source.passage.span,
        jaccard=round(jaccard, 4),
        containment=round(containment, 4),
        judgment_ranges=merge_ranges(r for key in shared for r in item.shingles[key]),
        indictment_ranges=merge_ranges(r for key in shared for r in source.shingles[key]),
        matches_charge_particulars=bool(rules.recital.search(source.passage.span.text)),
    )


def _paraphrases(
    record: CaseRecord, items: Sequence[_Shingled], sources: Sequence[_Shingled], rules: PassageRules, ctx: AnalysisContext
) -> list[ReusePair]:
    """For each passage, the most similar indictment passage above the cosine threshold that also shares content words."""
    settings = ctx.config.settings.reuse
    if not items or not sources:
        return []
    similarity = cosine_matrix(
        ctx.embedder.encode([item.passage.span.text for item in items]),
        ctx.embedder.encode([source.passage.span.text for source in sources]),
    )
    source_words = [content_words(source.passage) for source in sources]
    pairs = []
    for row, item in enumerate(items):
        words = content_words(item.passage)
        for col in np.argsort(-similarity[row], kind="stable"):
            cosine = float(similarity[row, col])
            if cosine < settings.paraphrase_cosine:
                break
            if len(words & source_words[col]) >= settings.paraphrase_min_shared_words:
                source = sources[col].passage
                pairs.append(
                    ReusePair(
                        **_pair_ids(record, "paraphrase", item.passage, source),
                        kind="paraphrase",
                        judgment=item.passage.span,
                        indictment=source.span,
                        cosine=round(cosine, 4),
                        matches_charge_particulars=bool(rules.recital.search(source.span.text)),
                    )
                )
                break
    return pairs


def find_pairs(
    record: CaseRecord, reasoning: Sequence[Passage], sources: Sequence[Passage], rules: PassageRules, ctx: AnalysisContext
) -> tuple[ReusePair, ...]:
    settings = ctx.config.settings.reuse
    eligible = [s for s in (shingle(p, settings.shingle_size) for p in reasoning) if s.words >= settings.min_passage_words]
    targets = [s for s in (shingle(p, settings.shingle_size) for p in sources) if s.words >= settings.min_passage_words]
    proposed = lsh_candidates(eligible, targets, settings)
    verbatim = []
    for item in eligible:
        for source in targets:  # document order keeps the output deterministic
            if source.passage.id in proposed.get(item.passage.id, ()):
                pair = _verbatim(record, item, source, rules, settings)
                verbatim.extend([pair] if pair else [])
    matched = {pair.judgment for pair in verbatim}
    paraphrased = _paraphrases(record, [item for item in eligible if item.passage.span not in matched], targets, rules, ctx)
    return tuple(sorted(verbatim + paraphrased, key=lambda pair: (pair.judgment.start, pair.indictment.start)))


def score(pairs: Sequence[ReusePair], reasoning: Sequence[Passage]) -> tuple[float | None, int, int, int]:
    """(score, verbatim chars, paraphrase chars, reasoning chars)."""
    reasoning_chars = sum(len(passage.span.text) for passage in reasoning)
    verbatim = merge_ranges((r.start, r.end) for pair in pairs if pair.kind == "verbatim" for r in pair.judgment_ranges)
    verbatim_chars = sum(r.end - r.start for r in verbatim)
    paraphrase_chars = sum(len(pair.judgment.text) for pair in pairs if pair.kind == "paraphrase")
    if not reasoning_chars:
        return None, verbatim_chars, paraphrase_chars, 0
    return min(1.0, (verbatim_chars + paraphrase_chars) / reasoning_chars), verbatim_chars, paraphrase_chars, reasoning_chars


def _pair_flag(record: CaseRecord, pair: ReusePair, standard: StandardRef, config: RatioConfig) -> Flag:
    if pair.kind == "verbatim":
        status = "verbatim_reuse"
        message = render(config.messages, "reuse_verbatim", percent=round(100 * (pair.containment or 0)), n=config.settings.reuse.shingle_size)
    else:
        status = "paraphrase_reuse"
        message = render(config.messages, "reuse_paraphrase", similarity=f"{pair.cosine:.2f}")
    return Flag(
        id=pair.flag_id or stable_id(pair.id, "flag"),
        case_id=record.case_id,
        module="reuse",
        standard_id=REUSE_STANDARD,
        standard_label=standard.label,
        status=status,
        message=message,
        evidence=(Evidence(role="judgment", span=pair.judgment), Evidence(role="indictment", span=pair.indictment)),
        citation=standard.citation,
        review_status=standard.review_status,
    )


def _passages_to_check(argument: Argument, reasoning: Sequence[Passage], ctx: AnalysisContext) -> list[Passage]:
    """Every reasoning passage, or the most similar ones when the reasoning is too long for one prompt."""
    limit = ctx.config.settings.reuse.argument_max_passages
    if len(reasoning) <= limit:
        return list(reasoning)
    vectors = ctx.embedder.encode([passage.span.text for passage in reasoning])
    similarity = cosine_matrix(vectors, ctx.embedder.encode([argument.text]))[:, 0]
    top = sorted(range(len(reasoning)), key=lambda i: (-float(similarity[i]), i))[:limit]
    return [reasoning[i] for i in sorted(top)]


def check_argument(
    record: CaseRecord, argument: Argument, reasoning: Sequence[Passage], standard: StandardRef, ctx: AnalysisContext
) -> tuple[ArgumentCheck, Flag | None]:
    shown = _passages_to_check(argument, reasoning, ctx)
    numbered = [(f"P{n}", passage) for n, passage in enumerate(shown, start=1)]
    reply = ctx.llm.complete_json(
        system=ARGUMENT_SYSTEM, user=argument_user_prompt(argument, numbered), schema=ArgumentReply, purpose="argument_check"
    )
    by_number = dict(enumerate(shown, start=1))
    responding: dict[str, Passage] = {}
    for raw in reply.responding:
        found = _NUMBER.search(raw)  # "P7", "7" and "P7: As to ..." all name passage 7
        passage = by_number.get(int(found.group(0))) if found else None
        if passage is not None:
            responding[passage.id] = passage
    spans = tuple(passage.span for passage in sorted(responding.values(), key=lambda p: p.span.start))
    note = filter_model_note(reply.note, ctx.config.messages.block_list)
    flag = None
    if not spans:
        flag = Flag(
            id=stable_id(record.case_id, "reuse", "unaddressed", argument.id),
            case_id=record.case_id,
            module="reuse",
            standard_id=ARGUMENT_STANDARD,
            standard_label=standard.label,
            status="unaddressed_argument",
            message=render(ctx.config.messages, "reuse_unaddressed", count=len(shown)),
            evidence=(Evidence(role="argument", span=argument.span),),
            citation=standard.citation,
            review_status=standard.review_status,
            model_note=note,
        )
    check = ArgumentCheck(
        argument_id=argument.id,
        argument=argument.span,
        addressed=bool(spans),
        responding=spans,
        passages_checked=len(shown),
        model_note=note,
        flag_id=flag.id if flag else None,
    )
    return check, flag


def _first(record: CaseRecord, doc_type: str) -> Document | None:
    documents = record.documents_of_type(doc_type)
    return documents[0] if documents else None


def run(record: CaseRecord, ctx: AnalysisContext) -> ReuseResult:
    judgment, indictment = _first(record, "judgment"), _first(record, "indictment")
    indictment_id = indictment.id if indictment else None
    if judgment is None:
        return ReuseResult(judgment_doc_id=None, indictment_doc_id=indictment_id)
    config = ctx.config
    rules = compile_rules(config.settings.reuse)
    passages = record.passages_of(judgment.id)
    reasons = classify(passages, judgment.text, record.citations, rules)
    reasoning = [p for p in passages if p.id in reasons and reasons[p.id] is None]
    excluded = tuple(
        ExcludedPassage(passage_id=p.id, span=p.span, reason=reason) for p in passages if (reason := reasons.get(p.id)) is not None
    )
    pairs: tuple[ReusePair, ...] = ()
    if indictment is not None:
        sources = [
            p for p in record.passages_of(indictment.id) if p.kind == "body" and not is_statute_quote(p, record.citations, rules)
        ]
        pairs = find_pairs(record, reasoning, sources, rules, ctx)
    reuse_standard = config.standard(REUSE_STANDARD)
    flags = [_pair_flag(record, pair, reuse_standard, config) for pair in pairs]
    checks = []
    if reasoning:
        argument_standard = config.standard(ARGUMENT_STANDARD)
        for argument in (a for a in record.arguments if a.party == "defense"):
            check, flag = check_argument(record, argument, reasoning, argument_standard, ctx)
            checks.append(check)
            flags.extend([flag] if flag else [])
    value, verbatim_chars, paraphrase_chars, reasoning_chars = score(pairs, reasoning)
    return ReuseResult(
        judgment_doc_id=judgment.id,
        indictment_doc_id=indictment_id,
        score=value if indictment is not None else None,
        verbatim_chars=verbatim_chars,
        paraphrase_chars=paraphrase_chars,
        reasoning_chars=reasoning_chars,
        reasoning_passage_ids=tuple(p.id for p in reasoning),
        pairs=pairs,
        excluded=excluded,
        arguments=tuple(checks),
        flags=tuple(flags),
    )
