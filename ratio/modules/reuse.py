"""Module 3: Reasoning Reuse Detector.

1. Split the judgment into the court's own reasoning and legitimate quotation (reuse_exclusions);
   the indictments' quoted provisions are left out of the comparison as well.
2. Verbatim reuse, on word 5-gram shingles: a reasoning passage is a copy when most of its 5-grams
   appear in the indictment (exact containment over every indictment passage, so a sentence stitched
   from two indictment sentences or excerpted from a long one counts), or when MinHash LSH proposes
   an indictment passage and exact Jaccard similarity confirms a near-copy.
3. Paraphrase: a reasoning passage with no verbatim match whose embedding cosine with an
   indictment passage reaches the threshold, and that shares content words with it.
4. Restated charge: a passage whose matches come only from the indictment's recital of the charge
   is shown as such and left out of the score; a court states the charge it rules on.
5. Score: share of the reasoning's characters traceable to the indictment (the matched 5-gram text
   of verbatim copies, the whole passage for paraphrases).
6. Unaddressed defence arguments: for each defence argument in the notes, the model lists the
   reasoning passages that respond to it; restating the argument is not a response.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from datasketch import MinHash, MinHashLSH
from unidecode import unidecode

from ratio.config import RatioConfig, ReuseSettings, StandardRef
from ratio.context import AnalysisContext, estimated_tokens, system_with_schema
from ratio.embeddings import cosine_matrix
from ratio.messages import filter_model_note, render
from ratio.modules.reuse_exclusions import PassageRules, classify, compile_rules
from ratio.modules.reuse_prompts import ARGUMENT_SYSTEM, ArgumentReply, argument_user_prompt
from ratio.results import ArgumentCheck, CharRange, ExcludedPassage, ReusePair, ReuseResult
from ratio.schema import Argument, CaseRecord, Document, Evidence, Flag, Passage, stable_id

REUSE_STANDARD = "reasoning_reuse"
ARGUMENT_STANDARD = "unaddressed_defense_argument"
MIN_CONTENT_WORD_CHARS = 3
_WORD = re.compile(r"\w+")
_PASSAGE_ID = re.compile(r"\s*P?(\d+)\s*")
_ID_PREFIX = re.compile(r"\s*P(\d+)\s*:")
_NO_PASSAGE = frozenset({"", "none", "no", "nothing", "n/a", "null", "-", "no passage", "no passages"})
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


@dataclass(frozen=True)
class Source:
    """An indictment passage to compare with; ``charge`` marks the recital of the charge."""

    passage: Passage
    charge: bool


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
    """Reasoning passage id -> ids of the indictment passages MinHash LSH proposes as near-copies."""
    indexed = [source for source in sources if source.shingles]
    if not indexed:
        return {}
    index = MinHashLSH(threshold=settings.lsh_threshold, num_perm=settings.minhash_num_perm, weights=settings.lsh_weights)
    for source in indexed:
        index.insert(source.passage.id, _minhash(source, settings))
    return {item.passage.id: set(index.query(_minhash(item, settings))) for item in reasoning if item.shingles}


def _pair_ids(record: CaseRecord, kind: str, judgment: Passage, indictment: Passage, flag_id: str) -> dict[str, str]:
    return {"id": stable_id(record.case_id, "reuse", kind, judgment.id, indictment.id), "flag_id": flag_id}


def _flag_id(record: CaseRecord, judgment: Passage) -> str:
    return stable_id(record.case_id, "reuse", judgment.id, "flag")


def _verbatim(record: CaseRecord, item: _Shingled, targets: list[tuple[_Shingled, bool]], index, proposed, settings) -> list[ReusePair]:
    """The pairs of one reasoning passage that copies the indictment, or [] when it does not."""
    shared: dict[int, set[Shingle]] = {}
    for key in item.shingles:
        for position in index.get(key, ()):
            shared.setdefault(position, set()).add(key)
    near_copies = {
        position
        for position, keys in shared.items()
        if targets[position][0].passage.id in proposed.get(item.passage.id, ())
        and len(keys) / len(item.shingles.keys() | targets[position][0].shingles.keys()) >= settings.verbatim_jaccard
    }
    floor = max(2, settings.min_source_share * len(item.shingles))
    sources = sorted(position for position, keys in shared.items() if len(keys) >= floor or position in near_copies)
    covered = set().union(*(shared[position] for position in sources)) if sources else set()
    containment = len(covered) / len(item.shingles) if item.shingles else 0.0
    if not sources or (containment < settings.verbatim_containment and not near_copies):
        return []
    pairs = []
    for position in sources:
        source, charge = targets[position]
        keys = shared[position]
        pairs.append(
            ReusePair(
                **_pair_ids(record, "verbatim", item.passage, source.passage, _flag_id(record, item.passage)),
                kind="verbatim",
                judgment=item.passage.span,
                indictment=source.passage.span,
                jaccard=round(len(keys) / len(item.shingles.keys() | source.shingles.keys()), 4),
                containment=round(len(keys) / len(item.shingles), 4),
                passage_containment=round(containment, 4),
                judgment_ranges=merge_ranges(r for key in keys for r in item.shingles[key]),
                indictment_ranges=merge_ranges(r for key in keys for r in source.shingles[key]),
                matches_charge_particulars=charge,
            )
        )
    return pairs


def _paraphrases(record: CaseRecord, items: Sequence[_Shingled], targets: list[tuple[_Shingled, bool]], ctx: AnalysisContext) -> list[ReusePair]:
    """For each passage, the most similar indictment passage above the cosine threshold that also shares content words."""
    settings = ctx.config.settings.reuse
    if not items or not targets:
        return []
    similarity = cosine_matrix(
        ctx.embedder.encode([item.passage.span.text for item in items]),
        ctx.embedder.encode([source.passage.span.text for source, _ in targets]),
    )
    source_words = [content_words(source.passage) for source, _ in targets]
    pairs = []
    for row, item in enumerate(items):
        words = content_words(item.passage)
        for col in np.argsort(-similarity[row], kind="stable"):
            cosine = float(similarity[row, col])
            if cosine < settings.paraphrase_cosine:
                break
            if len(words & source_words[col]) >= settings.paraphrase_min_shared_words:
                source, charge = targets[col]
                pairs.append(
                    ReusePair(
                        **_pair_ids(record, "paraphrase", item.passage, source.passage, _flag_id(record, item.passage)),
                        kind="paraphrase",
                        judgment=item.passage.span,
                        indictment=source.passage.span,
                        cosine=round(cosine, 4),
                        matches_charge_particulars=charge,
                    )
                )
                break
    return pairs


def find_pairs(record: CaseRecord, reasoning: Sequence[Passage], sources: Sequence[Source], ctx: AnalysisContext) -> tuple[ReusePair, ...]:
    settings = ctx.config.settings.reuse
    eligible = [s for s in (shingle(p, settings.shingle_size) for p in reasoning) if s.words >= settings.min_passage_words]
    targets = [(item, source.charge) for source in sources if (item := shingle(source.passage, settings.shingle_size)).words >= settings.min_passage_words]
    index: dict[Shingle, list[int]] = {}
    for position, (target, _) in enumerate(targets):
        for key in target.shingles:
            index.setdefault(key, []).append(position)
    proposed = lsh_candidates(eligible, [target for target, _ in targets], settings)
    verbatim = [pair for item in eligible for pair in _verbatim(record, item, targets, index, proposed, settings)]
    copied = {pair.judgment for pair in verbatim}
    paraphrased = _paraphrases(record, [item for item in eligible if item.passage.span not in copied], targets, ctx)
    return tuple(sorted(verbatim + paraphrased, key=lambda pair: (pair.judgment.start, pair.indictment.start)))


def flag_status(pairs: Sequence[ReusePair]) -> str:
    """The status of one judgment passage's matches: only the charge's wording, a copy, or a paraphrase."""
    if all(pair.matches_charge_particulars for pair in pairs):
        return "charge_wording"
    return "verbatim_reuse" if pairs[0].kind == "verbatim" else "paraphrase_reuse"


def by_passage(pairs: Sequence[ReusePair]) -> dict[str, list[ReusePair]]:
    """Flag id -> the pairs of that judgment passage, in document order."""
    grouped: dict[str, list[ReusePair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.flag_id or pair.id, []).append(pair)
    return grouped


def rescore(result: ReuseResult) -> ReuseResult:
    """Character counts and score from the pairs the result holds (also after some were dropped)."""
    verbatim, charge, paraphrase = [], [], 0
    for pairs in by_passage(result.pairs).values():
        status = flag_status(pairs)
        ranges = [(r.start, r.end) for pair in pairs for r in pair.judgment_ranges] or [(pairs[0].judgment.start, pairs[0].judgment.end)]
        if status == "charge_wording":
            charge += ranges
        elif status == "verbatim_reuse":
            verbatim += ranges
        else:
            paraphrase += len(pairs[0].judgment.text)
    verbatim_chars = sum(r.end - r.start for r in merge_ranges(verbatim))
    charge_chars = sum(r.end - r.start for r in merge_ranges(charge))
    score = None
    if result.score_note is None and result.reasoning_chars:
        score = min(1.0, (verbatim_chars + paraphrase) / result.reasoning_chars)
    return result.model_copy(
        update={"verbatim_chars": verbatim_chars, "paraphrase_chars": paraphrase, "charge_wording_chars": charge_chars, "score": score}
    )


def _passage_flag(record: CaseRecord, pairs: list[ReusePair], standard: StandardRef, config: RatioConfig) -> Flag:
    status = flag_status(pairs)
    percent = round(100 * (pairs[0].passage_containment or 0))
    if status == "paraphrase_reuse":
        message = render(config.messages, "reuse_paraphrase", similarity=f"{pairs[0].cosine:.2f}")
    elif status == "charge_wording":
        message = render(config.messages, "reuse_charge_wording", percent=percent, n=config.settings.reuse.shingle_size)
    else:
        message = render(config.messages, "reuse_verbatim", percent=percent, n=config.settings.reuse.shingle_size)
    evidence = (Evidence(role="judgment", span=pairs[0].judgment),) + tuple(Evidence(role="indictment", span=pair.indictment) for pair in pairs)
    return Flag(
        id=pairs[0].flag_id or stable_id(pairs[0].id, "flag"),
        case_id=record.case_id,
        module="reuse",
        standard_id=REUSE_STANDARD,
        standard_label=standard.label,
        status=status,
        message=message,
        evidence=evidence,
        citation=standard.citation,
        review_status=standard.review_status,
    )


def _ranked(argument: Argument, reasoning: Sequence[Passage], ctx: AnalysisContext) -> list[int]:
    """Reasoning passage indices, most similar to the argument first, each followed by the next passage."""
    vectors = ctx.embedder.encode([passage.span.text for passage in reasoning])
    similarity = cosine_matrix(vectors, ctx.embedder.encode([argument.text]))[:, 0]
    order: list[int] = []
    for i in sorted(range(len(reasoning)), key=lambda i: (-float(similarity[i]), i)):
        for j in (i, i + 1):  # a court's answer usually follows the sentence that names the argument
            if j < len(reasoning) and j not in order:
                order.append(j)
    return order


def _passages_to_check(argument: Argument, reasoning: Sequence[Passage], ctx: AnalysisContext) -> list[Passage]:
    """Every reasoning passage, or as many of the most relevant ones as fit one prompt and the limit."""
    settings = ctx.config.settings
    system = system_with_schema(ARGUMENT_SYSTEM, ArgumentReply)

    def fits(chosen: list[Passage]) -> bool:
        prompt = argument_user_prompt(argument, [(f"P{n}", p) for n, p in enumerate(chosen, start=1)])
        return estimated_tokens(system, prompt) + settings.llm.num_predict.argument_check <= settings.llm.num_ctx

    everything = list(reasoning)
    if len(everything) <= settings.reuse.argument_max_passages and fits(everything):
        return everything
    chosen: list[int] = []
    for i in _ranked(argument, reasoning, ctx):
        if len(chosen) == settings.reuse.argument_max_passages:
            break
        if fits([reasoning[j] for j in sorted([*chosen, i])]):
            chosen.append(i)
    return [reasoning[j] for j in sorted(chosen)]


def _responding(raw_items: list[str], count: int) -> tuple[set[int], bool]:
    """Passage numbers the model named, and whether any item was unreadable (quoted text, not an id)."""
    numbers: set[int] = set()
    unreadable = False
    for raw in raw_items:
        prefix = _ID_PREFIX.match(raw)
        tokens = [prefix.group(1)] if prefix else re.split(r",|\band\b", raw)
        for token in tokens:
            found = _PASSAGE_ID.fullmatch(token) if not prefix else re.match(r"(\d+)", token)
            if found and 1 <= int(found.group(1)) <= count:
                numbers.add(int(found.group(1)))
            elif token.strip().lower().strip(".") not in _NO_PASSAGE:
                unreadable = True
    return numbers, unreadable


def _unaddressed_flag(record: CaseRecord, argument: Argument, standard: StandardRef, message: str, note: str | None) -> Flag:
    return Flag(
        id=stable_id(record.case_id, "reuse", "unaddressed", argument.id),
        case_id=record.case_id,
        module="reuse",
        standard_id=ARGUMENT_STANDARD,
        standard_label=standard.label,
        status="unaddressed_argument",
        message=message,
        evidence=(Evidence(role="argument", span=argument.span),),
        citation=standard.citation,
        review_status=standard.review_status,
        model_note=note,
    )


def check_argument(
    record: CaseRecord, argument: Argument, reasoning: Sequence[Passage], standard: StandardRef, ctx: AnalysisContext
) -> tuple[ArgumentCheck, Flag | None]:
    messages = ctx.config.messages
    base = {"argument_id": argument.id, "argument": argument.span, "passages_total": len(reasoning)}
    if not reasoning:  # nothing in the judgment could answer it
        flag = _unaddressed_flag(record, argument, standard, render(messages, "reuse_unaddressed_no_reasoning"), None)
        return ArgumentCheck(**base, addressed=False, passages_checked=0, flag_id=flag.id), flag
    shown = _passages_to_check(argument, reasoning, ctx)
    numbered = [(f"P{n}", passage) for n, passage in enumerate(shown, start=1)]
    reply = ctx.llm.complete_json(
        system=ARGUMENT_SYSTEM, user=argument_user_prompt(argument, numbered), schema=ArgumentReply, purpose="argument_check"
    )
    note = filter_model_note(reply.note, messages.block_list)
    numbers, unreadable = _responding(reply.responding, len(shown))
    spans = tuple(shown[n - 1].span for n in sorted(numbers))
    if not spans and unreadable:  # an answer that names no passage is not evidence that none responds
        return ArgumentCheck(**base, addressed=False, checked=False, passages_checked=len(shown), model_note=note), None
    flag = None
    if not spans:
        message = render(messages, "reuse_unaddressed", count=len(shown), total=len(reasoning))
        flag = _unaddressed_flag(record, argument, standard, message, note)
    check = ArgumentCheck(
        **base, addressed=bool(spans), responding=spans, passages_checked=len(shown), model_note=note, flag_id=flag.id if flag else None
    )
    return check, flag


def _documents(record: CaseRecord, doc_type: str) -> tuple[Document, ...]:
    return record.documents_of_type(doc_type)


def _sources(record: CaseRecord, indictments: Sequence[Document], rules: PassageRules) -> list[Source]:
    """Every indictment's body text except its quoted provisions; the recital of the charge is marked."""
    sources = []
    for document in indictments:
        passages = record.passages_of(document.id)
        reasons = classify(passages, document.text, record.citations, rules)
        sources += [
            Source(passage, charge=reasons.get(passage.id) == "charge_recital")
            for passage in passages
            if passage.kind == "body" and reasons.get(passage.id) not in ("statute_quote", "header_or_signature")
        ]
    return sources


def run(record: CaseRecord, ctx: AnalysisContext) -> ReuseResult:
    judgments, indictments = _documents(record, "judgment"), _documents(record, "indictment")
    indictment_ids = tuple(doc.id for doc in indictments)
    first_indictment = indictment_ids[0] if indictment_ids else None
    if not judgments:
        return ReuseResult(judgment_doc_id=None, indictment_doc_id=first_indictment, indictment_doc_ids=indictment_ids, score_note="no judgment")
    judgment = judgments[0]  # the loader allows one judgment per case
    config = ctx.config
    rules = compile_rules(config.settings.reuse)
    passages = record.passages_of(judgment.id)
    reasons = classify(passages, judgment.text, record.citations, rules)
    reasoning = [p for p in passages if p.id in reasons and reasons[p.id] is None]
    excluded = tuple(
        ExcludedPassage(passage_id=p.id, span=p.span, reason=reason) for p in passages if (reason := reasons.get(p.id)) is not None
    )
    sources = _sources(record, indictments, rules)
    comparable = [s for s in sources if len(_WORD.findall(s.passage.span.text)) >= config.settings.reuse.min_passage_words]
    pairs = find_pairs(record, reasoning, sources, ctx) if comparable else ()
    reuse_standard = config.standard(REUSE_STANDARD)
    flags = [_passage_flag(record, group, reuse_standard, config) for group in by_passage(pairs).values()]
    checks = []
    argument_standard = config.standard(ARGUMENT_STANDARD)
    for argument in (a for a in record.arguments if a.party == "defense"):
        check, flag = check_argument(record, argument, reasoning, argument_standard, ctx)
        checks.append(check)
        flags.extend([flag] if flag else [])
    note = None if indictments and comparable else ("no indictment to compare with" if not indictments else "the indictment has no comparable text")
    if note is None and not reasoning:
        note = "the judgment has no passages of the court's own reasoning"
    result = ReuseResult(
        judgment_doc_id=judgment.id,
        indictment_doc_id=first_indictment,
        indictment_doc_ids=indictment_ids,
        score_note=note,
        reasoning_chars=sum(len(p.span.text) for p in reasoning),
        reasoning_passage_ids=tuple(p.id for p in reasoning),
        pairs=pairs,
        excluded=excluded,
        arguments=tuple(checks),
        flags=tuple(flags),
    )
    return rescore(result)
