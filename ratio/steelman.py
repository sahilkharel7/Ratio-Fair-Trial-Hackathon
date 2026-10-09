"""The steelman-the-state agent: the strongest reply the State could make to each adverse finding,
written by the local model from the record, so the reviewing lawyer can test a finding before
relying on it. It is never a finding, and its wording is shown only as model-generated and unverified.

For every finding whose status the State would contest (steelman.yaml: adverse_statuses), and never
for a judge pattern:
1. Retrieval, in code: the record passages most similar to the finding (note sentences and the
   sentences of the court documents) are numbered E1, E2, ... in document order.
2. One model call: the model argues the State's side from those passages only, on grounds from the
   reviewed catalogue for the finding's standard, each argument citing one passage and quoting it.
3. Checks, in code: an argument is dropped if its ground is not in the catalogue, its passage was not
   shown, its quote is not found in that passage (exactly, or with only whitespace and quotation marks
   differing), an independent ground quotes the finding's own evidence, or nothing is left of it after
   the block list (hard rule 3). Every argument kept rests on an exact span of the record (hard rule 2).
4. A second, short model call per remaining argument: does the quote itself show what the ground
   requires? An argument the check rejects is dropped. Grounds no kept argument rests on are listed
   as not supported by the record.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ratio.context import AnalysisContext
from ratio.embeddings import cosine_matrix
from ratio.llm import LLMResponseError, PromptTooLong
from ratio.messages import filter_model_note
from ratio.provenance import DocResolver, span_is_valid
from ratio.results import StateArgument, StateReply, SteelmanResult
from ratio.schema import CaseRecord, Flag, SourceSpan
from ratio.config import SteelmanGround
from ratio.steelman_prompts import CHECK_SYSTEM, GroundCheck, SteelmanReply, check_user_prompt, steelman_system, steelman_user_prompt

EXCLUDED_MODULES = frozenset({"judges"})  # arguing a judge's side would be a statement about a person
MIN_QUOTE_CHARS = 12
_PASSAGE_ID = re.compile(r"^\s*E(\d+)\s*$", re.IGNORECASE)
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


@dataclass(frozen=True)
class _Candidate:
    span: SourceSpan
    title: str
    order: tuple[int, int]  # document position, then offset: the order passages are shown in


def candidates(record: CaseRecord) -> list[_Candidate]:
    """Every sentence of the record the State could argue from: note sentences and body passages."""
    position = {doc.id: index for index, doc in enumerate(record.documents)}
    titles = {doc.id: doc.title for doc in record.documents}
    spans = [o.span for o in record.observations] + [p.span for p in record.passages if p.kind == "body"]
    unique = {(s.doc_id, s.start, s.end): s for s in spans}
    found = [_Candidate(s, titles[s.doc_id], (position[s.doc_id], s.start)) for s in unique.values()]
    return sorted(found, key=lambda c: c.order)


def locate(quote: str, span: SourceSpan) -> SourceSpan | None:
    """The quote's exact span inside one passage: verbatim, or with only whitespace and quotation
    marks differing. None if it is not there or is too short to mean anything."""
    cleaned = " ".join(quote.strip().strip('"“”').split())
    if len(cleaned) < MIN_QUOTE_CHARS:
        return None
    start = span.text.find(cleaned)
    if start < 0:
        words = [re.escape(word.translate(_QUOTES)) for word in cleaned.split()]
        pattern = re.compile(r"\s+".join(words))
        match = pattern.search(span.text.translate(_QUOTES))
        if match is None:
            return None
        start, end = match.start(), match.end()
    else:
        end = start + len(cleaned)
    return SourceSpan(doc_id=span.doc_id, start=span.start + start, end=span.start + end, text=span.text[start:end])


def _shown(flag: Flag, pool: list[_Candidate], vectors: np.ndarray, ctx: AnalysisContext) -> list[_Candidate]:
    query = " ".join([flag.message, *(item.span.text for item in flag.evidence[:4])])
    similarity = cosine_matrix(vectors, ctx.embedder.encode([query]))[:, 0]
    top = sorted(range(len(pool)), key=lambda i: (-float(similarity[i]), i))[: ctx.config.settings.steelman.top_k_passages]
    return sorted((pool[i] for i in top), key=lambda c: c.order)


def _own(span: SourceSpan, flag: Flag) -> bool:
    return any(span.overlaps(item.span) for item in flag.evidence)


def _verified(argument: StateArgument, ground: SteelmanGround, ctx: AnalysisContext) -> bool:
    answer = ctx.llm.complete_json(
        system=CHECK_SYSTEM, user=check_user_prompt(argument.span.text, ground.requires), schema=GroundCheck, purpose="steelman_check"
    )
    return answer.shows


def _check(reply: SteelmanReply, flag: Flag, shown: list[_Candidate], grounds: dict[str, SteelmanGround], ctx: AnalysisContext) -> tuple[list[StateArgument], list[str]]:
    kept: list[StateArgument] = []
    dropped: list[str] = []
    used: set[str] = set()
    for item in reply.arguments:
        number = _PASSAGE_ID.match(item.passage)
        if item.ground not in grounds:
            dropped.append(f"ground {item.ground!r} is not one of the grounds offered")
        elif item.ground in used:
            dropped.append(f"ground {item.ground!r} argued twice")
        elif number is None or not 1 <= int(number.group(1)) <= len(shown):
            dropped.append(f"passage {item.passage!r} was not shown to the model")
        elif (span := locate(item.quote, shown[int(number.group(1)) - 1].span)) is None:
            dropped.append(f"quote not found in passage {item.passage}: {item.quote[:60]!r}")
        elif grounds[item.ground].independent and _own(span, flag):
            dropped.append(f"ground {item.ground!r} must rest on other evidence than the finding's own")
        elif (argument := filter_model_note(item.argument, ctx.config.messages.block_list)) is None:
            dropped.append(f"argument on {item.ground!r} was empty")
        else:
            candidate = StateArgument(ground_id=item.ground, argument=argument, span=span, passage_label=f"E{number.group(1)}")
            try:
                verified = _verified(candidate, grounds[item.ground], ctx)
            except (LLMResponseError, PromptTooLong):
                verified = False
            if verified:
                kept.append(candidate)
                used.add(item.ground)
            else:
                dropped.append(f"the check found that the quote for {item.ground!r} does not show what the ground requires")
    limit = ctx.config.settings.steelman.max_arguments
    dropped += [f"argument on {a.ground_id!r} beyond the first {limit}" for a in kept[limit:]]
    return kept[:limit], dropped


def reply_to(flag: Flag, pool: list[_Candidate], vectors: np.ndarray, ctx: AnalysisContext) -> StateReply:
    config = ctx.config
    grounds = config.steelman.for_standard(flag.standard_id)
    shown = _shown(flag, pool, vectors, ctx)
    numbered = [(f"E{n}", c.title, c.span.text, _own(c.span, flag)) for n, c in enumerate(shown, start=1)]
    base = {"flag_id": flag.id, "standard_id": flag.standard_id, "passages_shown": len(shown)}
    try:
        reply = ctx.llm.complete_json(
            system=steelman_system(config.settings.steelman.max_arguments),
            user=steelman_user_prompt(flag, grounds, numbered),
            schema=SteelmanReply,
            purpose="steelman",
        )
    except (LLMResponseError, PromptTooLong) as exc:  # one unusable answer must not lose the case
        return StateReply(**base, checked=False, note=f"The model gave no usable answer: {exc}", unsupported_grounds=tuple(g.id for g in grounds))
    kept, dropped = _check(reply, flag, shown, {g.id: g for g in grounds}, ctx)
    argued = {a.ground_id for a in kept}
    return StateReply(**base, arguments=tuple(kept), dropped=tuple(dropped), unsupported_grounds=tuple(g.id for g in grounds if g.id not in argued))


def run(record: CaseRecord, flags: Sequence[Flag], ctx: AnalysisContext) -> SteelmanResult:
    adverse = [f for f in flags if f.status in ctx.config.steelman.adverse_statuses and f.module not in EXCLUDED_MODULES]
    pool = candidates(record)
    if not adverse or not pool:
        return SteelmanResult(model=ctx.llm.model)
    vectors = ctx.embedder.encode([c.span.text for c in pool])
    return SteelmanResult(replies=tuple(reply_to(flag, pool, vectors, ctx) for flag in adverse), model=ctx.llm.model)


def check_steelman(result: SteelmanResult, resolve: DocResolver) -> SteelmanResult:
    """Hard rule 2 for the replies: an argument whose quote is not found in its document is dropped."""
    replies = []
    for reply in result.replies:
        bad = [a for a in reply.arguments if not span_is_valid(a.span, resolve)]
        if bad:
            reply = reply.model_copy(
                update={
                    "arguments": tuple(a for a in reply.arguments if a not in bad),
                    "dropped": reply.dropped + tuple(f"quote of {a.ground_id!r} failed the source check" for a in bad),
                    "unsupported_grounds": reply.unsupported_grounds + tuple(dict.fromkeys(a.ground_id for a in bad)),
                }
            )
        replies.append(reply)
    return result.model_copy(update={"replies": tuple(replies)})
