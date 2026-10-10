"""A narrow, source-led Article 14(2) review worksheet, not a finding of a violation.

Charge and penalty statements are kept separate: a prosecution request is not the
statutory range or the sentence imposed. Every extracted item carries original text.
The English screening patterns identify passages for a lawyer to check; silence
and reused prosecution wording are never turned into a presumption-of-innocence verdict.
"""
from __future__ import annotations

import re
import math
from typing import Literal

from pydantic import Field

from ratio.schema import CaseRecord, Frozen, SourceSpan, stable_id

STANDARD = "ICCPR Article 14(2) · General Comment No. 32, paragraph 30"
NUMBER_WORDS = {word: n for n, word in enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"))}
TENS = {'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90}
ONES = '|'.join(list(NUMBER_WORDS)[1:10])
SMALL = '|'.join(sorted(list(NUMBER_WORDS)[:20],key=len,reverse=True))
COMPOUND = r'(?:'+ '|'.join(TENS)+r')(?:[- ](?:'+ONES+r'))?'
HUNDREDS = r'(?:'+ONES+r')\s+hundred(?:\s+(?:and\s+)?(?:'+COMPOUND+'|'+SMALL+r'))?'
NUMBER = r'(?:\d+(?:\.\d+)?|'+HUNDREDS+'|'+COMPOUND+'|'+SMALL+r')'
DURATION = re.compile(rf"(?<![\w-])(?P<lo>{NUMBER})(?:\s*(?:to|[-–])\s*(?P<hi>{NUMBER}))?(?:\s+|-)(?P<unit>years?|months?|days?)\b", re.I)


class PenaltyStatement(Frozen):
    stage: Literal["requested", "statutory", "imposed"]
    label: str
    kind: Literal["imprisonment", "life_imprisonment", "death_penalty", "fine", "other"]
    min_months: float | None = None
    max_months: float | None = None
    span: SourceSpan


class InnocencePrompt(Frozen):
    id: str
    pattern: Literal["burden_shift", "official_guilt_statement", "prejudicial_presentation"]
    title: str
    question: str
    span: SourceSpan
    status: Literal["needs_review"] = "needs_review"


class FocusRecord(Frozen):
    case_id: str
    defendant: str
    charge: str
    charge_span: SourceSpan | None = None
    penalties: tuple[PenaltyStatement, ...] = ()
    prompts: tuple[InnocencePrompt, ...] = ()
    standard: str = STANDARD
    screening_note: str = "English-language screening only. A passage is a prompt for legal review, not a confirmed violation."


def _passages(record: CaseRecord):
    for doc in record.documents:
        # Paragraphs preserve every original character; sentence breaks avoid assigning
        # a statutory range to a separate prosecution request in the same paragraph.
        for paragraph in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", doc.text):
            text = paragraph.group()
            for part in re.finditer(r".+?(?:[.!?](?=\s+[A-Z]|$)|$)", text, re.S):
                start = paragraph.start() + part.start()
                raw = part.group()
                offset = len(raw) - len(raw.lstrip())
                exact = raw.strip()
                if exact:
                    yield doc, SourceSpan(doc_id=doc.id, start=start+offset, end=start+offset+len(exact), text=exact)


def _number(value: str) -> float:
    if value[0].isdigit(): return float(value)
    words=value.lower().replace('-',' ').split()
    if 'hundred' in words:
        i=words.index('hundred')
        return NUMBER_WORDS[words[0]]*100+sum(NUMBER_WORDS.get(w,TENS.get(w,0)) for w in words[i+1:] if w!='and')
    return float(sum(NUMBER_WORDS.get(w,TENS.get(w,0)) for w in words))


def _penalty(doc_type: str, span: SourceSpan) -> PenaltyStatement | None:
    text, lower = span.text, span.text.lower()
    if not re.search(r"imprison|prison|sentenc|death penalty|fine", lower):
        return None
    if re.search(r"\bprosecut\w*\b.{0,100}\b(request\w*|seek\w*|ask\w*)\b", lower):
        stage = "requested"
    elif re.search(r"shall be punished|punishable|maximum penalty|maximum sentence|statutory|penalty of", lower):
        stage = "statutory"
    elif doc_type == "judgment" and re.search(r"\b(sentences?|sentenced)\b.{0,100}\b(imprison|prison|death|fine)", lower):
        stage = "imposed"
    else:
        return None
    if re.search(r"not (?:request|seek|sentenc)|no (?:prison|custodial) sentence|no (?:term of )?imprisonment|without imprisonment|acquitt", lower):
        return None
    if "death penalty" in lower or "sentenced to death" in lower:
        return PenaltyStatement(stage=stage, label="Death penalty", kind="death_penalty", span=span)
    if re.search(r"\blife(?:[- ]long)?\b.{0,25}\b(imprisonment|prison)\b|imprisonment for life", lower):
        return PenaltyStatement(stage=stage, label="Life imprisonment", kind="life_imprisonment", span=span)
    if "imprison" not in lower and "prison" not in lower:
        if re.search(r"\bfine[sd]?\b",lower):
            return PenaltyStatement(stage=stage, label="Fine — see the recorded request", kind="fine", span=span)
        return PenaltyStatement(stage=stage, label="Other penalty — see the recorded statement", kind="other", span=span)
    imprisonment = list(re.finditer(r"imprison\w*|\bprison\b", lower))
    durations = list(DURATION.finditer(text))
    duration = min(durations, key=lambda d: min(abs(d.start()-p.start()) for p in imprisonment), default=None)
    if duration is None:
        return PenaltyStatement(stage=stage, label="Imprisonment · term not specified", kind="imprisonment", span=span)
    low, high = _number(duration['lo']), _number(duration['hi'] or duration['lo'])
    if not(math.isfinite(low) and math.isfinite(high)) or not 0<=low<=high<=10000:
        return PenaltyStatement(stage=stage,label='Imprisonment · verify the recorded term',kind='imprisonment',span=span)
    unit = duration['unit'].lower()
    if unit.startswith('day'):
        # Day-level custody is shown literally, not converted using an invented month length.
        months_low = months_high = None
    else:
        months_low, months_high = (low*12, high*12) if unit.startswith('year') else (low, high)
    value = f"{low:g}–{high:g}" if low != high else f"{low:g}"
    before = text[max(0,duration.start()-35):duration.start()].lower()
    if low == high and re.search(r"up to|not more than|at most|maximum(?: of)?", before):
        months_low, value = None, f"Up to {low:g}"
    elif low == high and re.search(r"not less than|at least|minimum(?: of)?", before):
        months_high, value = None, f"At least {low:g}"
    return PenaltyStatement(stage=stage, label=f"{value} {unit} imprisonment", kind="imprisonment",
                            min_months=months_low, max_months=months_high, span=span)


def screen(record: CaseRecord, *, defendant: str | None = None) -> FocusRecord:
    penalties, prompts, charge_span = [], [], None
    for doc, span in _passages(record):
        text = span.text.lower()
        if charge_span is None and doc.type in {"indictment", "judgment"} and re.search(r"(?:accused|defendant|charged person).{0,70}\bcharged with\b", text) and not re.search(r"not charged|no longer charged|previously charged|charges? (?:was|were) (?:dropped|withdrawn|dismissed)",text):
            charge_span = span
        if doc.type in {"indictment", "judgment", "transcript", "monitoring_note"}:
            penalty = _penalty(doc.type, span)
            if penalty and not any(p.stage == penalty.stage and p.label == penalty.label for p in penalties):
                penalties.append(penalty)
        if doc.type not in {"judgment", "transcript", "monitoring_note"}:
            continue
        if re.search(r"must not|not required|no requirement|cannot require|wrong to require|should not|unlawful to require", text):
            continue
        pattern = None
        if re.search(r"(?:accused|defendant).{0,60}(?:must|has to|is required to).{0,35}(?:prove|establish|demonstrate).{0,60}(?:innocen|not guilty|did not)", text):
            pattern, title, question = "burden_shift", "Possible reversal of the burden of proof", "Did the court require the defendant to establish innocence, rather than require the prosecution to prove guilt?"
        elif re.search(r"before (?:the )?trial|pre[- ]trial|before (?:a|the) verdict", text) and re.search(r"police|minister|official|prosecutor", text) and re.search(r"publicly|press conference|television|media|news", text) and re.search(r"\bguilty\b|\bcriminal\b", text):
            pattern, title, question = "official_guilt_statement", "Public statement of guilt before judgment", "Who made the statement, when was it made, and did it affirm guilt before it was proved?"
        elif re.search(r"(?:accused|defendant).{0,100}(?:shackled|handcuffed|kept in a cage|placed in a cage)", text):
            pattern, title, question = "prejudicial_presentation", "Presentation of the defendant in court", "Was the restraint necessary in the circumstances, and did the presentation suggest that the defendant was a dangerous criminal?"
        if pattern:
            prompts.append(InnocencePrompt(id=stable_id(record.case_id,pattern,span.doc_id,span.start,span.text),
                                           pattern=pattern,title=title,question=question,span=span))
    name = defendant or (record.meta.title.rsplit(" v. ",1)[-1] if " v. " in record.meta.title else record.meta.title)
    charge=record.meta.charge_type
    if charge_span:
        charge=re.search(r'\bcharged with\s+(.+)',charge_span.text,re.I|re.S).group(1).strip().rstrip('.')
    return FocusRecord(case_id=record.case_id,defendant=name,charge=charge,
                       charge_span=charge_span,penalties=tuple(penalties),prompts=tuple(prompts))


def penalty_summary(focus: FocusRecord, stage: str) -> dict:
    items=[p for p in focus.penalties if p.stage==stage]
    specific=[p for p in items if p.max_months is not None or p.min_months is not None or p.kind!='imprisonment']
    labels={p.label for p in specific}
    if len(labels)>1:
        return {'label':'Several recorded terms — check the source sequence','statement':None,'ambiguous':True}
    selected=specific[0] if specific else items[0] if items else None
    return {'label':selected.label if selected else 'Not stated in the record',
            'statement':selected.model_dump(mode='json') if selected else None,'ambiguous':False}
