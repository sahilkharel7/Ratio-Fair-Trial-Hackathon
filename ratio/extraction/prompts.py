"""Prompt and reply schema for LLM extraction of events and arguments.

Each chunk is sent once. The prompt lists the chunk's sentences that contain a written date
(found deterministically), numbered, so a small model classifies sentences instead of searching
free text. The reply schema is flat (Ollama compiles it to a grammar). Quotes are only used to
locate source text; dates are re-parsed in code. Any change to this text changes the cache key.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ModelEventType = Literal[
    "arrest",
    "first_appearance",
    "counsel_access",
    "charge",
    "detention_extension",
    "trial_start",
    "verdict",
]


class ModelEvent(BaseModel):
    sentence: int
    type: ModelEventType
    quote: str
    date_text: str
    iso_date: str
    actors: list[str]


class ModelArgument(BaseModel):
    party: Literal["defense", "prosecution"]
    quote: str


class ChunkExtraction(BaseModel):
    events: list[ModelEvent]
    arguments: list[ModelArgument]


EXTRACTION_SYSTEM = """You extract facts from court-monitoring documents for a legal review tool.
Copy text exactly. Never paraphrase, summarise, translate or infer.

EVENTS. You get numbered DATED SENTENCES. For each one, decide which of these events it states:
- arrest: the person is first deprived of liberty (arrested, detained, or taken for questioning).
- first_appearance: the person is brought in person before a JUDGE or a court for the first time.
  An appearance before a prosecutor or the police is NOT a first_appearance.
- counsel_access: the person first meets or consults a lawyer.
- charge: the indictment is filed or the person is formally charged.
- detention_extension: a court extends pretrial detention.
- trial_start: the trial opens.
- verdict: the court delivers its verdict or judgment.
Most sentences state none of these: skip them. A sentence may state two events; then return two.
For each event give: sentence (its number), type, quote (the exact words of that sentence that
state the event), date_text (the one date of this event, exactly as written in that sentence),
iso_date (YYYY-MM-DD, or "" if day, month or year is missing), actors (people or bodies named).

Examples (not from this case):
[1] "Police took Ana Holm to the station on 2 March 2024 and held her overnight." -> arrest, "2 March 2024"
[2] "On 3 March 2024 she was questioned by the prosecutor." -> nothing (a prosecutor is not a judge)
[3] "On 4 March 2024 she was brought before the investigating judge, who remanded her." -> first_appearance, "4 March 2024"
[4] "Her detention was extended on 1 April 2024 and again on 1 June 2024." -> two detention_extension events, "1 April 2024" and "1 June 2024"
[5] "She said she received the indictment on 10 April 2024." -> nothing
[6] "The hearing on 5 May 2024 began at 10:00." -> nothing (a routine hearing is not an event)

ARGUMENTS (monitoring notes only; return [] for other documents). From the full text, list each
contention a party makes about what the court must decide in its judgment (guilt, the law, or
whether evidence may be used), together with its reason. Give party ("defense" or "prosecution")
and quote (the exact words). Not arguments: requests decided during a hearing (adjournments,
witnesses), announcements such as an appeal, sentencing requests, reminders of facts, and the
court's own rulings.
Examples (not from this case):
"Counsel argued that the confession was taken without a lawyer and must be excluded." -> defense
"Counsel asked for an adjournment." -> not an argument
"The defence announced that it would appeal." -> not an argument

Do not describe anyone's character or motives. If nothing applies, return empty lists."""


def extraction_user_prompt(doc_type: str, title: str, dated_sentences: list[str], text: str | None) -> str:
    numbered = "\n".join(f"[{i}] {sentence}" for i, sentence in enumerate(dated_sentences, start=1)) or "(none)"
    parts = [f"Document type: {doc_type}", f"Title: {title}", "", "DATED SENTENCES:", numbered]
    if text is not None:
        parts += ["", "FULL TEXT (for arguments):", "<<<", text, ">>>"]
    return "\n".join(parts)
