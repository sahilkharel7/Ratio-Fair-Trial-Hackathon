"""Prompt and reply schema for the State's strongest reply to one finding (ratio/steelman.py)."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ratio.config import SteelmanGround
from ratio.schema import Flag


class ModelStateArgument(BaseModel):
    ground: str
    passage: str
    quote: str = Field(max_length=400)
    argument: str = Field(max_length=500)


class SteelmanReply(BaseModel):
    arguments: list[ModelStateArgument] = Field(max_length=6)


class GroundCheck(BaseModel):
    # The note comes first so the model reads the quote before it answers.
    note: str = Field(max_length=200)
    shows: bool


CHECK_SYSTEM = """You check one sentence of a court record against one requirement.
Answer whether the quoted words themselves show what the requirement says. Mentioning the same subject is not enough, and words that show the opposite do not count.
First write a short note (at most 20 words) on what the quote shows, then answer true or false. Do not describe anyone's character or motives."""


def check_user_prompt(quote: str, requires: str) -> str:
    return f'Quote from the record: "{" ".join(quote.split())}"\nRequirement: the quote shows {requires}.'


STEELMAN_SYSTEM = """You prepare the strongest good-faith reply the State could make to one finding of a trial-monitoring review, so that the reviewing lawyer can test the finding before relying on it.
You get the finding, the grounds the State may rely on (each with an id), and numbered passages from the case record.

Rules:
- Every argument must favour the State: it gives a reason why the finding may be wrong, overstated or justified. A point that confirms the finding is not a reply; leave it out.
- Argue only from the numbered passages. Never add facts, documents or events that the passages do not state.
- Each ground says what a passage must show. Use a ground only if one passage shows exactly that; a passage that merely mentions the same subject is not enough.
- Each argument uses one ground id from the list, cites exactly one passage id (for example "E3"), and copies an exact quote of at most 40 words from that passage, the words that show what the ground requires.
- Write each argument in at most 50 words, as the State would put it. Do not describe anyone's character or motives.
- Give at most {max_arguments} arguments, the strongest first, each on a different ground. Fewer is better than weak: if no passage shows what any ground requires, return an empty list."""


def steelman_system(max_arguments: int) -> str:
    return STEELMAN_SYSTEM.format(max_arguments=max_arguments)


def steelman_user_prompt(flag: Flag, grounds: Sequence[SteelmanGround], passages: Sequence[tuple[str, str, str, bool]]) -> str:
    """``passages``: (id, document title, exact text, part of the finding's evidence) in the order shown."""
    lines = [f"Finding ({flag.standard_label}): {flag.message}", "", "The finding rests on:"]
    lines += [f'- "{" ".join(item.span.text.split())}"' for item in flag.evidence[:4]]
    lines += ["", "Grounds the State may rely on:"]
    lines += [
        f"- {ground.id}: {ground.label}. Use only if a passage shows {ground.requires}."
        + (" The passage must not be part of the finding." if ground.independent else "")
        for ground in grounds
    ]
    lines += ["", "Record passages (those marked [part of the finding] are the finding's own evidence):"]
    lines += [
        f"{passage_id} ({title}){' [part of the finding]' if own else ''}: {' '.join(text.split())}"
        for passage_id, title, text, own in passages
    ]
    return "\n".join(lines)
