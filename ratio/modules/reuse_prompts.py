"""Prompt and reply schema for checking whether the judgment responds to a defence argument."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ratio.schema import Argument, Passage


class ArgumentReply(BaseModel):
    # The note comes first so the model states its reading before it lists passages.
    note: str = Field(max_length=200)
    responding: list[str]


ARGUMENT_SYSTEM = """You check whether a court's judgment responds to an argument the defence made at trial.
You get one defence argument and numbered passages from the court's own reasoning.

A passage RESPONDS to the argument only if it deals with the argument's point: it accepts or rejects the argument, rules on the objection or request the argument makes, or gives a reason that directly answers it.
These are NOT a response:
- repeating or summarising the argument;
- relying on, describing or weighing the evidence or facts the argument objects to, without dealing with the objection itself;
- general findings that never mention the argument's point.

First write a short note (at most 25 words) on whether any passage deals with the argument's point, then list the ids of the responding passages (for example ["P2"]). Use an empty list when no passage responds. Do not describe anyone's character or motives."""


def argument_user_prompt(argument: Argument, passages: list[tuple[str, Passage]]) -> str:
    lines = [f'Defence argument (trial monitoring note): "{argument.text}"', "", "Passages from the court's reasoning:"]
    lines += [f"{passage_id}: {passage.span.text}" for passage_id, passage in passages]
    return "\n".join(lines)
