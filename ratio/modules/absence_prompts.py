"""Prompt and reply schema for labelling observations against one rubric guarantee."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ratio.config import RubricItem
from ratio.schema import Observation


class ObservationLabel(BaseModel):
    observation: str
    label: Literal["supports", "contradicts", "unrelated"]
    indicator_id: str
    note: str = Field(max_length=120)  # bounded so a long list of labels never runs out of tokens


class LabelReply(BaseModel):
    labels: list[ObservationLabel]


LABEL_SYSTEM = """You help a lawyer check a trial-monitoring record against one fair-trial guarantee.
You get the guarantee, its indicators (each with an id), and numbered observations from the notes.
Each observation may show the sentence before it, only to explain what words like "the request"
refer to. Label EVERY observation (by its number, for example "O3"):
- supports: the observation itself describes the fact in one of the compliance indicators. Give
  that indicator's id.
- contradicts: the observation itself describes the fact in one of the violation indicators (for
  example, a request for an interpreter being refused). Give that indicator's id.
- unrelated: everything else. A refusal or a problem that concerns a different right is unrelated
  to this guarantee. Facts listed under NOT EVIDENCE are unrelated. Use indicator_id "".
Judge only what each observation says; never infer from what is missing. Do not describe anyone's
character or motives. note: at most one short sentence on why."""


def label_user_prompt(item: RubricItem, observations: list[tuple[str, Observation, str | None]]) -> str:
    compliance = [f"- {i.id}: {i.text}" for part in item.parts for i in part.compliance]
    violation = [f"- {i.id}: {i.text}" for part in item.parts for i in part.violation]
    context = [f"- {i.text}" for i in item.context] or ["- (none)"]
    numbered = []
    for obs_id, obs, previous in observations:
        numbered.append(f"{obs_id} ({obs.hearing_date.isoformat() if obs.hearing_date else 'undated'}): {obs.text}")
        if previous:
            numbered.append(f"    (sentence before it: {previous})")
    return "\n".join(
        [
            f"GUARANTEE: {item.provision}, {item.name}",
            f"TEXT: {item.text}",
            "",
            'COMPLIANCE INDICATORS (label "supports"):',
            *compliance,
            'VIOLATION INDICATORS (label "contradicts"):',
            *violation,
            'NOT EVIDENCE (label "unrelated"):',
            *context,
            "",
            "OBSERVATIONS:",
            *numbered,
        ]
    )
