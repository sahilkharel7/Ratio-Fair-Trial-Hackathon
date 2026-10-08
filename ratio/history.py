"""The judicial history behind Module 4: the seeded synthetic cases and a person's decisions on the
registry's manual confirmation list. History cases are read by the deterministic loader only (their
rulings are coded by hand), so loading them needs no model."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ratio.extraction.build import load_case
from ratio.extraction.loader import MANIFEST_NAME, LoaderError, safe_yaml
from ratio.paths import ALIAS_DECISIONS, HISTORY_DIR, PUBLIC_ALIAS_DECISIONS
from ratio.schema import AliasDecision, CaseRecord, Frozen


class AliasDecisionsFile(Frozen):
    synthetic: bool
    decisions: tuple[AliasDecision, ...] = ()


def load_history(folder: Path = HISTORY_DIR) -> tuple[CaseRecord, ...]:
    """Every case folder (one holding case.yaml) directly inside ``folder``, in name order."""
    return tuple(load_case(path) for path in sorted(Path(folder).iterdir()) if (path / MANIFEST_NAME).is_file())


def decisions_file(synthetic: bool) -> Path:
    """Where a person records decisions on names of that kind of data."""
    return ALIAS_DECISIONS if synthetic else PUBLIC_ALIAS_DECISIONS


def load_all_alias_decisions() -> tuple[AliasDecision, ...]:
    return load_alias_decisions(ALIAS_DECISIONS) + load_alias_decisions(PUBLIC_ALIAS_DECISIONS)


def load_alias_decisions(path: Path = ALIAS_DECISIONS) -> tuple[AliasDecision, ...]:
    """The recorded decisions, each marked with the file's kind of data; none when the file does not exist."""
    path = Path(path)
    if not path.is_file():
        return ()
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LoaderError(f"{path.name} must be saved as UTF-8 text") from exc
    data = safe_yaml(text, path.name)
    try:
        recorded = AliasDecisionsFile.model_validate(data)
    except ValidationError as exc:
        raise LoaderError(f"{path.name} is invalid: {exc}") from exc
    return tuple(decision.model_copy(update={"synthetic": recorded.synthetic}) for decision in recorded.decisions)
