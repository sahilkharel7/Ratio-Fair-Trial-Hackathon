"""Claim-specific, source-coded historical outcomes. No individual success prediction.

This small registry extends the precedent reading workflow with adjudication stage.
It does not change Similar cases' matching or mix reference decisions into the case
store. An inadmissibility decision is not a no-violation merits determination.
"""

from __future__ import annotations

import json
from pathlib import Path

from ratio.paths import CONFIG_DIR

REGISTRY = CONFIG_DIR / "innocence_outcomes.json"


def registry(path: Path = REGISTRY) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary(
    records: list[dict],
    *,
    forum="UN Human Rights Committee",
    pattern: str | None = None,
) -> dict:
    chosen = [
        r
        for r in records
        if r["forum"] == forum
        and r["standard"] == "ICCPR Article 14(2)"
        and (not pattern or r["pattern"] == pattern)
    ]
    eligible = [
        r
        for r in chosen
        if r["phase"] == "merits"
        and r["outcome"] in {"violation_found", "no_violation"}
    ]
    favourable = sum(r["outcome"] == "violation_found" for r in eligible)
    return {
        "forum": forum,
        "pattern": pattern,
        "records": chosen,
        "merits_decisions": len(eligible),
        "violation_found": favourable,
        "no_violation": len(eligible) - favourable,
        "inadmissible": sum(r["outcome"] == "inadmissible" for r in chosen),
        "observed_merits_rate": favourable / len(eligible)
        if len(eligible) >= 5
        else None,
        "success_probability": None,
        "small_sample": len(eligible) < 5,
        "note": "Selected research examples only. This fraction is not a prediction, not a representative success rate, and does not measure release, acquittal or implementation.",
    }
