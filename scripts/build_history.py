"""Generate the synthetic judicial-history cases for Module 4 from data/demo/history/seed.yaml.

Each case gets a short SYNTHETIC monitoring summary, a case.yaml, and a rulings.yaml whose quotes
are copied verbatim from the summary, plus a README.md with the seed matrix.

    python scripts/build_history.py
"""

from __future__ import annotations

import datetime as dt
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = REPO_ROOT / "data" / "demo" / "history"
MARKER = "SYNTHETIC: fictional case; all people, courts, country, language and laws are invented."


class _NoAliasDumper(yaml.SafeDumper):
    """Never write YAML anchors/aliases (the loader refuses them)."""

    def ignore_aliases(self, data: object) -> bool:
        return True


def _dump(data: dict) -> str:
    return yaml.dump(data, Dumper=_NoAliasDumper, sort_keys=False, allow_unicode=True)

DETENTION = {
    "ordered": ("detention_ordered", "the court ordered that {name} be detained pending trial"),
    "refused": ("detention_refused", "the court refused the request for pretrial detention and released {name} on bail"),
}
HEARINGS = {
    "public": ("hearing_public", "All hearings were held in public."),
    "closed": ("hearing_closed", "The court closed part of the trial to the public at the request of the prosecution."),
}
EVIDENCE = {
    "admitted": ("contested_evidence_admitted", "The court admitted the evidence that the defence had contested."),
    "excluded": ("contested_evidence_excluded", "The court excluded the evidence that the defence had contested."),
}
MOTION = {
    "denied": ("defense_motion_denied", "The court denied the defence motion to adjourn the trial."),
    "granted": ("defense_motion_granted", "The court granted the defence request to call an additional witness."),
}


def long_date(value: dt.date) -> str:
    return f"{value.day} {value.strftime('%B')} {value.year}"


def ruling(code: str, quote: str, judge: str, date: dt.date | None, value: float | None = None) -> dict:
    entry = {"code": code, "doc": "summary.txt", "quote": quote, "judge": judge, "date": date}
    if value is not None:
        entry["value"] = value
    return entry


def render_case(case: dict, defaults: dict) -> dict[str, str]:
    """File name -> text for one history case folder."""
    court = case.get("court", defaults["court"])
    charge_type = case.get("charge_type", defaults["charge_type"])
    name, judge = case["defendant"], case["judge"]
    first, verdict_date = case["first_appearance"], case["verdict_date"]

    code, clause = DETENTION[case["detention"]]
    detention_quote = clause.format(name=name)
    rulings = [ruling(code, detention_quote, judge, first)]
    body = [f"At the first appearance on {long_date(first)}, {detention_quote}."]
    for key, table in (("hearings", HEARINGS), ("evidence", EVIDENCE), ("motion", MOTION)):
        if case.get(key):
            code, sentence = table[case[key]]
            body.append(sentence)
            rulings.append(ruling(code, sentence, judge, None))
    if case["verdict"] == "conviction":
        months = case["sentence_months"]
        verdict_quote = f"the court convicted {name}"
        sentence_quote = f"a sentence of {months} months' imprisonment"
        closing = f"On {long_date(verdict_date)}, {verdict_quote} and imposed {sentence_quote}."
        rulings.append(ruling("verdict_conviction", verdict_quote, judge, verdict_date))
        rulings.append(ruling("sentence_months", sentence_quote, judge, verdict_date, float(months)))
    else:
        verdict_quote = f"the court acquitted {name}"
        closing = f"On {long_date(verdict_date)}, {verdict_quote}."
        rulings.append(ruling("verdict_acquittal", verdict_quote, judge, verdict_date))

    summary = "\n".join(
        [
            MARKER,
            "CASE SUMMARY FROM TRIAL MONITORING",
            f"Case: Republic of Calderra v. {name} ({court}, case no. {case['case_no']})",
            f"Presiding Judge: {judge}",
            f"Charge: {charge_type}",
            "",
            " ".join(body),
            "",
            closing,
            "",
        ]
    )
    manifest = {
        "synthetic": True,
        "case_id": case["id"],
        "title": f"Republic of Calderra v. {name}",
        "court": court,
        "charge_type": charge_type,
        "data_provenance": "synthetic",
        "documents": [{"path": "summary.txt", "type": "monitoring_note", "title": "Monitoring summary"}],
        "rulings": "rulings.yaml",
    }
    header = f"# {MARKER}\n"
    return {
        "summary.txt": summary,
        "case.yaml": header + _dump(manifest),
        "rulings.yaml": header + _dump({"synthetic": True, "rulings": rulings}),
    }


def render_readme(seed: dict) -> str:
    rows = [
        "| Case | Judge (as written) | Court | Detention at first appearance | Hearings | Contested evidence | Defence motion | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in seed["cases"]:
        court = case.get("court", seed["defaults"]["court"])
        rows.append(
            f"| {case['id']} | {case['judge']} | {court} | {case['detention']} | {case['hearings']} | "
            f"{case.get('evidence', '-')} | {case.get('motion', '-')} | {case['verdict']} |"
        )
    return "\n".join(
        [
            f"<!-- {MARKER} -->",
            "# Synthetic judicial history (Module 4 demo data)",
            "",
            f"**{MARKER}** Generated by `scripts/build_history.py` from `seed.yaml`; do not edit by hand.",
            "",
            "The demo case (`data/demo/case`) adds one more case for Judge Ilena Varda. Designed outcome on her",
            "profile (9 cases) against the other judges of the same court and charge type (10 cases):",
            "",
            "- Pretrial detention at first appearance: 9/9 vs 4/10: pattern that warrants review",
            "- Conviction at first instance: 8/9 vs 7/10: shown, not distinguishable from the baseline",
            "- Hearing closed to the public: 5/9 vs 2/10: shown, not distinguishable from the baseline",
            "- Contested evidence admitted: 3 cases: hidden (fewer than 5)",
            "- Every defence motion denied: 4 cases: hidden (fewer than 5)",
            "",
            "`hist-initials-01` names the judge only as \"I. Varda\" and goes to the manual confirmation list.",
            "`hist-port-elsin-01` is a judge with the same name at another court and is never merged.",
            "",
            *rows,
            "",
        ]
    )


def history_files(seed: dict) -> dict[str, str]:
    """Relative path -> text for everything the generator writes."""
    files = {"README.md": render_readme(seed)}
    for case in seed["cases"]:
        for name, text in render_case(case, seed["defaults"]).items():
            files[f"{case['id']}/{name}"] = text
    return files


def load_seed(path: Path = HISTORY_DIR / "seed.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    seed = load_seed()
    for case in seed["cases"]:
        shutil.rmtree(HISTORY_DIR / case["id"], ignore_errors=True)
    for relative, text in history_files(seed).items():
        path = HISTORY_DIR / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"wrote {len(seed['cases'])} synthetic history cases to {HISTORY_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
