"""Public references retrieved locally; case comparisons use Ollama exclusively."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Literal

from pydantic import Field, create_model

from ratio.config import default_config
from ratio.llm import OllamaClient


def comparisons(library, research, case_id):
    record = library.cases.load_case(case_id)
    if record is None:
        raise KeyError("Case not found")
    fingerprint = hashlib.sha256(
        (record.model_dump_json() + str(research.path.stat().st_mtime_ns)).encode()
    ).hexdigest()
    with library.connect() as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS local_comparisons(case_id TEXT PRIMARY KEY,fingerprint TEXT,result_json TEXT)"
        )
        cached = c.execute(
            "SELECT result_json FROM local_comparisons WHERE case_id=? AND fingerprint=?",
            (case_id, fingerprint),
        ).fetchone()
    if cached:
        return json.loads(cached[0])
    focus = library.focus(case_id)
    span = (
        focus.prompts[0].span.model_dump(mode="json")
        if focus and focus.prompts
        else None
    )
    if span is None:
        for doc in record.documents:
            match = re.search(
                r"presumption\s+of\s+innocence|presumed\s+innocent|burden\s+of\s+proof",
                doc.text,
                re.IGNORECASE,
            )
            if match:
                start = max(0, match.start() - 200)
                end = min(len(doc.text), match.end() + 600)
                span = {
                    "doc_id": doc.id,
                    "start": start,
                    "end": end,
                    "text": doc.text[start:end],
                }
                break
    if span is None:
        return {
            "comparisons": [],
            "note": "No explicit presumption-of-innocence passage located. This is not a finding of compliance.",
        }
    found = research.search(
        "presumption of innocence", kind="ccpr_views", mode="meaning", limit=20
    )
    refs = []
    seen = set()
    for hit in found["hits"]:
        if hit["document"]["id"] in seen or hit["document"]["private"]:
            continue
        seen.add(hit["document"]["id"])
        refs.append(hit)
        if len(refs) == 3:
            break
    if not refs:
        return {
            "comparisons": [],
            "note": "No verified public reference passage retrieved; no comparison generated.",
        }
    ids = tuple(h["document"]["id"] for h in refs)
    section = create_model(
        "LocalComparison",
        reference_id=(Literal[ids], ...),
        similarity=(str, Field(max_length=320)),
        limits=(str, Field(max_length=320)),
    )
    schema = create_model(
        "LocalComparisons",
        comparisons=(list[section], Field(min_length=1, max_length=3)),
    )
    client = OllamaClient(
        default_config().settings.llm,
        model=os.environ.get("RATIO_MODEL", "qwen2.5:1.5b"),
    )
    answer = client.complete_json(
        system="Compare only these supplied passages. Identify speaker, procedural stage and common issue. Distinguish allegations from findings and report assessments. Explain material differences. Never predict success, declare a case violation, invent a citation, or treat ECHR as binding ICCPR law. Use only supplied reference IDs. Return JSON.",
        user=json.dumps(
            {
                "case_passage": span["text"][:1000],
                "references": [
                    {
                        "id": h["document"]["id"],
                        "title": h["document"]["title"],
                        "passage": h["text"][:1000],
                    }
                    for h in refs
                ],
            }
        ),
        schema=schema,
        purpose="local_reference_comparison",
    )
    mapped = {h["document"]["id"]: h for h in refs}
    items = []
    for item in answer.comparisons:
        hit = mapped[item.reference_id]
        items.append(
            {
                "case_span": span,
                "reference": hit["document"],
                "reference_span": {
                    "doc_id": "precedent:" + item.reference_id,
                    "start": hit["start"],
                    "end": hit["end"],
                    "text": hit["text"],
                },
                "similarity": item.similarity,
                "limits": item.limits,
            }
        )
    result = {
        "comparisons": items,
        "note": "Local Ollama reading comparison. Similar wording is not a legal finding; verify the full sources.",
    }
    with library.connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO local_comparisons VALUES(?,?,?)",
            (case_id, fingerprint, json.dumps(result)),
        )
    return result
