"""Export only the committed synthetic demo for the public web preview.

Run with the local model installed: python scripts/export_web_demo.py
This never reads the user's SQLite store, uploads, or runtime model cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratio import netguard
from ratio.config import default_config
from ratio.display import marked_html, reuse_marks
from ratio.extraction.build import load_case
from ratio.history import load_history, load_alias_decisions
from ratio.modules.reuse import by_passage
from ratio.paths import ALIAS_DECISIONS, DEMO_CASE_DIR
from ratio.pipeline import analyze_judges, demo_llm, process


def main() -> None:
    netguard.install()
    cfg = default_config()
    record, analysis, _ = process(load_case(DEMO_CASE_DIR), demo_llm(cfg), cfg)
    records = [record, *load_history()]
    assert all(r.meta.synthetic and all(d.synthetic for d in r.documents) for r in records)
    judges = analyze_judges(records, {record.case_id: analysis}, load_alias_decisions(ALIAS_DECISIONS), cfg)
    assert all(p.synthetic for p in judges.profiles)
    reuse = analysis.reuse
    matches = by_passage(reuse.pairs)
    numbers = {flag_id: str(n) for n, flag_id in enumerate(matches, start=1)}
    comparison = {}
    for doc_id in [reuse.judgment_doc_id, *reuse.indictment_doc_ids]:
        doc = record.document(doc_id)
        marks = reuse_marks(reuse, numbers, doc_id, cfg.messages.label)
        spans = [p.span for p in record.passages if p.id in set(reuse.reasoning_passage_ids)] if doc_id == reuse.judgment_doc_id else []
        spans += [s for pair in reuse.pairs for s in (pair.judgment, pair.indictment) if s.doc_id == doc_id]
        first, last = min(s.start for s in spans), max(s.end for s in spans)
        headings = [p.span for p in record.passages_of(doc_id) if p.kind == 'heading']
        start = max((h.start for h in headings if h.start <= first), default=0)
        end = min((h.start for h in headings if h.start >= last), default=len(doc.text))
        comparison[doc_id] = {
            'whole': marked_html(doc.text, marks),
            'reasoning': marked_html(doc.text[start:end], marks, offset=start),
        }
    payload = {
        'synthetic': True,
        'preview': {'mode': 'recorded-demo', 'source_branch': 'frontend-improvements', 'legal_analysis': 'local-only'},
        'record': {**record.model_dump(mode='json'), 'case_id': record.case_id},
        'analysis': analysis.model_dump(mode='json'),
        'judges': judges.model_dump(mode='json'),
        'history': [{**r.model_dump(mode='json'), 'case_id': r.case_id} for r in records[1:]],
        'messages': cfg.messages.model_dump(mode='json'),
        'benchmarks': cfg.benchmarks.model_dump(mode='json'),
        'comparison': comparison,
    }
    target = ROOT / 'web' / 'public' / 'demo.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Exported {len(record.documents)} documents, {len(analysis.all_flags())} findings, and {len(judges.profiles)} synthetic judge profiles.')


if __name__ == '__main__':
    main()
