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

from unidecode import unidecode
from ratio import netguard
from ratio.config import default_config
from ratio.display import Mark, marked_html, reuse_marks
from ratio.extraction.build import load_case
from ratio.history import load_history, load_alias_decisions
from ratio.modules.reuse import by_passage
from ratio.feedback import case_findings
from ratio.jurisprudence import Reference, for_finding, for_follow_up
from ratio.report import draft_report
from ratio.paths import ALIAS_DECISIONS, DEMO_CASE_DIR
from ratio.pipeline import analyze_judges, demo_llm, process


def reference_payload(ref):
    return {**ref.entry.model_dump(mode="json"), "citation": ref.citation, "shown": ref.shown,
            "source": ref.source.model_dump(mode="json")}


def renewal_comparisons(record, renewal, messages):
    """Presentation-only marks: comparison scores and statuses remain pipeline outputs."""
    def start_of_grounds(order):
        starts = [p.span.start for p in record.passages_of(order.doc_id) if p.id in order.grounds_passage_ids]
        if not starts:
            return 0
        return max((p.span.start for p in record.passages_of(order.doc_id)
                    if p.kind == 'heading' and p.span.start <= min(starts)), default=min(starts))

    def excluded(order):
        return [Mark(e.span.start, e.span.end, 'excluded',
                     '' if e.reason == 'header_or_signature' else messages.label(e.reason)) for e in order.excluded]

    def matches(order, doc_id, side):
        marks = []
        for pair in order.pairs:
            span = pair.later if side == 'later' else pair.earlier
            ranges = pair.later_ranges if side == 'later' else pair.earlier_ranges
            if span.doc_id == doc_id:
                marks += [Mark(r.start, r.end, pair.kind) for r in ranges] or [Mark(span.start, span.end, pair.kind)]
        return marks

    orders = {order.doc_id: order for order in renewal.orders}
    result = {}
    for order in renewal.orders:
        sides = {}
        for doc_id in [order.doc_id, *dict.fromkeys(pair.earlier.doc_id for pair in order.pairs)]:
            own = orders[doc_id]
            marks = excluded(own) + matches(order, doc_id, 'later' if doc_id == order.doc_id else 'earlier')
            doc = record.document(doc_id)
            start = start_of_grounds(own)
            sides[doc_id] = {'whole': marked_html(doc.text, marks),
                             'grounds': marked_html(doc.text[start:], marks, offset=start)}
        result[order.doc_id] = sides
    return result


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
        'preview': {'mode': 'recorded-demo', 'source_branch': 'codex/combined-legal-workspace', 'analysis_base': 'a4bc765', 'legal_analysis': 'local-only'},
        'record': {**record.model_dump(mode='json'), 'case_id': record.case_id},
        'analysis': analysis.model_dump(mode='json'),
        'judges': judges.model_dump(mode='json'),
        'history': [{**r.model_dump(mode='json'), 'case_id': r.case_id} for r in records[1:]],
        'messages': cfg.messages.model_dump(mode='json'),
        'benchmarks': cfg.benchmarks.model_dump(mode='json'),
        'comparison': comparison,
        'review_findings': [flag.model_dump(mode='json') for flag in case_findings(analysis, judges)],
        'jurisprudence': {
            'checked': cfg.jurisprudence.checked,
            'entries': [reference_payload(Reference(e, cfg.jurisprudence.sources[e.source])) for e in cfg.jurisprudence.entries],
            'by_finding': {f.id: [reference_payload(r) for r in for_finding(f, cfg.jurisprudence)] for f in case_findings(analysis, judges)},
            'by_follow_up': {f.rubric_id: [reference_payload(r) for r in for_follow_up(f, cfg.jurisprudence)] for f in analysis.absence.follow_ups},
        },
        'steelman_catalogue': cfg.steelman.model_dump(mode='json'),
        # Presentation guard for English review wording: match Python's Latin/Greek/Cyrillic
        # transliteration, in addition to NFKC and removal of invisible characters.
        'review_transliteration': {chr(n): unidecode(chr(n)) for lo, hi in ((128, 1328), (0x1C80, 0x1C90), (0x1E00, 0x2000), (0xA640, 0xA6A0))
                                  for n in range(lo, hi) if unidecode(chr(n)) != chr(n)},
        'renewal_comparison': renewal_comparisons(record, analysis.renewal, cfg.messages),
        'report_markdown': draft_report(record, analysis, judges, cfg, history=records[1:]),
    }
    target = ROOT / 'web' / 'public' / 'demo.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Exported {len(record.documents)} documents, {len(analysis.all_flags())} findings, and {len(judges.profiles)} synthetic judge profiles.')


if __name__ == '__main__':
    main()
