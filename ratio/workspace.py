"""Shared local case-workspace operations used by React's loopback API and Streamlit."""
from __future__ import annotations

from pathlib import Path

from ratio.extraction.build import load_case
from ratio.library import LibraryStore
from ratio.outcomes import registry, summary
from ratio.paths import DEMO_CASE_DIR, DEMO_DIR
from ratio.schema import CaseRecord
from ratio.store import CaseStore


def seed_collection(library: LibraryStore):
    """Explicit demo action. Never overwrite a case that is already stored."""
    folders=[DEMO_CASE_DIR,*sorted((DEMO_DIR/'focused').iterdir())]
    for folder in folders:
        base=load_case(folder)
        record=library.cases.load_case(base.case_id)
        if record is None:
            record=base
            library.cases.save_case(record)
        library.register(record)


def case_payload(library: LibraryStore, case_id: str) -> dict | None:
    record=library.cases.load_case(case_id)
    if record is None: return None
    focus=library.focus(case_id)
    serialized={**record.model_dump(mode='json'),'case_id':record.case_id}
    from urllib.parse import quote
    with library.connect() as conn:
        mapping={r['path']:r['intake_id'] for r in conn.execute('SELECT path,intake_id FROM matter_files WHERE case_id=?',(case_id,))}
    for doc in serialized['documents']:
        if doc['path'] in mapping:
            doc['download_url']=f"/api/cases/{quote(case_id,safe='')}/files/{mapping[doc['path']]}"
    return {'record':serialized,
            'focus':focus.model_dump(mode='json'),'decisions':library.decisions(case_id),
            'notes':library.notes(case_id),
            'analysis_available':library.cases.load_analysis(case_id) is not None}


def focus_report(library: LibraryStore, case_id: str) -> str:
    payload=case_payload(library,case_id)
    if payload is None: raise KeyError(case_id)
    from ratio.report import md
    focus=payload['focus'];record=library.cases.load_case(case_id)
    lines=[f"# Case review worksheet: {md(record.meta.title)}",'',
           '> Synthetic example.' if record.meta.synthetic else f"> Public record: {md(record.meta.source_note or '')}",'',
           f"Court: {md(record.meta.court)}",f"Defendant: {md(focus['defendant'])}",f"Charge: {md(focus['charge'])}",'',
           '## Penalty statements','']
    for item in focus['penalties']:
        lines.extend([f"### {item['stage'].capitalize()}: {md(item['label'])}",md(item['span']['doc_id']),'',
                      *['> '+line for line in md(item['span']['text']).splitlines()],''])
    lines.extend(['## Presumption of innocence',focus['standard'],focus['screening_note'],''])
    if not focus['prompts']:
        lines.extend(['No explicit screening passage was identified. This does not establish compliance; verify the record with the monitor.',''])
    decisions=library.decisions(case_id);current={}
    for decision in decisions:
        if decision['decision']=='reopened': current.pop(decision['prompt_id'],None)
        else: current[decision['prompt_id']]=decision
    for prompt in focus['prompts']:
        lines.extend([f"### {md(prompt['title'])}",md(prompt['question']),md(prompt['span']['doc_id']),'',
                      *['> '+line for line in md(prompt['span']['text']).splitlines()],''])
        decision=current.get(prompt['id'])
        if decision: lines.extend([f"Reviewer assessment: {decision['decision']}",f"Reason: {md(decision['reason'])}",''])
    lines.extend(['## Decision history',''])
    for d in decisions: lines.append(f"- {d['created_at']} · {d['decision']} · {md(d['reason'])}")
    lines.extend(['','## Working notes',''])
    for note in library.notes(case_id): lines.extend([f"{note['created_at']}: {md(note['note'])}",''])
    lines.extend(['','## International review context','Historical source-coded outcomes are research material, not a prediction of success or proof that a remedy was implemented.'])
    return '\n'.join(lines)+'\n'
