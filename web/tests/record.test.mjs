import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const data = JSON.parse(readFileSync(new URL('../public/demo.json', import.meta.url), 'utf8'));
const records = [data.record, ...data.history];
const documents = new Map(records.flatMap(r => r.documents).map(d => [d.id, d]));

test('the public bundle contains only the synthetic demo and committed history', () => {
  assert.equal(data.synthetic, true);
  assert.equal(data.preview.mode, 'recorded-demo');
  for (const record of records) {
    assert.equal(record.meta.synthetic, true);
    assert.ok(record.case_id);
    for (const doc of record.documents) assert.equal(doc.synthetic, true);
  }
  assert.equal(data.record.documents.length, 6);
  assert.ok(data.judges.profiles.find(p => p.case_ids.includes(data.record.case_id)));
});

test('every exported evidence span matches the original source characters', () => {
  let count = 0;
  function check(value) {
    if (!value || typeof value !== 'object') return;
    if ('doc_id' in value && 'start' in value && 'end' in value && 'text' in value) {
      const doc = documents.get(value.doc_id);
      assert.ok(doc, `missing source ${value.doc_id}`);
      assert.equal(Array.from(doc.text).slice(value.start, value.end).join(''), value.text);
      count++;
    }
    Object.values(value).forEach(check);
  }
  check(data);
  assert.ok(count > 100, 'the full evidence record was exported');
});

test('legal statuses and analysis results survive the export', () => {
  const modules = [data.analysis.absence, data.analysis.clock, data.analysis.reuse];
  assert.equal(modules.flatMap(m => m.flags).length, 11);
  assert.equal(data.analysis.dropped_flags, 0);
  assert.deepEqual(data.analysis.clock.intervals.filter(i => i.status === 'exceeds_benchmark').map(i => i.benchmark_id), ['gc35_48h']);
  assert.equal(data.analysis.absence.assessments.length, 7);
  assert.equal(data.analysis.reuse.pairs.length, 5);
  assert.ok(data.comparison[data.analysis.reuse.judgment_doc_id].reasoning.includes('ratio-verbatim'));
});
