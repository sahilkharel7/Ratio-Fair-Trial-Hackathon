import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {penaltyView,outcomeSummary,latestAssessments} from '../src/library-client.mjs';
const bundle=JSON.parse(readFileSync(new URL('../public/demo.json',import.meta.url),'utf8'));

test('the collection is three distinct synthetic matters with source-backed stakes',()=>{
  assert.equal(bundle.collection.cases.length,3);
  assert.equal(bundle.collection.records.length,3);
  const venn=bundle.collection.records.find(p=>p.record.case_id==='venn-2025');
  assert.equal(penaltyView(venn.focus,'requested').label,'5 years imprisonment');
  assert.equal(penaltyView(venn.focus,'imposed').label,'4 years imprisonment');
  assert.equal(penaltyView(venn.focus,'statutory').label,'2–6 years imprisonment');
  const sorin=bundle.collection.records.find(p=>p.record.case_id==='sorin-2025');
  assert.equal(sorin.focus.prompts[0].pattern,'burden_shift');
  assert.ok(bundle.collection.records.every(p=>p.record.meta.synthetic));
});
test('the browser never converts a narrow or procedural cohort into case-specific prospects',()=>{
  const summary=outcomeSummary(bundle.outcomes.records);
  assert.equal(summary.merits,5);assert.equal(summary.favourable,4);assert.equal(summary.inadmissible,2);
  assert.equal(summary.rate,.8);assert.equal(summary.prediction,null);
  const specific=outcomeSummary(bundle.outcomes.records,'burden_shift');
  assert.equal(specific.merits,1);assert.equal(specific.rate,null);assert.equal(specific.prediction,null);
});
test('reopened focused assessments preserve their historical entry without remaining current',()=>{
  const history=[{prompt_id:'p',decision:'supported',reason:'Source confirms the concern'},{prompt_id:'p',decision:'reopened',reason:''}];
  assert.equal(latestAssessments(history).p,undefined);assert.equal(history.length,2);
});
test('conflicting terms are presented as an ambiguity, not an arbitrary exposure figure',()=>{
  const focus={penalties:[{stage:'requested',kind:'imprisonment',label:'3 years imprisonment',max_months:36},{stage:'requested',kind:'imprisonment',label:'5 years imprisonment',max_months:60}]};
  assert.equal(penaltyView(focus,'requested').ambiguous,true);
  assert.equal(penaltyView(focus,'requested').statement,null);
});
