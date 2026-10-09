import React, {createContext, useContext, useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './styles.css';

const Source = createContext(null);
const navigation = [['/', 'Case overview'], ['/coverage', 'Rights coverage'], ['/timeline', 'Procedural timeline'], ['/reuse', 'Reasoning comparison'], ['/judges', 'Judicial history']];
// Match Python's displayed percentages, including ties rounded to the even digit.
const percent = value => {if(value==null)return 'n/a';const n=value*100,low=Math.floor(n);return `${n-low===.5?(low%2===0?low:low+1):Math.round(n)}%`;};
const date = value => value ? new Date(value).toLocaleDateString('en-GB', {day:'numeric', month:'short', year:'numeric', timeZone:'UTC'}) : 'Undated';
const points = (text, start, end) => Array.from(text).slice(start, end).join('');
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const flags = analysis => [analysis.absence, analysis.clock, analysis.reuse].flatMap(module => module?.flags || []);

function Badge({status, messages}) {
  const tone = ['evidence_of_violation','exceeds_benchmark'].includes(status) ? 'adverse' : ['evidence_of_compliance','within_benchmark','addressed_argument'].includes(status) ? 'positive' : /review|unaddressed|verbatim/.test(status) ? 'review' : 'neutral';
  return <span className={`badge ${tone}`}>{messages.labels[status] || status}</span>;
}
function Metrics({items}) {return <div className="metrics">{items.map(([label,value,note,tone]) => <div key={label} className={`metric ${tone || ''}`}><div>{label}</div><strong>{value}</strong><small>{note}</small></div>)}</div>;}
function Section({title, description, children}) {return <section className="section"><h2>{title}</h2>{description && <p className="section-caption">{description}</p>}{children}</section>;}
function Notice({messages}) {return <div className="data-notice">{messages.notes.synthetic_banner}</div>;}
function Header({data,title,description,crossCase=false}) {return <><p className="eyebrow">{crossCase ? 'Cross-case analysis' : 'Case analysis'}</p><h1>{title}</h1><Notice messages={data.messages}/>{!crossCase && <p className="caption">{data.record.meta.title} · {data.record.meta.court}</p>}{description && <p className="intro">{description}</p>}</>;}
function Evidence({items=[], heading=''}) {
  const {documents, openSource, messages} = useContext(Source);
  return <div className="evidence-list">{items.map((item,index) => <div className="evidence" key={`${item.span.doc_id}-${item.span.start}-${index}`}><blockquote><cite>{messages.labels[`role_${item.role}`]} · {documents[item.span.doc_id]?.title}</cite>“{item.span.text}”</blockquote><button onClick={() => openSource(item.span, heading)}>Source</button></div>)}</div>;
}
function Workstream({title,value,description,to,navigate}) {return <article className="panel workstream"><div><h3>{title}</h3><strong>{value}</strong></div><p>{description}</p><a href={to} onClick={e=>{e.preventDefault();navigate(to);}}>Review {title.toLowerCase()} <span aria-hidden="true">→</span></a></article>;}
function download(data) {
  const blob = new Blob([JSON.stringify({synthetic:true,record:data.record,analysis:data.analysis},null,2)], {type:'application/json'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href=url;link.download='ratio-synthetic-demo-review.json';link.click();URL.revokeObjectURL(url);
}

function Overview({data,navigate,onAbout}) {
  const {record,analysis,messages} = data;
  const [query,setQuery] = useState('');
  const {openSource,openDocument} = useContext(Source);
  const docs = record.documents.filter(d => !query.trim() || `${d.title}\n${d.text}`.toLowerCase().includes(query.toLowerCase()));
  const statuses = analysis.absence.assessments.map(a=>a.status);
  const profile = data.judges.profiles.find(p=>p.case_ids.includes(record.case_id));
  return <><h1>Case overview</h1><Notice messages={messages}/>
    <div className="matter"><p className="eyebrow">Current matter</p><h2>{record.meta.title}</h2><dl><div><dt>Court</dt><dd>{record.meta.court}</dd></div><div><dt>Charge</dt><dd>{record.meta.charge_type}</dd></div><div><dt>Presiding judge</dt><dd>{record.meta.presiding_judge}</dd></div></dl></div>
    <Metrics items={[
      ['Source documents',record.documents.length,`${record.observations.length} monitoring note sentences`],
      ['Findings to review',flags(analysis).length,'Each linked to its exact source'],
      ['Monitor follow-ups',analysis.absence.follow_ups.length,'Gaps in the monitoring record','review'],
      ['Procedural events',analysis.clock.timeline.filter(e=>e.type!=='hearing'&&e.date).length,'Distinct dated events, excluding hearings'],
    ]}/>
    <Section title="Review workstreams" description="Move from the overview to the evidence behind each result."><div className="workstreams">
      <Workstream title="Rights coverage" value={statuses.length} description={`${statuses.filter(s=>s==='evidence_of_violation').length} with evidence of violation · ${statuses.filter(s=>s==='evidence_of_compliance').length} with evidence of compliance · ${statuses.filter(s=>s==='no_evidence').length} follow-up questions`} to="/coverage" navigate={navigate}/>
      <Workstream title="Procedural timeline" value={analysis.clock.intervals.filter(i=>i.status==='exceeds_benchmark').length} description="Intervals longer than a confirmed benchmark. Other intervals are measured for legal review." to="/timeline" navigate={navigate}/>
      <Workstream title="Reasoning comparison" value={percent(analysis.reuse.score)} description={`Of the court's reasoning traceable to the indictment. ${plural(analysis.reuse.arguments.filter(a=>a.flag_id).length,'defence argument')} without a response.`} to="/reuse" navigate={navigate}/>
      <Workstream title="Judicial history" value={profile?.case_ids.length || 0} description={`${profile?.display_name} · ${plural(profile?.case_ids.length || 0,'case')} · ${plural(profile?.flags.length || 0,'pattern')} that warrants review.`} to="/judges" navigate={navigate}/>
    </div></Section>
    <details className="panel"><summary>Analysis record and method</summary><p>{flags(analysis).length} findings, each linked to the exact text it rests on. {analysis.dropped_flags} dropped because their source text could not be found.</p><p className="caption">Model: {analysis.llm_model}. Recorded analysis generated locally; no model runs on this website.</p><button onClick={()=>download(data)}>Download synthetic review JSON</button></details>
    <Section title="Source documents" description="Search the record or open a document to read its full text.">
      <label className="search-label" htmlFor="source-search">Search source documents</label><input id="source-search" type="search" placeholder="Search titles or exact words in the record…" value={query} onChange={e=>setQuery(e.target.value)}/>
      {query.trim() && <p className="caption">{docs.length} of {record.documents.length} documents match. Search uses exact words.</p>}
      {docs.length ? <div className="table-wrap"><table><thead><tr><th>Document</th><th>Type</th><th>Hearing date</th><th><span className="sr-only">Action</span></th></tr></thead><tbody>{docs.map(doc=><tr key={doc.id}><td>{doc.title}</td><td className="muted">{doc.type.replaceAll('_',' ')}</td><td>{doc.date?date(doc.date):'—'}</td><td><button aria-label={`Open ${doc.title}`} onClick={()=>{const start=query.trim()?doc.text.toLowerCase().indexOf(query.toLowerCase()):-1;if(start>=0){const text=doc.text.slice(start,start+query.length);openSource({doc_id:doc.id,start:Array.from(doc.text.slice(0,start)).length,end:Array.from(doc.text.slice(0,start+query.length)).length,text},'Search result');}else openDocument(doc);}}>Open</button></td></tr>)}</tbody></table></div>:<p className="empty">No source documents match this search. Try a shorter phrase.</p>}
    </Section><div className="preview-explanation"><div><h3>Working with your own case?</h3><p>Use the local Python app to import records and run analysis on your computer.</p></div><button onClick={onAbout}>About this preview</button></div>
  </>;
}

function Coverage({data}) {
  const {absence} = data.analysis;
  const [selected,setSelected] = useState(absence.assessments[0].rubric_id);
  const a=absence.assessments.find(a=>a.rubric_id===selected), flag=absence.flags.find(f=>f.id===a.flag_id);
  const statuses=absence.assessments.map(a=>a.status);
  return <><Header data={data} title="Rights coverage" description="Each guarantee is marked as evidence of compliance, evidence of violation, or no evidence. Missing evidence becomes a follow-up question for the monitor."/>
    <Metrics items={[["Evidence of violation",statuses.filter(s=>s==='evidence_of_violation').length,'Review the contrary monitoring notes','review'],['Evidence of compliance',statuses.filter(s=>s==='evidence_of_compliance').length,'Supported by the monitoring record','positive'],['Monitor follow-ups',statuses.filter(s=>s==='no_evidence').length,'No evidence recorded; follow up with the monitor']]}/>
    <Section title="Guarantee review" description="Select a guarantee to inspect its evidence and the parts covered by the record."><div className="split coverage-split"><div className="guarantees" aria-label="Fair-trial guarantees">{absence.assessments.map(item=><button className={`guarantee panel ${item.rubric_id===selected?'selected':''}`} aria-pressed={item.rubric_id===selected} onClick={()=>setSelected(item.rubric_id)} key={item.rubric_id}><strong>{item.provision}</strong><span>{item.name}</span><Badge messages={data.messages} status={item.status}/></button>)}</div>
    <article className="panel sticky-detail" aria-live="polite"><h3>{a.provision}: {a.name}</h3><div className="badges"><Badge messages={data.messages} status={a.status}/><Badge messages={data.messages} status={a.review_status}/></div><p className="caption">{a.citation}</p>
    {flag?<><p>{flag.message}</p><Evidence items={flag.evidence} heading={flag.standard_label}/>{flag.model_note&&<details><summary>Model note (unverified)</summary><p>{flag.model_note}</p></details>}</>:<><div className="info"><strong>Follow-up for the monitor</strong><p>{a.follow_up?.question}</p></div><h4>Not covered by the notes</h4><ul>{a.parts.filter(p=>p.required&&p.status==='no_evidence').map(p=><li key={p.part_id}>{p.label}{p.hearings_missing.length?` · ${p.hearings_missing.map(date).join(', ')}`:''}</li>)}</ul>{a.unlabelled_notes>0&&<p>{a.unlabelled_notes} shortlisted notes were not labelled by the model.</p>}<p className="caption">Possibly relevant, not counted as evidence:</p><Evidence items={a.follow_up?.context} heading="Context"/></>}
    <h4>Coverage by part</h4><div className="table-wrap"><table><thead><tr><th>Part</th><th>Status</th></tr></thead><tbody>{a.parts.map(p=><tr key={p.part_id}><td>{p.label}{p.required&&<small className="required">Required</small>}</td><td><Badge messages={data.messages} status={p.status}/></td></tr>)}</tbody></table></div></article></div></Section></>;
}

function TimelineChart({data}) {
  const events=data.analysis.clock.timeline.filter(e=>e.date), start=Math.min(...events.map(e=>Date.parse(e.date))), end=Math.max(...events.map(e=>Date.parse(e.date)));
  const x=value=>250+((typeof value==='number'?value:Date.parse(value))-start)/(end-start)*570;
  const names=data.messages.event_labels, rows=Object.keys(names).filter(type=>events.some(e=>e.type===type));
  const height=rows.length*40+80;
  return <div className="chart-scroll"><svg viewBox={`0 0 850 ${height}`} role="img" aria-label="Procedural events from February to July 2025">
    {Array.from({length:6},(_,i)=>{const time=start+(end-start)*i/5;return <g key={i}><line x1={x(time)} x2={x(time)} y1="15" y2={height-45} stroke="#e8edf1"/><text x={x(time)} y={height-18} textAnchor="middle" className="chart-label">{date(time).replace(' 2025','')}</text></g>;})}
    {rows.map((type,i)=><g key={type}><text x="232" y={37+i*40} textAnchor="end" className="chart-label">{names[type]}</text>{events.filter(e=>e.type===type).map(e=><circle key={e.id} cx={x(e.date)} cy={32+i*40} r="5" fill="#28587d"><title>{names[type]} · {date(e.date)} · {data.messages.labels[`timeline_${e.state}`]}</title></circle>)}</g>)}
  </svg></div>;
}
function Timeline({data}) {
  const clock=data.analysis.clock;
  return <><Header data={data} title="Procedural timeline" description="Only a confirmed benchmark can be marked as exceeded. Other intervals are cited and measured for legal review."/>
    <Section title="Chronology" description="Distinct events from the source record. Hover over a point for its date and status."><div className="panel"><TimelineChart data={data}/></div></Section>
    <Section title="Benchmark review" description="Confirmed benchmarks appear first. Each interval includes the source dates and its citation."><div className="stack">{[...clock.intervals].sort((a,b)=>(b.status==='exceeds_benchmark')-(a.status==='exceeds_benchmark')).map(i=>{
      const benchmark=data.benchmarks.benchmarks.find(b=>b.id===i.benchmark_id), flag=clock.flags.find(f=>f.id===i.flag_id);
      const duration=i.min_hours==null?'Cannot be measured':i.min_hours===i.max_hours?`${i.min_hours} hours`:`Between ${i.min_hours/24} and ${i.max_hours/24} days (${i.min_hours}–${i.max_hours} hours; dates are day-level)`;
      const citation=flag?.citation||`${benchmark.provision}; ${data.benchmarks.sources[benchmark.citation.instrument].symbol}, para. ${benchmark.citation.paras}`;
      return <article key={i.id} className="panel"><h3>{data.messages.event_labels[benchmark.from_event]} → {data.messages.event_labels[benchmark.to_event]} <span className="caption">· {benchmark.provision}</span></h3><div className="badges"><Badge messages={data.messages} status={i.status}/><Badge messages={data.messages} status={i.review_status}/></div><p>{flag?.message||`${duration}. ${i.threshold_hours==null?'No confirmed threshold; measured and shown only.':''}`}</p><p className="caption">{citation}</p><p className="caption">{benchmark.note}</p><details open={i.status==='exceeds_benchmark'}><summary>Sources ({i.evidence.length})</summary><Evidence items={i.evidence} heading={benchmark.name}/></details></article>;
    })}</div></Section>
    <Section title="Event record" description="Open an event to read every source mention and any date conflicts."><div className="stack">{[...clock.timeline].sort((a,b)=>Date.parse(a.date)-Date.parse(b.date)).map(e=><details className="panel" key={e.id}><summary>{date(e.date)} · {data.messages.event_labels[e.type]} <span className="caption">· {plural(e.mentions.length,'source')}</span></summary>{e.review_reasons.length>0&&<p>{e.review_reasons.join('; ')}</p>}<Evidence items={e.mentions} heading={data.messages.event_labels[e.type]}/></details>)}</div></Section></>;
}

function Reasoning({data}) {
  const reuse=data.analysis.reuse;
  const [whole,setWhole]=useState(false);
  const matchFlags=reuse.flags.filter(f=>f.status!=='unaddressed_argument');
  return <><Header data={data} title="Reasoning comparison" description="Quoted statutes, the recited charge, and party positions are excluded first. Highlighted text matches the indictment; its legal significance remains for the reviewing lawyer."/>
    <Metrics items={[["Traceable reasoning",percent(reuse.score),'Share matching indictment text'],['Verbatim',percent(reuse.verbatim_chars/reuse.reasoning_chars),'Word-for-word matches'],['Close paraphrase',percent(reuse.paraphrase_chars/reuse.reasoning_chars),'Similar wording in the reasoning'],['Unanswered arguments',reuse.arguments.filter(a=>a.flag_id).length,'Defence arguments with no response found']]}/>
    <Section title="Document comparison" description="Percentages measure characters of the court's reasoning, with quotations excluded. Match numbers connect both documents.">
    <div className="segmented" role="group" aria-label="Document view"><button aria-pressed={!whole} className={!whole?'active':''} onClick={()=>setWhole(false)}>Court's reasoning</button><button aria-pressed={whole} className={whole?'active':''} onClick={()=>setWhole(true)}>Whole documents</button></div>
    <p className="legend"><mark className="ratio-verbatim">Verbatim</mark><mark className="ratio-paraphrase">Close paraphrase</mark><span className="muted">Greyed text is quotation or not reasoning.</span></p>
    <div className="split">{[reuse.judgment_doc_id,reuse.indictment_doc_ids[0]].map(docId=><div key={docId}><h3>{data.record.documents.find(d=>d.id===docId).title}</h3><div className="panel document-panel" dangerouslySetInnerHTML={{__html:data.comparison[docId][whole?'whole':'reasoning']}}/></div>)}</div>
    </Section><Section title="Matched passages" description="Read each match in context on either side of the comparison."><div className="stack">{matchFlags.map((f,index)=><article className="panel" key={f.id}><div className="row-heading"><h3>{index+1}. Matched passage</h3><Badge messages={data.messages} status={f.status}/></div><p>{f.message}</p><Evidence items={f.evidence} heading={`Matched passage ${index+1}`}/></article>)}</div></Section>
    <Section title="Defence arguments from the notes"><div className="stack">{reuse.arguments.map(a=><article className="panel" key={a.argument_id}><Badge messages={data.messages} status={!a.checked?'unchecked_argument':a.addressed?'addressed_argument':'unaddressed_argument'}/><Evidence items={[{role:'argument',span:a.argument}]} heading="Defence argument"/>{a.flag_id&&<p>{reuse.flags.find(f=>f.id===a.flag_id)?.message}</p>}{a.responding.length>0&&<><p className="caption">Answered in the reasoning ({a.passages_checked} of {a.passages_total} passages checked):</p><Evidence items={a.responding.map(span=>({role:'judgment',span}))} heading="Response in the judgment"/></>}{a.model_note&&<details><summary>Model note (unverified)</summary><p>{a.model_note}</p></details>}</article>)}</div></Section></>;
}

function Judges({data}) {
  const profiles=data.judges.profiles;
  const [selected,setSelected]=useState(profiles.find(p=>p.case_ids.includes(data.record.case_id)).judge_id);
  const profile=profiles.find(p=>p.judge_id===selected);
  const rate=r=>`${r.k} of ${r.n} cases, ${percent(r.rate)} (${percent(r.confidence)} interval ${percent(r.ci_low)} to ${percent(r.ci_high)})`;
  return <><Header data={data} title="Judicial history" crossCase/><div className="warning">{data.messages.notes.selection_bias}</div>
    <label htmlFor="judge">Judge</label><select id="judge" value={selected} onChange={e=>setSelected(e.target.value)}>{profiles.map(p=><option key={p.judge_id} value={p.judge_id}>{p.display_name} · {p.court}</option>)}</select>
    <div className="matter"><p className="eyebrow">Selected profile</p><h2>{profile.display_name}</h2><p className="caption">{profile.court} · {profile.charge_types.join('; ')} · {plural(profile.case_ids.length,'case')}</p><p>Indicators compared: <strong>{profile.k_compared}</strong>. Rates count each case once.</p></div>
    <Section title="Coded ruling indicators" description="Compared with other judges of the same court and charge type. Small samples are not shown."><div className="stack">{profile.indicators.map(i=><article className="panel" key={`${i.rate_id}-${i.charge_type}`}><div className="row-heading"><h3>{i.label}</h3>{i.pattern&&<Badge messages={data.messages} status="pattern_warrants_review"/>}{!i.shown&&<Badge messages={data.messages} status="hidden_indicator"/>}</div>{i.shown&&<><div className="rate-comparison"><div><h4>This judge</h4><strong>{percent(i.judge.rate)}</strong><p>{rate(i.judge)}</p></div><div><h4>Baseline · {plural(i.baseline_judges,'other judge')}</h4><strong>{percent(i.baseline.rate)}</strong><p>{rate(i.baseline)}</p></div></div><p className="caption">Difference from baseline: {i.difference_ci.map(v=>`${Math.round(v*100)>0?'+':''}${Math.round(v*100)}`).join(' to ')} percentage points ({(i.difference_confidence*100).toFixed(1)}% interval).</p></>}<p>{i.message}</p>{i.outcomes.length>0&&<details><summary>Counted cases and coded rulings ({i.outcomes.length})</summary>{i.outcomes.map(o=><div key={o.case_id}><h4>{o.title} · {o.counted?'Counted in numerator':'Not counted in numerator'}</h4><Evidence items={[o.ruling]} heading={`${o.title}: coded ruling`}/></div>)}</details>}</article>)}</div></Section>
    <Section title="Descriptive observations"><div className="stack">{profile.descriptive.map(d=><article className="panel" key={`${d.code}-${d.charge_type}`}><h3>{d.label}</h3><p>{d.message}</p>{d.shown?<><p>{plural(d.cases,'case')} · Range {Math.min(...d.values)} to {Math.max(...d.values)}</p><details><summary>Values and coded rulings ({d.evidence.length})</summary><Evidence items={d.evidence} heading={d.label}/></details></>:<Badge messages={data.messages} status="hidden_indicator"/>}</article>)}</div></Section>
    <details className="panel"><summary>Names waiting for manual confirmation ({data.judges.manual_confirmations.length})</summary><p className="caption">These rulings do not count in any judge's rates or baseline until a person confirms the name.</p><div className="table-wrap"><table><thead><tr><th>Name as written</th><th>Court</th><th>Possible match</th><th>Reason</th></tr></thead><tbody>{data.judges.manual_confirmations.map(c=><tr key={`${c.case_id}-${c.raw_name}`}><td>{c.raw_name}</td><td>{c.court}</td><td>{c.candidate_name||'None'}</td><td>{c.reason}</td></tr>)}</tbody></table></div></details>
  </>;
}

function SourceDialog({value,onClose,messages}) {
  const ref=useRef(null);
  useEffect(()=>{if(value)ref.current.showModal();else ref.current.close();},[value]);
  const doc=value?.document, span=value?.span;
  const valid=doc&&(!span||points(doc.text,span.start,span.end)===span.text);
  const begin=span?Math.max(0,span.start-700):0, end=span?Math.min(Array.from(doc?.text||'').length,span.end+700):0;
  return <dialog ref={ref} className="source-dialog" onCancel={onClose} onClick={e=>{if(e.target===ref.current)onClose();}}><div className="dialog-heading"><h2>{span?'Source text':'Source document'}</h2><button onClick={onClose} aria-label="Close source viewer">×</button></div>{value&&<><Notice messages={messages}/><h3>{value.heading||doc?.title}</h3>{!valid?<p className="warning">This passage was not found in its source document, so it is not shown.</p>:<><p className="caption">{doc.title}{span?` · line ${points(doc.text,0,span.start).split('\n').length} · characters ${span.start} to ${span.end} · checked against the source: exact match`:' · Original source text'}</p><div className="source-text">{span?<>{begin>0?'…\n':''}{points(doc.text,begin,span.start)}<mark>{span.text}</mark>{points(doc.text,span.end,end)}{end<Array.from(doc.text).length?'\n…':''}</>:doc.text}</div>{span&&<details><summary>Whole document</summary><div className="source-text">{points(doc.text,0,span.start)}<mark>{span.text}</mark>{points(doc.text,span.end)}</div></details>}</>}</>}</dialog>;
}
function AboutDialog({open,onClose}) {
  const ref=useRef(null);
  useEffect(()=>{if(open)ref.current.showModal();else ref.current.close();},[open]);
  return <dialog className="about-dialog" ref={ref} onCancel={onClose}><div className="dialog-heading"><h2>About this preview</h2><button aria-label="Close preview information" onClick={onClose}>×</button></div><p>This is an interactive preview of Ratio's synthetic demo, exported from the local Python analysis pipeline. All people, courts, country, language, and laws in the record are fictional.</p><p>Rights coverage, measured intervals, reasoning matches, and judicial history use the real recorded demo results. Source buttons open the original synthetic documents and verify exact text before highlighting it.</p><h3>New-case analysis runs locally</h3><p>This website does not accept uploads or run an AI model. To analyze public or synthetic case records, run the Python application on your computer:</p><pre>streamlit run app/main.py</pre><p>Legal conclusions remain with the reviewing lawyer.</p><a href="https://github.com/sahilkharel7/Ratio-Fair-Trial-Hackathon/tree/codex/vercel-preview" target="_blank" rel="noreferrer">View the deployment branch →</a></dialog>;
}
function App() {
  const [data,setData]=useState(null),[error,setError]=useState(''),[route,setRoute]=useState(window.location.pathname),[source,setSource]=useState(null),[about,setAbout]=useState(false),[drawer,setDrawer]=useState(false);
  useEffect(()=>{fetch('/demo.json').then(r=>{if(!r.ok)throw new Error('The demo record could not be loaded.');return r.json();}).then(d=>{if(d.synthetic!==true||d.history.some(r=>!r.meta.synthetic)||!d.record.meta.synthetic)throw new Error('This preview only accepts synthetic demo data.');setData(d);}).catch(e=>setError(e.message));const handler=()=>setRoute(window.location.pathname);window.addEventListener('popstate',handler);return()=>window.removeEventListener('popstate',handler);},[]);
  function navigate(to){window.history.pushState({},'',to);setRoute(to);setDrawer(false);window.scrollTo({top:0,behavior:'instant'});document.title=`Ratio · ${navigation.find(n=>n[0]===to)?.[1]||'Trial review demo'}`;}
  const documents=data?Object.fromEntries([data.record,...data.history].flatMap(r=>r.documents).map(d=>[d.id,d])):{};
  const context={documents,messages:data?.messages,openSource:(span,heading)=>setSource({document:documents[span.doc_id],span,heading}),openDocument:document=>setSource({document})};
  return <Source.Provider value={context}><a className="skip-link" href="#main">Skip to review</a><button className="mobile-menu" aria-expanded={drawer} aria-controls="workspace-nav" onClick={()=>setDrawer(!drawer)}>{drawer?'Close navigation':'Menu'}</button>{drawer&&<button className="drawer-scrim" aria-label="Close navigation" onClick={()=>setDrawer(false)}/>}
    <aside id="workspace-nav" className={`sidebar ${drawer?'open':''}`}><a className="brand" href="/" onClick={e=>{e.preventDefault();navigate('/');}} aria-label="Ratio case overview"><span className="monogram" aria-hidden="true">r.</span><span><strong>ratio</strong><small>Fair trial analysis</small></span></a><p className="eyebrow">Review workspace</p><nav>{navigation.map(([path,label])=><a key={path} href={path} aria-current={route===path?'page':undefined} onClick={e=>{e.preventDefault();navigate(path);}}>{label}</a>)}</nav><div className="sidebar-note"><p className="eyebrow">Built for the record</p><p>From observation to evidence.<br/>Every finding, back to its source.</p><hr/><p>ICCPR Articles 9 &amp; 14</p><small>Analysis supports your review.<br/>Legal conclusions remain yours.</small></div></aside>
    <main id="main"><div className="masthead"><span>Trial review workspace</span><button className="demo-status" onClick={()=>setAbout(true)}><i aria-hidden="true"/>Hosted demo · Recorded analysis</button></div>{error?<div className="empty" role="alert"><h1>Demo unavailable</h1><p>{error}</p><button onClick={()=>location.reload()}>Reload</button></div>:!data?<div className="empty" role="status">Loading the synthetic trial record…</div>:<>{route==='/'?<Overview data={data} navigate={navigate} onAbout={()=>setAbout(true)}/>:route==='/coverage'?<Coverage data={data}/>:route==='/timeline'?<Timeline data={data}/>:route==='/reuse'?<Reasoning data={data}/>:route==='/judges'?<Judges data={data}/>:<><h1>Page not found</h1><p>This review page does not exist.</p><a href="/" onClick={e=>{e.preventDefault();navigate('/');}}>Return to case overview</a></>}<footer>Synthetic demonstration · Recorded local analysis · Legal review remains with you.</footer></>}
    </main>{data&&<SourceDialog value={source} onClose={()=>setSource(null)} messages={data.messages}/>}<AboutDialog open={about} onClose={()=>setAbout(false)}/>
  </Source.Provider>;
}

createRoot(document.getElementById('root')).render(<App/>);
