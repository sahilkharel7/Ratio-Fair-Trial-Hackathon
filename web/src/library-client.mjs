export async function requestJSON(path,body) {
  const response=await fetch(path,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const result=await response.json();
  if(!response.ok)throw new Error(result.error||'The workspace request failed.');
  return result;
}
export async function workspaceMode(bundle) {
  const response=await fetch('/api/library',{cache:'no-store'});
  if(response.headers.get('content-type')?.includes('application/json')) {
    const result=await response.json();
    if(!response.ok||result.mode!=='local_sqlite')throw new Error(result.error||'The local case workspace is unavailable.');
    return result;
  }
  let history=[];
  try{history=JSON.parse(localStorage.getItem('ratio-focus-demo-assessments-v1')||'[]');}catch{}
  return {mode:'demo',cases:(bundle.collection?.cases||[]).map(c=>{const decisions=history.filter(d=>d.case_id===c.case_id),latest=latestAssessments(decisions);return {...c,decisions,pending_prompts:c.focus.prompts.filter(p=>!latest[p.id]||latest[p.id].decision==='follow_up').length};}),pending_files:[],outcomes:bundle.outcomes};
}
export function penaltyView(focus,stage) {
  const items=focus.penalties.filter(p=>p.stage===stage);
  const specific=items.filter(p=>p.min_months!=null||p.max_months!=null||p.kind!=='imprisonment');
  if(new Set(specific.map(p=>p.label)).size>1)return {label:'Several recorded terms — verify the source sequence',statement:null,ambiguous:true};
  const statement=specific[0]||items[0];
  return {label:statement?.label||'Not stated in the record',statement,ambiguous:false};
}
export function latestAssessments(history=[]) {
  const result={};
  for(const row of history)if(row.decision==='reopened')delete result[row.prompt_id];else result[row.prompt_id]=row;
  return result;
}
export function outcomeSummary(records,pattern='',country='') {
  const chosen=records.filter(r=>r.forum==='UN Human Rights Committee'&&r.standard==='ICCPR Article 14(2)'&&(!pattern||r.pattern===pattern)&&(!country||r.country===country));
  const merits=chosen.filter(r=>r.phase==='merits'&&['violation_found','no_violation'].includes(r.outcome));
  const favourable=merits.filter(r=>r.outcome==='violation_found').length;
  return {records:chosen,merits:merits.length,favourable,noViolation:merits.length-favourable,inadmissible:chosen.filter(r=>r.outcome==='inadmissible').length,
    rate:merits.length>=5?favourable/merits.length:null,prediction:null};
}
export async function encodeFiles(files) {
  const result=[];
  for(const file of files) {
    const bytes=new Uint8Array(await file.arrayBuffer());
    let binary='';
    for(let i=0;i<bytes.length;i+=32768)binary+=String.fromCharCode(...bytes.slice(i,i+32768));
    result.push({name:file.name,content:btoa(binary)});
  }
  return result;
}
