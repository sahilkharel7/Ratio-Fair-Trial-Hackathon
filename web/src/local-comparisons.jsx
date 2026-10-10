import React, {useState} from "react";
import {requestJSON} from "./library-client.mjs";
export function LocalComparisons({caseId,openSource}) {
 const [result,setResult]=useState(null),[busy,setBusy]=useState(false);
 async function load(){setBusy(true);try{setResult(await requestJSON(`/api/cases/${encodeURIComponent(caseId)}/comparisons`));}catch{setResult({comparisons:[],note:"Local comparison unavailable; no interpretation generated."});}finally{setBusy(false);}}
 return <section className="ai-case-brief"><div className="section-head"><h2>Compare source passages</h2><button disabled={busy} onClick={load}>{busy?"Comparing locally…":"Compare with references locally"}</button></div>
 <p className="caption">Public references are retrieved on this computer. Ollama compares the passages locally; case content and search queries do not go to a cloud model.</p>
 {result&&<><p className="caption">{result.note}</p>{result.comparisons.map((item,i)=><section key={i} className="outcome-method"><h3>{item.reference.title}</h3><div className="ai-brief-sections"><div><strong>This case</strong><blockquote>{item.case_span.text}</blockquote><button className="text-action" onClick={()=>openSource(item.case_span,"Case comparison passage")}>Open case source</button></div><div><strong>Legal reference</strong><blockquote>{item.reference_span.text}</blockquote><a href={`/api/precedents/${encodeURIComponent(item.reference.id)}/original`}>Read original decision</a></div></div><p>{item.similarity}</p><p className="caption">Limits: {item.limits}</p></section>)}</>}
 </section>;
}
