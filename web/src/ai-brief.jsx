import React, { useEffect, useState } from "react";
import { requestJSON } from "./library-client.mjs";

export function AICaseBrief({ caseId, openSource }) {
  const [brief, setBrief] = useState(null), [error, setError] = useState("");
  const pending = brief?.status === "queued" || brief?.status === "running";
  useEffect(() => {
    let active = true;
    requestJSON(`/api/cases/${encodeURIComponent(caseId)}/brief`)
      .then((value) => { if (active) setBrief(value); })
      .catch((e) => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [caseId]);
  useEffect(() => {
    if (!pending) return;
    let active = true;
    const timer = setInterval(() => {
      requestJSON(`/api/cases/${encodeURIComponent(caseId)}/brief`)
        .then((value) => { if (active) { setBrief(value); setError(""); } })
        .catch((e) => { if (active) setError(e.message); });
    }, 2000);
    return () => { active = false; clearInterval(timer); };
  }, [caseId, pending]);
  async function generate(force = false) {
    setError("");
    setBrief((value) => ({ ...value, status: "queued" }));
    try { setBrief(await requestJSON(`/api/cases/${encodeURIComponent(caseId)}/brief`, { force })); }
    catch (e) { setError(e.message); setBrief((value) => ({ ...value, status: "failed" })); }
  }
  return (
    <section className="ai-case-brief" aria-labelledby="ai-brief-heading">
      <div className="section-head">
        <div><p className="eyebrow">A reading aid for your review</p><h2 id="ai-brief-heading">One-page case brief</h2></div>
        {brief?.status === "ready" ? <a className="primary" href={`/api/cases/${encodeURIComponent(caseId)}/brief.pdf`}>Download one-pager</a>
          : <button className="primary" disabled={pending} onClick={() => generate()}>{pending ? "Drafting the brief…" : "Draft case brief"}</button>}
      </div>
      <p className="caption">Selected public or synthetic source excerpts are sent to Groq. The draft covers the charge, recorded consequences and presumption of innocence, with supporting passages. Check the full originals before relying on it.</p>
      {pending && <p role="status">{brief.status === "queued" ? "Queued on this computer." : "Reading selected source passages and checking citations."} Free-tier limits can delay a draft; you can continue reviewing the case.</p>}
      {(error || brief?.error) && <p className="warning" role="alert">{error || brief.error}</p>}
      {brief?.status === "ready" && <>
        <p className="caption">{brief.method}</p>
        <div className="ai-brief-sections">{brief.sections.map((section, i) => <section key={i}>
          <h3>{section.label}</h3><p>{section.summary}</p>
          <button className="text-action" onClick={() => openSource(section.span, section.label)}>Read supporting passage</button>
          <small>{section.pinpoint}</small>
        </section>)}</div>
        <button className="text-action" onClick={() => generate(true)}>Regenerate draft</button>
      </>}
    </section>
  );
}
