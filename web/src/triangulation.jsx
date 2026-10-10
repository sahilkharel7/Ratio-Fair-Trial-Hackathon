import React, { useEffect, useState } from "react";
import { requestJSON } from "./library-client.mjs";
import "./triangulation.css";

const DEFAULT_RIGHTS = [{ id: "14(2)", label: "Presumption of innocence (ICCPR Art. 14(2); General Comment 32, para. 30)" }];

function original(reference) {
  return `/api/precedents/${encodeURIComponent(reference.id)}/original`;
}

function Fact({ fact, reference, openSource }) {
  return (
    <div className="triangulation-fact">
      <p className="triangulation-fact-head">
        <strong>{fact.fact}</strong>
        {fact.comparison && (
          <span className={`triangulation-badge ${fact.comparison}`}>
            {fact.comparison === "similar" ? "Similar" : "Different"}
          </span>
        )}
      </p>
      <div className="triangulation-pair">
        <div>
          <span className="triangulation-side">This case</span>
          <blockquote>{fact.case_span.text}</blockquote>
          <button className="text-action" onClick={() => openSource(fact.case_span, "Triangulation · this case")}>
            View source
          </button>
        </div>
        <div>
          <span className="triangulation-side">{reference.title}</span>
          <blockquote>{fact.reference_span.text}</blockquote>
          <a className="text-action" href={original(reference)} target="_blank" rel="noreferrer">
            Read the original
          </a>
        </div>
      </div>
      {fact.why_relevant && <p>Why it matters: {fact.why_relevant}</p>}
      {fact.explanation && <p className="caption">{fact.explanation}</p>}
    </div>
  );
}

export function Triangulation({ caseId, openSource }) {
  const [rights, setRights] = useState(DEFAULT_RIGHTS),
    [right, setRight] = useState("14(2)"),
    [result, setResult] = useState(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    requestJSON("/api/rights")
      .then((reply) => setRights(reply.rights))
      .catch(() => setRights(DEFAULT_RIGHTS));
  }, []);
  async function run() {
    setBusy(true);
    setError("");
    try {
      setResult(
        await requestJSON(
          `/api/cases/${encodeURIComponent(caseId)}/triangulation?right=${encodeURIComponent(right)}`,
        ),
      );
    } catch (e) {
      setResult(null);
      setError(e.message || "Triangulation is unavailable.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="ai-case-brief triangulation" id="triangulation">
      <div className="section-head">
        <h2>Triangulation</h2>
      </div>
      <p className="caption">
        Compare this case with past public cases on one right. Retrieval and the local model both run on this
        computer; the lawyer draws the conclusions.
      </p>
      <div className="triangulation-controls">
        <label htmlFor="triangulation-right">Right</label>
        <select id="triangulation-right" value={right} onChange={(e) => setRight(e.target.value)}>
          {rights.map((r) => (
            <option key={r.id} value={r.id}>
              {r.label}
            </option>
          ))}
        </select>
        <button disabled={busy} onClick={run}>
          {busy ? "Finding similar cases locally…" : "Find similar past cases"}
        </button>
      </div>
      {error && <p className="caption" role="alert">{error}</p>}
      {result && (
        <>
          <p className="triangulation-pointer">{result.label}</p>
          <p className="caption">
            {result.note} {result.method}.
          </p>
          {result.similar_cases.length === 0 && (
            <p className="caption">No similar past case was found on this computer for this right.</p>
          )}
          {result.similar_cases.map((c, i) => {
            const facts = c.facts.length
              ? c.facts
              : c.passages.slice(0, 1).map((p) => ({
                  fact: "Closest passage (no model explanation)",
                  case_span: result.case_passages[p.closest_case_passage],
                  reference_span: p.reference_span,
                }));
            return (
              <section key={c.reference.id} className="outcome-method triangulation-case">
                <h3>
                  {i + 1}. {c.reference.title}
                </h3>
                <p className="caption">
                  Score {c.score.toFixed(2)} · passage similarity {c.cosine.toFixed(2)}
                  {c.shared_indicators.length > 0 && ` · both mention: ${c.shared_indicators.join("; ")}`}
                  {c.cites_right && " · cites the right"} · model text, unverified
                </p>
                {facts.map((f, j) => (
                  <Fact key={j} fact={f} reference={c.reference} openSource={openSource} />
                ))}
              </section>
            );
          })}
        </>
      )}
    </section>
  );
}
