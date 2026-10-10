import React, { useState } from "react";
import { requestJSON } from "./library-client.mjs";
import "./triangulation.css";

const LABEL = "Model answer, unverified.";

function Citation({ citation, openSource }) {
  const excerpt = `“${citation.span.text.slice(0, 110)}${citation.span.text.length > 110 ? "…" : ""}”`;
  if (citation.kind === "case")
    return (
      <button className="text-action" onClick={() => openSource(citation.span, "Cited in the answer")}>
        {citation.document}: {excerpt}
      </button>
    );
  return (
    <a
      className="text-action"
      href={`/api/precedents/${encodeURIComponent(citation.reference.id)}/original`}
      target="_blank"
      rel="noreferrer"
    >
      {citation.reference.title}: {excerpt}
    </a>
  );
}

// The conversation lives only in this component's state: nothing is written to disk.
export function CaseChat({ caseId, openSource }) {
  const [question, setQuestion] = useState(""),
    [turns, setTurns] = useState([]),
    [busy, setBusy] = useState(false);
  async function ask(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setBusy(true);
    setQuestion("");
    try {
      const answer = await requestJSON(`/api/cases/${encodeURIComponent(caseId)}/chat`, { question: q });
      setTurns((t) => [...t, { q, answer }]);
    } catch (err) {
      setTurns((t) => [
        ...t,
        { q, answer: { supported: false, answer: err.message || "No answer was generated.", citations: [], label: LABEL } },
      ]);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="ai-case-brief case-chat" id="case-chat">
      <div className="section-head">
        <h2>Ask about this case</h2>
      </div>
      <p className="chat-banner" role="note">
        Local model only. Nothing leaves this computer.
      </p>
      {turns.map((t, i) => (
        <div key={i} className="chat-turn">
          <p className="chat-question">
            <strong>You:</strong> {t.q}
          </p>
          <div className={`chat-answer${t.answer.supported ? "" : " unsupported"}`}>
            <p className="caption">{t.answer.label || LABEL}</p>
            <p>{t.answer.answer}</p>
            {t.answer.citations.length > 0 && (
              <ul className="chat-citations">
                {t.answer.citations.map((c, j) => (
                  <li key={j}>
                    <Citation citation={c} openSource={openSource} />
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      ))}
      <form className="chat-form" onSubmit={ask}>
        <input
          aria-label="Question about this case"
          value={question}
          maxLength={500}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about this case's documents, for example: who had to prove what?"
        />
        <button className="primary" disabled={busy || !question.trim()}>
          {busy ? "Reading locally…" : "Ask"}
        </button>
      </form>
    </section>
  );
}
