import React, { useEffect, useRef, useState } from "react";
import { requestJSON } from "./library-client.mjs";
import "./triangulation.css";
import "./case-chat.css";

const LABEL = "Model answer, unverified.";
const SUGGESTIONS = [
  "Who had to prove what?",
  "What did the court rely on?",
  "What penalty was requested?",
];

function Citation({ citation, openSource }) {
  const excerpt = `“${citation.span.text.slice(0, 110)}${citation.span.text.length > 110 ? "…" : ""}”`;
  if (citation.kind === "case")
    return (
      <button
        className="text-action"
        onClick={() => openSource(citation.span, "Cited in the answer")}
      >
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

function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <path
        d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v8a2.5 2.5 0 0 1-2.5 2.5H10l-4.2 3.6a.5.5 0 0 1-.8-.4V16h0A2.5 2.5 0 0 1 4 13.5z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// A floating chat in the bottom-right corner of a case page. The conversation lives
// only in this component's state, so nothing is written to disk; closing the panel
// keeps it, and it ends when the page changes case.
export function CaseChat({ caseId, openSource }) {
  const [question, setQuestion] = useState(""),
    [turns, setTurns] = useState([]),
    [busy, setBusy] = useState(false),
    [pending, setPending] = useState(""),
    [open, setOpen] = useState(false);
  const input = useRef(null),
    launcher = useRef(null),
    log = useRef(null);

  useEffect(() => {
    if (open) input.current?.focus();
  }, [open]);
  useEffect(() => {
    log.current?.scrollTo({
      top: log.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns, busy]);

  function close() {
    setOpen(false);
    launcher.current?.focus();
  }
  async function ask(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setBusy(true);
    setPending(q);
    setQuestion("");
    try {
      const answer = await requestJSON(
        `/api/cases/${encodeURIComponent(caseId)}/chat`,
        { question: q },
      );
      setTurns((t) => [...t, { q, answer }]);
    } catch (err) {
      setTurns((t) => [
        ...t,
        {
          q,
          answer: {
            supported: false,
            answer: err.message || "No answer was generated.",
            citations: [],
            label: LABEL,
          },
        },
      ]);
    } finally {
      setBusy(false);
      setPending("");
    }
  }

  return (
    <>
      <button
        ref={launcher}
        className={`chat-launcher${open ? " is-open" : ""}`}
        aria-expanded={open}
        aria-controls="case-chat"
        onClick={() => (open ? close() : setOpen(true))}
      >
        <ChatIcon />
        <span>{open ? "Close chat" : "Ask about this case"}</span>
        {!open && turns.length > 0 && (
          <b aria-label={`${turns.length} answers`}>{turns.length}</b>
        )}
      </button>
      <section
        className="chat-panel"
        id="case-chat"
        role="dialog"
        aria-modal="false"
        aria-labelledby="case-chat-title"
        hidden={!open}
        onKeyDown={(e) => e.key === "Escape" && close()}
      >
        <header className="chat-panel-head">
          <div>
            <h2 id="case-chat-title">Ask about this case</h2>
            <p className="chat-banner" role="note">
              Local model only. Nothing leaves this computer.
            </p>
          </div>
          <button
            className="chat-close"
            aria-label="Close chat"
            onClick={close}
          >
            ×
          </button>
        </header>
        <div className="chat-log" ref={log} aria-live="polite">
          {turns.length === 0 && !busy && (
            <div className="chat-empty">
              <p>
                Ask a question about this case's documents. Answers quote the
                record and are for the reviewing lawyer to check.
              </p>
              <div className="chat-suggestions">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => {
                      setQuestion(s);
                      input.current?.focus();
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {turns.map((t, i) => (
            <div key={i} className="chat-turn">
              <p className="chat-question">{t.q}</p>
              <div
                className={`chat-answer${t.answer.supported ? "" : " unsupported"}`}
              >
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
          {busy && <p className="chat-question">{pending}</p>}
          {busy && (
            <p className="chat-reading" role="status">
              <span aria-hidden="true" />
              Reading the record locally…
            </p>
          )}
        </div>
        <form className="chat-form" onSubmit={ask}>
          <input
            ref={input}
            aria-label="Question about this case"
            value={question}
            maxLength={500}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about this case's documents…"
          />
          <button className="primary" disabled={busy || !question.trim()}>
            {busy ? "Reading…" : "Ask"}
          </button>
        </form>
      </section>
    </>
  );
}
