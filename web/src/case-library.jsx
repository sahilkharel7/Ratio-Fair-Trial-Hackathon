import React, { useContext, useEffect, useRef, useState } from "react";
import { Icon } from "./workspace.jsx";
import { firstMatch } from "./research.mjs";
import { downloadFile, markdownLiteral } from "./workflow.mjs";
import {
  workspaceMode,
  requestJSON,
  penaltyView,
  latestAssessments,
  outcomeSummary,
  encodeFiles,
} from "./library-client.mjs";

const patternNames = {
  burden_shift: "Burden of proof",
  official_guilt_statement: "Public statements of guilt",
  prejudicial_presentation: "Courtroom presentation",
  other: "Other / procedural stage",
};
const outcomeNames = {
  violation_found: "Article 14(2) violation found",
  no_violation: "No Article 14(2) violation found",
  inadmissible: "Inadmissible · no merits determination",
  not_examined: "Claim not examined",
};
const decisionNames = {
  follow_up: "Need more evidence",
  supported: "Concern supported by the record",
  not_supported: "Concern not supported",
  reopened: "Assessment reopened",
};

export function useCaseWorkspace(bundle) {
  const [workspace, setWorkspace] = useState(null),
    [error, setError] = useState("");
  async function refresh() {
    if (!bundle) return;
    try {
      setWorkspace(await workspaceMode(bundle));
      setError("");
    } catch (e) {
      setError(e.message);
    }
  }
  useEffect(() => {
    refresh();
  }, [bundle]);
  return { workspace, error, refresh };
}
export function CaseCollection({
  workspace,
  error,
  refresh,
  navigate,
  onImport,
}) {
  const [court, setCourt] = useState(""),
    [query, setQuery] = useState(""),
    [order, setOrder] = useState("recent");
  if (error)
    return (
      <div className="empty" role="alert">
        <h1>Case collection unavailable</h1>
        <p>{error}</p>
        <button onClick={refresh}>Try again</button>
      </div>
    );
  if (!workspace)
    return (
      <p className="empty" role="status">
        Opening the case collection…
      </p>
    );
  const courts = [...new Set(workspace.cases.map((c) => c.court))].sort();
  let cases = workspace.cases.filter(
    (c) =>
      (!court || c.court === court) &&
      `${c.defendant} ${c.title} ${c.charge} ${c.case_id}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  cases = [...cases].sort(
    order === "name"
      ? (a, b) => a.defendant.localeCompare(b.defendant)
      : (a, b) => b.updated_at.localeCompare(a.updated_at),
  );
  const waiting = cases.filter((c) => c.pending_prompts > 0).length;
  return (
    <>
      <div className="collection-heading">
        <div>
          <p className="eyebrow">Court workspace</p>
          <h1>Your case collection</h1>
          <p>Start with the person, the charge, and what is at stake.</p>
        </div>
        <button className="primary" onClick={onImport}>
          <Icon name="file" />
          Add case documents
        </button>
      </div>
      <div className="focus-ribbon">
        <span>
          <Icon name="scale" />
          Demo focus: <strong>Presumption of innocence</strong>
        </span>
        <span>ICCPR Article 14(2)</span>
      </div>
      <div className="collection-filters">
        <div>
          <label htmlFor="court-filter">Court</label>
          <select
            id="court-filter"
            value={court}
            onChange={(e) => setCourt(e.target.value)}
          >
            <option value="">All courts</option>
            {courts.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </div>
        <div className="collection-search">
          <label htmlFor="case-search">Find a case</label>
          <input
            id="case-search"
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Defendant, charge, or case reference…"
          />
        </div>
        <div>
          <label htmlFor="case-sort">Order</label>
          <select
            id="case-sort"
            value={order}
            onChange={(e) => setOrder(e.target.value)}
          >
            <option value="recent">Recently added</option>
            <option value="name">Defendant name</option>
          </select>
        </div>
      </div>
      <div className="collection-summary">
        <span>
          {cases.length} case{cases.length === 1 ? "" : "s"} · {waiting} with
          source prompts awaiting review
        </span>
        <span>
          {workspace.mode === "local_sqlite"
            ? "Saved on this computer"
            : "Synthetic demonstration collection"}
        </span>
      </div>
      {cases.length ? (
        <div className="matter-ledger">
          <div className="ledger-heading">
            <span>Defendant &amp; charge</span>
            <span>Prosecution seeks</span>
            <span>Review</span>
            <span />
          </div>
          {cases.map((c) => {
            const sought =
              c.penalty_summaries?.requested ||
              penaltyView(c.focus, "requested");
            return (
              <article className="ledger-row" key={c.case_id}>
                <div className="ledger-matter">
                  <a
                    href={`/cases/${encodeURIComponent(c.case_id)}`}
                    onClick={(e) => {
                      e.preventDefault();
                      navigate(`/cases/${encodeURIComponent(c.case_id)}`);
                    }}
                  >
                    {c.defendant}
                  </a>
                  <p>{c.charge}</p>
                  <div className="ledger-meta">
                    <span>{c.case_id}</span>
                    <span>{c.document_count} documents</span>
                    <span>
                      {c.synthetic ? "Synthetic example" : "Public record"}
                    </span>
                  </div>
                  <small>{c.court}</small>
                </div>
                <div className="ledger-stakes">
                  <strong>{sought.label}</strong>
                  <small>Recorded request</small>
                </div>
                <div className="ledger-status">
                  <span
                    className={`case-status ${c.pending_prompts ? "attention" : ""}`}
                  >
                    {c.pending_prompts
                      ? `${c.pending_prompts} passage${c.pending_prompts === 1 ? "" : "s"} to check`
                      : "Record check needed"}
                  </span>
                  <small>
                    {c.focus.prompts.length
                      ? "Source-linked screening"
                      : "No explicit screening passage"}
                  </small>
                </div>
                <button
                  className="text-action"
                  onClick={() =>
                    navigate(`/cases/${encodeURIComponent(c.case_id)}`)
                  }
                >
                  Open case <Icon name="arrow" size={15} />
                </button>
              </article>
            );
          })}
        </div>
      ) : (
        <div className="empty">
          <h2>No cases in this view</h2>
          <p>Add documents or try a different court or search.</p>
          <button className="primary" onClick={onImport}>
            Add case documents
          </button>
          {workspace.mode === "local_sqlite" && (
            <button
              onClick={async () => {
                await requestJSON("/api/demo", {});
                refresh();
              }}
            >
              Load synthetic examples
            </button>
          )}
        </div>
      )}
      {workspace.pending_files.length > 0 && (
        <div className="intake-notice">
          <Icon name="file" />
          <div>
            <strong>
              {workspace.pending_files.length} uploaded documents waiting for
              case assignment
            </strong>
            <p>
              The originals are saved. Review their grouping and document types
              to create the case entries.
            </p>
          </div>
          <button onClick={onImport}>Continue import</button>
        </div>
      )}
      <div className="collection-research-link">
        <div>
          <p className="eyebrow">Research context</p>
          <h2>What happened in international review?</h2>
          <p>
            Read source-coded Article 14(2) outcomes, with merits and
            inadmissibility kept separate.
          </p>
        </div>
        <button
          className="text-action"
          onClick={() => navigate("/international")}
        >
          Explore decisions <Icon name="arrow" size={15} />
        </button>
      </div>
    </>
  );
}

export function InternationalDecisions({
  registry,
  pattern: initialPattern = "",
  navigate,
}) {
  const [pattern, setPattern] = useState(initialPattern),
    [country, setCountry] = useState("");
  const records = registry?.records || [],
    summary = outcomeSummary(records, pattern, country);
  return (
    <>
      <div className="collection-heading">
        <div>
          <p className="eyebrow">
            Research context · UN Human Rights Committee
          </p>
          <h1>Presumption of innocence decisions</h1>
          <p>
            Read the outcome of the Article 14(2) claim, alongside its source.
          </p>
        </div>
        <button onClick={() => navigate("/")}>Case collection</button>
      </div>
      <div className="outcome-filters">
        <div>
          <label htmlFor="outcome-pattern">Issue in the record</label>
          <select
            id="outcome-pattern"
            value={pattern}
            onChange={(e) => setPattern(e.target.value)}
          >
            <option value="">All presumption-of-innocence examples</option>
            {Object.entries(patternNames).map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="outcome-country">State party</label>
          <select
            id="outcome-country"
            value={country}
            onChange={(e) => setCountry(e.target.value)}
          >
            <option value="">All states in this selection</option>
            {[...new Set(records.map((r) => r.country))].sort().map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>
      <div className="outcome-summary">
        <div>
          <span>Selected merits decisions</span>
          <strong>
            {summary.favourable} <small>of {summary.merits}</small>
          </strong>
          <p>Article 14(2) violation found</p>
        </div>
        <div>
          <span>Observed fraction</span>
          <strong>
            {summary.rate == null ? "—" : `${Math.round(summary.rate * 100)}%`}
          </strong>
          <p>
            {summary.rate == null
              ? "Fewer than five coded merits decisions"
              : "In this selected merits cohort only"}
          </p>
        </div>
        <div>
          <span>Procedural stage</span>
          <strong>{summary.inadmissible}</strong>
          <p>Inadmissible · excluded from the merits denominator</p>
        </div>
      </div>
      <p className="outcome-method-note">
        {registry?.selection} A historical finding does not establish release,
        acquittal or implementation. Case-specific prospects are not estimated.
      </p>
      <details className="outcome-method">
        <summary>How these outcomes are counted</summary>
        <p>
          Each communication counts once. The denominator includes only explicit
          Article 14(2) merits outcomes from this forum and selected
          issue/state. Inadmissible, unexamined and unknown claims are excluded.
          A violation of another article does not count as a favourable Article
          14(2) finding. The majority outcome is counted; a dissent is not
          substituted for it.
        </p>
        <p>
          These are source-checked draft labels for legal review. The examples
          are deliberately curated, so this percentage is not a representative
          success rate or a forecast for an uploaded case. Remedy implementation
          is not tracked in this starter collection.
        </p>
      </details>
      <div className="outcome-cards">
        {summary.records.map((r) => (
          <OutcomeCard key={r.id} record={r} />
        ))}
      </div>
      {!summary.records.length && (
        <p className="empty">No coded decision matches these filters.</p>
      )}
    </>
  );
}
function OutcomeCard({ record: r }) {
  return (
    <article className="outcome-card">
      <div className="outcome-card-heading">
        <div>
          <p className="eyebrow">
            {r.country} · {r.year} · Communication {r.communication}
          </p>
          <h2>{r.title}</h2>
        </div>
        <span className={`outcome-status ${r.outcome}`}>
          {outcomeNames[r.outcome] || r.outcome}
        </span>
      </div>
      <p>{r.summary}</p>
      <details>
        <summary>Source and coding</summary>
        <blockquote>“{r.excerpt}”</blockquote>
        <p className="caption">{r.citation}</p>
        <p className="caption">
          {r.source_kind.replaceAll("_", " ")} · Source checked{" "}
          {r.source_checked} · Draft coding for legal review · Implementation
          not tracked
        </p>
      </details>
      <a href={r.source_url} target="_blank" rel="noreferrer">
        Read the official source <Icon name="arrow" size={13} />
      </a>
    </article>
  );
}

export function CaseReview({
  caseId,
  workspace,
  bundle,
  context,
  navigate,
  onOpen,
  refresh,
}) {
  const { openSource, openDocument } = useContext(context);
  const [payload, setPayload] = useState(null),
    [error, setError] = useState(""),
    [query, setQuery] = useState(""),
    [note, setNote] = useState(""),
    [notice, setNotice] = useState("");
  const [demoHistory, setDemoHistory] = useState(() => {
    try {
      return JSON.parse(
        localStorage.getItem("ratio-focus-demo-assessments-v1") || "[]",
      );
    } catch {
      return [];
    }
  });
  useEffect(() => {
    let active = true;
    setPayload(null);
    setError("");
    setNotice("");
    onOpen(null);
    if (!workspace) return;
    const load =
      workspace.mode === "local_sqlite"
        ? requestJSON(`/api/cases/${encodeURIComponent(caseId)}`)
        : Promise.resolve(
            (() => {
              const p = bundle.collection.records.find(
                (p) => p.record.case_id === caseId,
              );
              try {
                const notes = JSON.parse(
                  localStorage.getItem("ratio-focus-demo-notes-v1") || "{}",
                );
                return p ? { ...p, notes: notes[caseId] || p.notes || [] } : p;
              } catch {
                return p;
              }
            })(),
          );
    load
      .then((p) => {
        if (!p) throw new Error("This case is not in the collection.");
        if (active) {
          setPayload(p);
          onOpen(p);
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [caseId, workspace?.mode]);
  const history =
    workspace?.mode === "local_sqlite"
      ? payload?.decisions || []
      : demoHistory.filter((d) => d.case_id === caseId);
  async function save(decision) {
    if (workspace.mode === "local_sqlite") {
      await requestJSON("/api/focus-reviews", decision);
      const fresh = await requestJSON(
        `/api/cases/${encodeURIComponent(caseId)}`,
      );
      setPayload(fresh);
      refresh();
    } else {
      const next = [
        ...demoHistory,
        { ...decision, created_at: new Date().toISOString() },
      ];
      setDemoHistory(next);
      localStorage.setItem(
        "ratio-focus-demo-assessments-v1",
        JSON.stringify(next),
      );
      refresh();
    }
    setNotice("Assessment saved with its source and history.");
  }
  async function saveNote(e) {
    e.preventDefault();
    if (!note.trim()) return;
    try {
      if (workspace.mode === "local_sqlite") {
        await requestJSON("/api/case-notes", { case_id: caseId, note });
        setPayload(
          await requestJSON(`/api/cases/${encodeURIComponent(caseId)}`),
        );
      } else {
        const next = [
          ...(payload.notes || []),
          { note, created_at: new Date().toISOString() },
        ];
        const all = JSON.parse(
          localStorage.getItem("ratio-focus-demo-notes-v1") || "{}",
        );
        localStorage.setItem(
          "ratio-focus-demo-notes-v1",
          JSON.stringify({ ...all, [caseId]: next }),
        );
        setPayload((p) => ({ ...p, notes: next }));
      }
      setNote("");
      setNotice(
        workspace.mode === "local_sqlite"
          ? "Working note saved on this computer."
          : "Demo note saved in this browser.",
      );
    } catch (e) {
      setNotice(e.message);
    }
  }
  if (error)
    return (
      <div className="empty" role="alert">
        <h1>Case unavailable</h1>
        <p>{error}</p>
        <button onClick={() => navigate("/")}>Case collection</button>
      </div>
    );
  if (!payload)
    return (
      <p className="empty" role="status">
        Opening the stored case record…
      </p>
    );
  const { record, focus } = payload,
    latest = latestAssessments(history),
    patterns = [...new Set(focus.prompts.map((p) => p.pattern))];
  const comparable = (
    workspace.outcomes?.records || bundle.outcomes.records
  ).filter((r) => !patterns.length || patterns.includes(r.pattern));
  const docs = record.documents.filter(
    (d) =>
      !query ||
      `${d.title} ${d.text}`.toLowerCase().includes(query.toLowerCase()),
  );
  function exportReport() {
    if (workspace.mode === "local_sqlite") {
      window.location.href = `/api/cases/${encodeURIComponent(caseId)}/report`;
      return;
    }
    const lines = [
      `# Case review: ${markdownLiteral(record.meta.title)}`,
      "",
      "Synthetic demonstration",
      `Court: ${markdownLiteral(record.meta.court)}`,
      `Charge: ${markdownLiteral(focus.charge)}`,
      "",
      "## Penalty statements",
    ];
    for (const p of focus.penalties)
      lines.push(
        `${p.stage}: ${p.label}`,
        ...markdownLiteral(p.span.text)
          .split("\n")
          .map((line) => "> " + line),
        "",
      );
    lines.push(
      "## Presumption of innocence",
      focus.standard,
      focus.screening_note,
    );
    for (const p of focus.prompts) {
      lines.push(
        p.title,
        ...markdownLiteral(p.span.text)
          .split("\n")
          .map((line) => "> " + line),
      );
      const d = latest[p.id];
      if (d)
        lines.push(
          `Reviewer assessment: ${d.decision}`,
          markdownLiteral(d.reason),
        );
    }
    for (const n of payload.notes || [])
      lines.push("Working note: " + markdownLiteral(n.note));
    lines.push(
      "",
      "International outcomes are research context, not a success prediction.",
    );
    downloadFile(lines.join("\n"), `${caseId}-case-review.md`, "text/markdown");
  }
  return (
    <>
      <button
        className="text-action back-to-cases"
        onClick={() => navigate("/")}
      >
        <Icon name="arrow" size={14} />
        Case collection
      </button>
      <div className="case-review-heading">
        <div>
          <p className="eyebrow">{record.meta.court}</p>
          <h1>{focus.defendant}</h1>
          <p>
            {record.meta.title} · {caseId}{" "}
            <span className="synthetic-pill">
              {record.meta.synthetic ? "Synthetic example" : "Public record"}
            </span>
          </p>
        </div>
        <button onClick={exportReport}>
          <Icon name="download" />
          Case worksheet
        </button>
      </div>
      <section className="charge-panel">
        <p className="eyebrow">The charge</p>
        <h2>{focus.charge}</h2>
        {focus.charge_span ? (
          <button
            className="text-action"
            onClick={() =>
              openSource(focus.charge_span, "Charge as stated in the record")
            }
          >
            Read the charge in its source <Icon name="arrow" size={14} />
          </button>
        ) : (
          <p className="caption">
            Entered in case metadata · Verify against the indictment.
          </p>
        )}
      </section>
      <div className="stake-panels">
        {[
          ["requested", "Prosecution seeks"],
          ["imposed", "Sentence imposed"],
          ["statutory", "Quoted statutory penalty"],
        ].map(([stage, label]) => {
          const view = penaltyView(focus, stage);
          return (
            <section key={stage} className={`stake-panel ${stage}`}>
              <span>{label}</span>
              <strong>{view.label}</strong>
              {view.statement && (
                <button
                  className="text-action"
                  onClick={() => openSource(view.statement.span, label)}
                >
                  View source <Icon name="file" size={13} />
                </button>
              )}
              {view.ambiguous && (
                <small>
                  Review the original statements before assigning a term to this
                  defendant.
                </small>
              )}
            </section>
          );
        })}
      </div>
      <details className="penalty-history">
        <summary>
          All recorded penalty statements ({focus.penalties.length})
        </summary>
        {focus.penalties.map((p, i) => (
          <div key={i}>
            <strong>
              {p.stage}: {p.label}
            </strong>
            <p>{p.span.text}</p>
            <button
              className="text-action"
              onClick={() => openSource(p.span, p.stage + " penalty statement")}
            >
              Open source
            </button>
          </div>
        ))}
      </details>
      <div className="focus-case-grid">
        <div>
          <section className="innocence-section">
            <div className="section-head">
              <div>
                <p className="eyebrow">Focused review</p>
                <h2>Presumption of innocence</h2>
              </div>
              <span className="count-label">
                {focus.prompts.length} source prompt
                {focus.prompts.length === 1 ? "" : "s"}
              </span>
            </div>
            <p className="focus-standard">{focus.standard}</p>
            <p className="caption">
              Source-led English screening. The lawyer determines whether the
              record supports a violation.
            </p>
            {focus.prompts.length ? (
              focus.prompts.map((p) => (
                <FocusPrompt
                  key={p.id}
                  prompt={p}
                  caseId={caseId}
                  record={record}
                  current={latest[p.id]}
                  history={history.filter((d) => d.prompt_id === p.id)}
                  save={save}
                  openSource={openSource}
                />
              ))
            ) : (
              <div className="record-gap">
                <h3>Verify the record with the monitor</h3>
                <p>
                  No explicit screening passage was identified. This does not
                  establish compliance.
                </p>
                <ul>
                  <li>Who carried the burden of proving the charge?</li>
                  <li>Did a public official affirm guilt before judgment?</li>
                  <li>
                    Did courtroom presentation suggest that the defendant was a
                    dangerous criminal?
                  </li>
                </ul>
              </div>
            )}
          </section>
          <section className="source-record-section">
            <div className="section-head">
              <div>
                <h2>Original case record</h2>
                <p>Read the document or find an exact word.</p>
              </div>
              <span className="count-label">
                {record.documents.length} documents
              </span>
            </div>
            <label className="sr-only" htmlFor="focus-source-search">
              Search this case’s documents
            </label>
            <input
              id="focus-source-search"
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Find words in the original record…"
            />
            {docs.map((doc) => (
              <div className="source-record-row" key={doc.id}>
                <Icon name="file" />
                <div>
                  <strong>{doc.title}</strong>
                  <small>{doc.type.replaceAll("_", " ")}</small>
                </div>
                <button
                  className="text-action"
                  onClick={() => {
                    const span = firstMatch(doc, query);
                    span
                      ? openSource(span, "Search result")
                      : openDocument(doc);
                  }}
                >
                  {query ? "Read match" : "Read document"}
                  <Icon name="arrow" size={13} />
                </button>
              </div>
            ))}
            {!docs.length && (
              <p className="caption">No document matches this search.</p>
            )}
          </section>
        </div>
        <aside className="case-working-rail">
          <section className="rail-panel">
            <div className="rail-heading">
              <Icon name="bookmark" />
              <h2>International review context</h2>
            </div>
            <p className="caption">
              {patterns.length
                ? "Research examples for the recorded issue."
                : "Research examples to guide the record check."}{" "}
              They do not estimate this case’s prospects.
            </p>
            {comparable.slice(0, 3).map((r) => (
              <div className="compact-precedent" key={r.id}>
                <strong>{r.title}</strong>
                <small>{outcomeNames[r.outcome]}</small>
                <a href={r.source_url} target="_blank" rel="noreferrer">
                  Official source →
                </a>
              </div>
            ))}
            <button
              className="text-action"
              onClick={() =>
                navigate(
                  `/international${patterns.length === 1 ? "?pattern=" + patterns[0] : ""}`,
                )
              }
            >
              Compare coded outcomes <Icon name="arrow" size={14} />
            </button>
          </section>
          <section className="rail-panel next-step-panel">
            <div className="rail-heading">
              <Icon name="check" />
              <h2>Your next step</h2>
            </div>
            <form onSubmit={saveNote}>
              <label htmlFor="case-working-note">Working note</label>
              <textarea
                id="case-working-note"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                rows={5}
                placeholder="Missing evidence, a question for the monitor, or the action to take next…"
              />
              <button type="submit" className="primary">
                Save working note
              </button>
            </form>
            <small>
              {workspace.mode === "local_sqlite"
                ? "Saved in this case’s SQLite record."
                : "Synthetic demo notes saved in this browser."}
            </small>
            {(payload.notes || []).map((n, i) => (
              <p className="saved-case-note" key={i}>
                {n.note}
              </p>
            ))}
          </section>
          <details className="rail-panel">
            <summary>Supporting analysis</summary>
            <p className="caption">
              Broader review tools remain available. The focused workspace keeps
              the charge, stakes and Article 14(2) review in the foreground.
            </p>
            {caseId === "venn-2025" ? (
              <button
                className="text-action"
                onClick={() => navigate("/overview")}
              >
                Open full analysis record →
              </button>
            ) : (
              <p className="caption">
                This record has focused screening. Run the full local pipeline
                when the additional analysis is needed.
              </p>
            )}
          </details>
        </aside>
      </div>
      {notice && (
        <p className="case-save-notice" role="status">
          {notice}
        </p>
      )}
    </>
  );
}
function FocusPrompt({
  prompt: p,
  caseId,
  current,
  history,
  save,
  openSource,
}) {
  const [decision, setDecision] = useState(current?.decision || "follow_up"),
    [reason, setReason] = useState(current?.reason || ""),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  async function submit(e) {
    e.preventDefault();
    if (!reason.trim()) {
      setError("Record a reason and the next step.");
      return;
    }
    setBusy(true);
    try {
      await save({
        case_id: caseId,
        prompt_id: p.id,
        decision,
        reason,
        source_snapshot: p,
      });
      setError("");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <article className="focus-prompt">
      <div className="row-heading">
        <h3>{p.title}</h3>
        <span className="case-status attention">
          {current ? decisionNames[current.decision] : "Needs legal review"}
        </span>
      </div>
      <p>{p.question}</p>
      <blockquote>{p.span.text}</blockquote>
      <button
        className="text-action"
        onClick={() => openSource(p.span, p.title)}
      >
        Check the original passage <Icon name="arrow" size={14} />
      </button>
      <details className="focus-assessment" open={current ? true : undefined}>
        <summary>Record your assessment</summary>
        <form onSubmit={submit}>
          <label htmlFor={`assess-${p.id}`}>Assessment</label>
          <select
            id={`assess-${p.id}`}
            value={decision}
            onChange={(e) => setDecision(e.target.value)}
          >
            {["follow_up", "supported", "not_supported"].map((id) => (
              <option key={id} value={id}>
                {decisionNames[id]}
              </option>
            ))}
          </select>
          <label htmlFor={`reason-${p.id}`}>Reason and next step</label>
          <textarea
            id={`reason-${p.id}`}
            rows={3}
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Explain what the source establishes, or what evidence is still needed…"
          />
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          <div className="decision-actions">
            <button className="primary" disabled={busy}>
              {busy ? "Saving…" : "Save assessment"}
            </button>
            {current && (
              <button
                type="button"
                onClick={() =>
                  save({
                    case_id: caseId,
                    prompt_id: p.id,
                    decision: "reopened",
                    reason: "",
                    source_snapshot: p,
                  })
                }
              >
                Reopen
              </button>
            )}
          </div>
        </form>
      </details>
      {history.length > 0 && (
        <details className="focus-history">
          <summary>Assessment history ({history.length})</summary>
          {history.map((d, i) => (
            <p key={i}>
              {decisionNames[d.decision]} ·{" "}
              {new Date(d.created_at).toLocaleString()}
              <br />
              {d.reason}
            </p>
          ))}
        </details>
      )}
    </article>
  );
}

export function IntakeDialog({ open, onClose, workspace, refresh, navigate }) {
  const ref = useRef(null);
  const [files, setFiles] = useState([]),
    [provenance, setProvenance] = useState("public"),
    [source, setSource] = useState(""),
    [declared, setDeclared] = useState(false),
    [pending, setPending] = useState([]),
    [picked, setPicked] = useState([]),
    [types, setTypes] = useState({}),
    [form, setForm] = useState({
      reference: "",
      defendant: "",
      title: "",
      court: "",
      charge: "",
    }),
    [sameCase, setSameCase] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    if (open) {
      ref.current.showModal();
      setPending(workspace?.pending_files || []);
    } else ref.current.close();
  }, [open]);
  async function upload(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (
        files.length > 100 ||
        files.some((f) => f.size > 5_000_000) ||
        files.reduce((n, f) => n + f.size, 0) > 50_000_000
      )
        throw new Error(
          "Upload at most 100 files, 5 MB each and 50 MB in total.",
        );
      const result = await requestJSON("/api/intake", {
        files: await encodeFiles(files),
        provenance,
        source_note: source,
      });
      setPending(result.pending_files);
      setFiles([]);
      refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function create(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const selected = pending.filter((f) => picked.includes(f.id));
      if (!selected.length || selected.some((f) => !types[f.id]))
        throw new Error(
          "Select this case’s files and confirm every document type.",
        );
      const reference =
        form.reference.trim() || "case-" + crypto.randomUUID().slice(0, 8);
      const manifest = {
        case_id: reference,
        title: form.title.trim() || form.defendant.trim(),
        court: form.court.trim(),
        charge_type: form.charge.trim() || "Not recorded",
        data_provenance: selected[0].provenance,
        synthetic: selected[0].provenance === "synthetic",
        source_note: selected[0].source_note || null,
        documents: selected.map((f) => ({
          path: f.name,
          title: f.name,
          type: types[f.id],
        })),
      };
      const result = await requestJSON("/api/cases", {
        manifest,
        file_ids: Object.fromEntries(selected.map((f) => [f.name, f.id])),
        defendant: form.defendant.trim(),
      });
      await refresh();
      setPicked([]);
      setSameCase(false);
      onClose();
      navigate(`/cases/${encodeURIComponent(result.record.case_id)}`);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <dialog ref={ref} className="intake-dialog" onCancel={onClose}>
      <div className="dialog-heading">
        <div>
          <p className="eyebrow">Case collection</p>
          <h2>Add case documents</h2>
        </div>
        <button aria-label="Close case import" onClick={onClose}>
          <Icon name="close" />
        </button>
      </div>
      {workspace?.mode !== "local_sqlite" ? (
        <div className="local-upload-guide">
          <h3>Use the local workspace for your documents</h3>
          <p>
            This hosted collection contains synthetic examples. The local React
            workspace saves uploaded PDFs and case entries in SQLite on your
            computer.
          </p>
          <ol>
            <li>Build the interface.</li>
            <li>Start the local workspace server.</li>
            <li>
              Open its link, upload the documents, and assign them to a case.
            </li>
          </ol>
          <pre>
            npm --prefix web run build{"\n"}.venv/bin/python
            scripts/serve_workspace.py --demo
          </pre>
          <p className="caption">
            Open http://127.0.0.1:8503. The existing Streamlit app uses the same
            case database.
          </p>
        </div>
      ) : (
        <div className="intake-body">
          <section>
            <h3>1. Add PDFs and source documents</h3>
            <p className="caption">
              Originals are saved before case assignment. Textless PDFs remain
              in the queue until an OCR version is supplied.
            </p>
            <form onSubmit={upload}>
              <label htmlFor="input-files">Documents · PDF, TXT or MD</label>
              <input
                id="input-files"
                type="file"
                accept=".pdf,.txt,.md"
                multiple
                onChange={(e) => setFiles([...e.target.files])}
              />
              <div className="intake-metadata">
                <div>
                  <label htmlFor="input-provenance">Material</label>
                  <select
                    id="input-provenance"
                    value={provenance}
                    onChange={(e) => setProvenance(e.target.value)}
                  >
                    <option value="public">Public records</option>
                    <option value="synthetic">Synthetic examples</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="input-source">Publication/source note</label>
                  <input
                    id="input-source"
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                    required={provenance === "public"}
                    placeholder="Where the public documents were published"
                  />
                </div>
              </div>
              <label className="intake-check">
                <input
                  type="checkbox"
                  checked={declared}
                  onChange={(e) => setDeclared(e.target.checked)}
                />
                These are public or synthetic documents and contain no
                confidential monitoring material.
              </label>
              <button
                className="primary"
                disabled={!files.length || !declared || busy}
              >
                {busy
                  ? "Saving documents…"
                  : `Save ${files.length || ""} documents for assignment`}
              </button>
            </form>
          </section>
          {pending.length > 0 && (
            <section className="assign-case-section">
              <h3>2. Assign documents to one case</h3>
              <p className="caption">
                {pending.length} unassigned documents. Select the documents
                belonging together and confirm their roles.
              </p>
              <form onSubmit={create}>
                <div className="intake-file-list">
                  {pending.map((f) => (
                    <div className="intake-file" key={f.id}>
                      <label>
                        <input
                          type="checkbox"
                          disabled={!!f.problem}
                          checked={picked.includes(f.id)}
                          onChange={(e) =>
                            setPicked((previous) =>
                              e.target.checked
                                ? [...previous, f.id]
                                : previous.filter((id) => id !== f.id),
                            )
                          }
                        />
                        <span>
                          {f.name}
                          {f.problem && <small>{f.problem}</small>}
                        </span>
                      </label>
                      <label className="sr-only" htmlFor={`file-type-${f.id}`}>
                        Document type: {f.name}
                      </label>
                      <select
                        id={`file-type-${f.id}`}
                        value={types[f.id] || ""}
                        disabled={!picked.includes(f.id)}
                        onChange={(e) =>
                          setTypes({ ...types, [f.id]: e.target.value })
                        }
                      >
                        <option value="">Select document type</option>
                        {[
                          "indictment",
                          "judgment",
                          "monitoring_note",
                          "transcript",
                          "detention_order",
                        ].map((type) => (
                          <option key={type} value={type}>
                            {type.replaceAll("_", " ")}
                          </option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
                <div className="intake-case-fields">
                  {[
                    ["defendant", "Defendant under review"],
                    ["title", "Case title as written"],
                    ["court", "Court and chamber"],
                    ["reference", "Short collection reference (optional)"],
                    ["charge", "Charge summary (optional)"],
                  ].map(([key, label]) => (
                    <div key={key}>
                      <label htmlFor={"case-" + key}>{label}</label>
                      <input
                        id={"case-" + key}
                        value={form[key]}
                        onChange={(e) =>
                          setForm({ ...form, [key]: e.target.value })
                        }
                        required={["defendant", "court"].includes(key)}
                      />
                    </div>
                  ))}
                </div>
                <label className="intake-check">
                  <input
                    type="checkbox"
                    checked={sameCase}
                    onChange={(e) => setSameCase(e.target.checked)}
                  />
                  These selected documents belong to this one case, and their
                  document types are correct.
                </label>
                <button
                  className="primary"
                  disabled={!picked.length || !sameCase || busy}
                >
                  {busy ? "Creating case…" : "Create stored case"}
                </button>
              </form>
            </section>
          )}
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
        </div>
      )}
    </dialog>
  );
}
