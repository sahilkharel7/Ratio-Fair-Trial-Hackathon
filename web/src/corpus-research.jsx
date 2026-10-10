import React, { useEffect, useRef, useState } from "react";
import { BriefSections } from "./court-library.jsx";
import { requestJSON } from "./library-client.mjs";

const kinds = {
  court_judgment: "Court judgments",
  ccpr_views: "Human Rights Committee decisions",
  wgad_opinion: "Working Group opinions",
  trialwatch_report: "TrialWatch monitoring reports",
  collection_document: "Uploaded reference collections",
};
const suggestions = [
  "presumption of innocence",
  "burden of proof",
  "access to counsel",
  "defence witnesses",
  "pretrial detention",
];

export function CorpusResearch({ onImport, refresh }) {
  const [catalogue, setCatalogue] = useState(null),
    [query, setQuery] = useState("presumption of innocence"),
    [kind, setKind] = useState(""),
    [state, setState] = useState(""),
    [mode, setMode] = useState("phrase"),
    [result, setResult] = useState(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [source, setSource] = useState(null),
    [staging, setStaging] = useState(false);
  const sequence = useRef(0),
    readerRef = useRef(null);
  useEffect(() => {
    if (source) {
      readerRef.current?.scrollIntoView({ block: "start" });
      readerRef.current?.focus();
    }
  }, [source]);
  useEffect(() => {
    let active = true;
    requestJSON("/api/precedents")
      .then((data) => {
        if (active) setCatalogue(data);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
      sequence.current++;
    };
  }, []);
  async function search(phrase = query) {
    const current = ++sequence.current;
    setBusy(true);
    setError("");
    setSource(null);
    try {
      const response = await requestJSON(
        "/api/precedents/search?" +
          new URLSearchParams({ q: phrase, kind, state, mode }),
      );
      if (current === sequence.current) setResult(response);
    } catch (e) {
      if (current === sequence.current) setError(e.message);
    } finally {
      if (current === sequence.current) setBusy(false);
    }
  }
  async function read(hit) {
    setError("");
    const current = ++sequence.current;
    try {
      const document = await requestJSON(
        "/api/precedents/" + encodeURIComponent(hit.document.id),
      );
      if (current === sequence.current)
        setSource({ ...document, hit: hit.text ? hit : null });
    } catch (e) {
      setError(e.message);
    }
  }
  async function stageSource() {
    setStaging(true);
    setError("");
    try {
      await requestJSON(
        `/api/precedents/${encodeURIComponent(source.id)}/stage`,
        {},
      );
      await refresh();
      onImport();
    } catch (e) {
      setError(e.message);
    } finally {
      setStaging(false);
    }
  }
  return (
    <section className="corpus-research" aria-labelledby="corpus-heading">
      <div className="collection-heading">
        <div>
          <p className="eyebrow">Local reference library</p>
          <h2 id="corpus-heading">Research the downloaded cases</h2>
          <p>
            Find the passage, read its context, then decide whether the
            authority fits your matter.
          </p>
        </div>
        {catalogue?.available && (
          <span className="case-status">
            {catalogue.documents.length} saved documents
          </span>
        )}
      </div>
      <p className="outcome-method-note">
        Reference reports and decisions stay separate from your active matters
        and the manually coded outcomes above. A wording match does not
        establish a violation.
      </p>
      {error && (
        <p role="alert" className="empty">
          {error}
        </p>
      )}
      {!catalogue && !error && (
        <p role="status">Opening the reference library…</p>
      )}
      {catalogue?.available ? (
        <>
          <form
            className="corpus-search"
            onSubmit={(e) => {
              e.preventDefault();
              search();
            }}
          >
            <div>
              <label htmlFor="corpus-query">Phrase in the record</label>
              <input
                id="corpus-query"
                type="search"
                maxLength={300}
                minLength={3}
                required
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="presumption of innocence"
              />
            </div>
            <div>
              <label htmlFor="corpus-mode">Search method</label>
              <select
                id="corpus-mode"
                value={mode}
                onChange={(e) => setMode(e.target.value)}
              >
                <option value="phrase">Exact phrase</option>
                <option value="meaning">Meaning and wording · local</option>
              </select>
            </div>
            <div>
              <label htmlFor="corpus-kind">Source</label>
              <select
                id="corpus-kind"
                value={kind}
                onChange={(e) => setKind(e.target.value)}
              >
                <option value="">All sources</option>
                {Object.entries(kinds).map(([id, label]) => (
                  <option value={id} key={id}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="corpus-state">State</label>
              <select
                id="corpus-state"
                value={state}
                onChange={(e) => setState(e.target.value)}
              >
                <option value="">All states</option>
                {[
                  ...new Set(
                    catalogue.documents.map((d) => d.state).filter(Boolean),
                  ),
                ]
                  .sort()
                  .map((s) => (
                    <option key={s}>{s}</option>
                  ))}
              </select>
            </div>
            <button className="primary" disabled={busy}>
              {busy ? "Searching…" : "Search passages"}
            </button>
          </form>
          <div
            className="corpus-suggestions"
            aria-label="Suggested research phrases"
          >
            {suggestions.map((phrase) => (
              <button
                key={phrase}
                onClick={() => {
                  setQuery(phrase);
                  search(phrase);
                }}
                disabled={busy}
              >
                {phrase}
              </button>
            ))}
          </div>
          {result && (
            <p role="status" className="caption">
              {result.total_documents} documents{" "}
              {result.method.startsWith("Exact")
                ? "contain the phrase"
                : "in shown results"}{" "}
              · {result.method}. Showing up to 20 passages.
            </p>
          )}
          {result?.hits.map((hit) => (
            <article
              className="outcome-card"
              key={`${hit.document.id}-${hit.match_start}`}
            >
              <p className="eyebrow">
                {hit.document.body} ·{" "}
                {hit.document.state || "State not recorded"} ·{" "}
                {hit.document.year || "Year not recorded"}
              </p>
              <h3>{hit.document.title}</h3>
              <p className="caption">
                {kinds[hit.document.kind]} ·{" "}
                {hit.document.private
                  ? "Private reference"
                  : "Public reference"}{" "}
                · line {hit.line}
              </p>
              <blockquote className="corpus-excerpt">{hit.text}</blockquote>
              <button onClick={() => read(hit)}>
                Read saved source in context
              </button>
              {!hit.document.private &&
                hit.document.url.startsWith("https://") && (
                  <a href={hit.document.url} target="_blank" rel="noreferrer">
                    Original publication
                  </a>
                )}
            </article>
          ))}
          {result && !result.hits.length && (
            <p className="empty">
              No matching passage in this selection. Try another phrase or
              broaden the filters. Absence from search is not evidence of
              compliance.
            </p>
          )}
          {source && (
            <section
              ref={readerRef}
              tabIndex={-1}
              className="corpus-reader"
              aria-label="Saved reference text"
            >
              <div className="collection-heading">
                <h3>{source.title}</h3>
                <button onClick={() => setSource(null)}>Close source</button>
              </div>
              <p className="caption">
                {source.attribution} · text checksum{" "}
                {source.text_sha256.slice(0, 12)}
              </p>
              {source.local_pdf && (
                <div className="corpus-suggestions">
                  <a
                    href={`/api/precedents/${encodeURIComponent(source.id)}/original`}
                  >
                    Download saved original PDF
                  </a>
                  <button disabled={staging} onClick={stageSource}>
                    {staging
                      ? "Adding original PDF…"
                      : "Use this PDF in a case review"}
                  </button>
                </div>
              )}
              {source.local_pdf && (
                <p className="caption">
                  The saved original is copied to local intake. Confirm the
                  defendant, court and document role before creating the matter.
                </p>
              )}
              {source.brief && (
                <details className="outcome-method">
                  <summary>One-page reading aid</summary>
                  <a
                    href={`/api/precedents/${encodeURIComponent(source.id)}/brief.pdf`}
                  >
                    Download one-pager
                  </a>
                  <BriefSections brief={source.brief} />
                </details>
              )}
              {source.hit && (
                <>
                  <p className="caption">
                    Matched passage, original characters {source.hit.start}–
                    {source.hit.end}
                  </p>
                  <blockquote className="corpus-excerpt">
                    {source.hit.text}
                  </blockquote>
                </>
              )}
              <details open>
                <summary>Full saved document text</summary>
                <pre tabIndex={0}>{source.text}</pre>
              </details>
            </section>
          )}
          <details className="outcome-method">
            <summary>
              Browse all {catalogue.documents.length} saved references
            </summary>
            <ul>
              {catalogue.documents.map((d) => (
                <li key={d.id}>
                  <button
                    onClick={() =>
                      read({
                        document: d,
                      })
                    }
                  >
                    {d.title}
                  </button>{" "}
                  · {d.state} · {d.year}
                </li>
              ))}
            </ul>
          </details>
        </>
      ) : (
        catalogue && <p className="empty">{catalogue.message}</p>
      )}
      {!!catalogue?.directories?.length && (
        <details className="outcome-method">
          <summary>Expand your research in official case directories</summary>
          {catalogue.directories.map((d) => (
            <p key={d.url}>
              <a href={d.url} target="_blank" rel="noreferrer">
                {d.title}
              </a>{" "}
              — {d.scope}
            </p>
          ))}
        </details>
      )}
    </section>
  );
}
