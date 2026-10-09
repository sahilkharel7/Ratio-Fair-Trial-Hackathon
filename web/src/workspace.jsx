import React, { useContext, useRef, useEffect, useState } from "react";
import { matchesQuery, firstMatch } from "./research.mjs";

export function Icon({ name, size = 18 }) {
  const paths = {
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="m16 16 4 4" />
      </>
    ),
    folder: <path d="M3 6h6l2 2h10v12H3z M3 6V4h6l2 2" />,
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    file: (
      <>
        <path d="M5 3h9l5 5v13H5z M14 3v5h5 M8 12h8 M8 16h6" />
      </>
    ),
    grid: (
      <>
        <path d="M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z" />
      </>
    ),
    scale: (
      <>
        <path d="M12 3v18 M5 21h14 M4 7h16 M6 7l-4 7h8z M18 7l-4 7h8z" />
      </>
    ),
    timeline: (
      <>
        <path d="M5 3v18 M8 5h12 M8 12h8 M8 19h12" />
        <circle cx="5" cy="5" r="1.5" />
        <circle cx="5" cy="12" r="1.5" />
        <circle cx="5" cy="19" r="1.5" />
      </>
    ),
    compare: (
      <>
        <path d="M3 4h7v16H3z M14 4h7v16h-7z M6 8h1 M6 12h1 M17 8h1 M17 12h1" />
      </>
    ),
    bookmark: <path d="M6 3h12v18l-6-4-6 4z" />,
    arrow: <path d="M4 12h16 M14 6l6 6-6 6" />,
    check: <path d="m5 12 4 4L19 6" />,
    download: (
      <>
        <path d="M12 3v12 m-5-5 5 5 5-5 M4 16v5h16v-5" />
      </>
    ),
    close: <path d="m6 6 12 12 M6 18 18 6" />,
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] || paths.file}
    </svg>
  );
}
export function ResearchBar({ navigate, onSearch, query }) {
  const [value, setValue] = useState(query || "");
  const ref = useRef(null);
  useEffect(() => setValue(query || ""), [query]);
  useEffect(() => {
    const handler = (e) => {
      if (
        e.key === "/" &&
        !["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName) &&
        !document.querySelector("dialog[open]")
      ) {
        e.preventDefault();
        ref.current.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  return (
    <div className="research-band">
      <form
        className="global-search"
        onSubmit={(e) => {
          e.preventDefault();
          onSearch(value.trim());
        }}
      >
        <span className="search-scope">
          <Icon name="folder" /> Current matter
        </span>
        <label className="sr-only" htmlFor="global-search">
          Search this matter
        </label>
        <input
          ref={ref}
          id="global-search"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Search the record, a finding, or a provision…"
          type="search"
        />
        <kbd aria-hidden="true">/</kbd>
        <button className="primary" aria-label="Search matter">
          <Icon name="search" />
          <span>Search</span>
        </button>
      </form>
      <button className="search-browse" onClick={() => navigate("/research")}>
        Browse record <Icon name="arrow" size={15} />
      </button>
    </div>
  );
}
export function MatterHeader({ data, navigate, download }) {
  return (
    <div className="matter-header">
      <div>
        <div className="matter-kicker">
          <span className="matter-reference">MATTER / VENN–2025</span>
          <span className="synthetic-pill">Synthetic record</span>
        </div>
        <h1>{data.record.meta.title}</h1>
        <p>
          {data.record.meta.court}
          <span>Criminal proceedings</span>
        </p>
      </div>
      <button className="export-button" onClick={download}>
        <Icon name="download" /> Export review
      </button>
    </div>
  );
}
export function ReviewQueue({ data, navigate, context }) {
  const { openSource } = useContext(context);
  const [filter, setFilter] = useState("all");
  const priority = {
    evidence_of_violation: 0,
    exceeds_benchmark: 1,
    unaddressed_argument: 2,
    verbatim_reuse: 3,
    close_paraphrase: 4,
    evidence_of_compliance: 5,
  };
  const items = [
    ...data.analysis.absence.flags,
    ...data.analysis.clock.flags,
    ...data.analysis.reuse.flags,
  ].sort((a, b) => (priority[a.status] ?? 4) - (priority[b.status] ?? 4));
  const filtered = items.filter((f) => filter === "all" || f.module === filter);
  const routes = { absence: "/coverage", clock: "/timeline", reuse: "/reuse" };
  const names = {
    absence: "Rights coverage",
    clock: "Procedure",
    reuse: "Reasoning",
  };
  return (
    <section className="review-queue">
      <div className="section-head">
        <div>
          <h2>Findings for legal review</h2>
          <p>Issues linked to their supporting sources.</p>
        </div>
        <span className="count-label">{filtered.length} findings</span>
      </div>
      <div className="queue-tabs" role="group" aria-label="Filter findings">
        {[
          ["all", "All findings"],
          ["absence", "Rights coverage"],
          ["clock", "Procedure"],
          ["reuse", "Reasoning"],
        ].map(([id, label]) => (
          <button
            key={id}
            aria-pressed={filter === id}
            className={filter === id ? "active" : ""}
            onClick={() => setFilter(id)}
          >
            {label}
            <span>
              {id === "all"
                ? items.length
                : items.filter((f) => f.module === id).length}
            </span>
          </button>
        ))}
      </div>
      <div className="queue-list">
        {filtered.map((f, index) => (
          <article className="queue-item" key={f.id}>
            <div
              className={`issue-symbol ${f.module} ${f.status === "evidence_of_compliance" ? "compliant" : ""}`}
            >
              <Icon
                name={
                  f.module === "clock"
                    ? "clock"
                    : f.module === "absence"
                      ? "scale"
                      : "compare"
                }
              />
            </div>
            <div className="issue-content">
              <div className="issue-meta">
                <span>{names[f.module]}</span>
                <span>{data.messages.labels[f.status]}</span>
              </div>
              <a
                className="issue-title"
                href={
                  routes[f.module] +
                  (f.module === "absence" ? `?guarantee=${f.standard_id}` : "")
                }
                onClick={(e) => {
                  e.preventDefault();
                  navigate(
                    routes[f.module] +
                      (f.module === "absence"
                        ? `?guarantee=${f.standard_id}`
                        : ""),
                  );
                }}
              >
                {f.standard_label}
                {f.module === "reuse" && f.status !== "unaddressed_argument"
                  ? ` · Passage ${data.analysis.reuse.flags.filter((x) => x.status !== "unaddressed_argument").findIndex((x) => x.id === f.id) + 1}`
                  : ""}
              </a>
              <p>{f.message}</p>
              <div className="issue-foot">
                <span>
                  <Icon name="file" size={13} />
                  {f.evidence.length} source passages
                </span>
                <span>Requires legal judgment</span>
              </div>
            </div>
            <button
              className="text-action"
              onClick={() => openSource(f.evidence[0].span, f.standard_label)}
            >
              View source <Icon name="arrow" size={15} />
            </button>
          </article>
        ))}
      </div>
    </section>
  );
}
export function ReviewRail({ data, navigate, context }) {
  const { recent, saved, notes, setNotes } = useContext(context);
  const docs = data.record.documents;
  return (
    <aside className="review-rail">
      <section className="rail-panel">
        <div className="rail-heading">
          <Icon name="folder" />
          <h2>Matter details</h2>
        </div>
        <dl>
          <dt>Matter reference</dt>
          <dd>VENN–2025</dd>
          <dt>Presiding judge</dt>
          <dd>{data.record.meta.presiding_judge}</dd>
          <dt>Charge</dt>
          <dd>{data.record.meta.charge_type}</dd>
          <dt>Record</dt>
          <dd>{docs.length} source documents</dd>
          <dt>Review framework</dt>
          <dd>ICCPR Articles 9 &amp; 14</dd>
        </dl>
        <button className="text-action" onClick={() => navigate("/judges")}>
          View judicial history <Icon name="arrow" size={15} />
        </button>
      </section>
      <section className="rail-panel">
        <div className="rail-heading">
          <Icon name="bookmark" />
          <h2>Your working file</h2>
        </div>
        <div className="saved-summary">
          <strong>{saved.length}</strong>
          <span>saved documents &amp; passages</span>
        </div>
        <button className="text-action" onClick={() => navigate("/saved")}>
          Open saved research <Icon name="arrow" size={15} />
        </button>
        <label htmlFor="matter-notes">Matter notes</label>
        <textarea
          id="matter-notes"
          value={notes.matter || ""}
          onChange={(e) => setNotes({ ...notes, matter: e.target.value })}
          placeholder="Record a question or your next step…"
          rows={4}
        />
        <small>Saved in this browser. Use only demo notes.</small>
      </section>
      <section className="rail-panel research-tip">
        <p className="eyebrow">Document comparison</p>
        <h3>Judgment and indictment</h3>
        <p>
          Compare the judgment with the indictment, with matching passages
          linked on both sides.
        </p>
        <button className="text-action" onClick={() => navigate("/reuse")}>
          Compare documents <Icon name="arrow" size={15} />
        </button>
      </section>
    </aside>
  );
}
export function Research({ data, query, navigate, context, scope = "all" }) {
  const { openSource, openDocument, toggleSaved, isSaved } =
    useContext(context);
  const [type, setType] = useState("all"),
    [sort, setSort] = useState("record");
  useEffect(() => setType("all"), [query, scope]);
  const allFlags = [
    ...data.analysis.absence.flags,
    ...data.analysis.clock.flags,
    ...data.analysis.reuse.flags,
  ];
  const docs = data.record.documents.filter((d) =>
    matchesQuery(`${d.title}\n${d.text}`, query),
  );
  const findings = allFlags.filter((f) =>
    matchesQuery(
      [
        f.standard_label,
        f.message,
        f.citation,
        ...f.evidence.map((e) => e.span.text),
      ].join("\n"),
      query,
    ),
  );
  let visible = docs.filter((d) => type === "all" || d.type === type);
  if (sort === "newest")
    visible = [...visible].sort((a, b) =>
      (b.date || "").localeCompare(a.date || ""),
    );
  const routes = { absence: "/coverage", clock: "/timeline", reuse: "/reuse" };
  return (
    <>
      <div className="page-heading">
        <div>
          <p className="eyebrow">Matter research</p>
          <h1>{query ? `Results for ${query}` : "Explore the case record"}</h1>
          <p className="caption">
            {query
              ? "All words must appear; use quotation marks for an exact phrase."
              : "Original documents and recorded findings, together in one place."}
          </p>
        </div>
      </div>
      <div className="results-layout">
        <aside className="result-filters">
          <h2>Narrow your results</h2>
          <p className="filter-label">Content</p>
          {[
            ["all", "All content", docs.length + findings.length],
            ["documents", "Source documents", docs.length],
            ["findings", "Analysis findings", findings.length],
          ].map(([id, label, count]) => (
            <a
              key={id}
              className={scope === id ? "selected" : ""}
              href={`/research?q=${encodeURIComponent(query)}&scope=${id}`}
              onClick={(e) => {
                e.preventDefault();
                navigate(
                  `/research?q=${encodeURIComponent(query)}&scope=${id}`,
                );
              }}
            >
              {label}
              <span>{count}</span>
            </a>
          ))}
          {scope !== "findings" && (
            <>
              <p className="filter-label">Document type</p>
              {[
                ["all", "All types"],
                ["monitoring_note", "Monitoring notes"],
                ["indictment", "Indictment"],
                ["judgment", "Judgment"],
              ].map(([id, label]) => (
                <button
                  key={id}
                  className={type === id ? "selected" : ""}
                  aria-pressed={type === id}
                  onClick={() => setType(id)}
                >
                  {label}
                  <span>
                    {id === "all"
                      ? docs.length
                      : docs.filter((d) => d.type === id).length}
                  </span>
                </button>
              ))}
            </>
          )}
          <div className="filter-note">
            <Icon name="check" />
            <p>
              Source passages are checked against the original text before they
              are highlighted.
            </p>
          </div>
        </aside>
        <div className="research-results">
          <div className="result-toolbar">
            <strong>
              {(scope === "findings" ? 0 : visible.length) +
                (scope === "documents" || type !== "all"
                  ? 0
                  : findings.length)}{" "}
              result
              {(scope === "findings" ? 0 : visible.length) +
                (scope === "documents" || type !== "all"
                  ? 0
                  : findings.length) ===
              1
                ? ""
                : "s"}
            </strong>
            {scope !== "findings" && (
              <>
                <label className="sr-only" htmlFor="sort-results">
                  Sort documents
                </label>
                <select
                  id="sort-results"
                  value={sort}
                  onChange={(e) => setSort(e.target.value)}
                >
                  <option value="record">Record order</option>
                  <option value="newest">Newest documents</option>
                </select>
              </>
            )}
          </div>
          {scope !== "findings" &&
            visible.map((doc, index) => {
              const span = firstMatch(doc, query),
                start = span ? Math.max(0, span.start - 100) : 0;
              const excerpt = Array.from(doc.text)
                .slice(start, start + 360)
                .join("");
              return (
                <article className="search-result" key={doc.id}>
                  <div className="result-number">{index + 1}</div>
                  <div className="result-content">
                    <div className="issue-meta">
                      <span>{doc.type.replaceAll("_", " ")}</span>
                      <span>Original source</span>
                    </div>
                    <button
                      className="result-title"
                      onClick={() =>
                        span
                          ? openSource(span, "Search result")
                          : openDocument(doc)
                      }
                    >
                      {doc.title}
                    </button>
                    <p className="result-citation">
                      VENN–2025 · {data.record.meta.court}
                    </p>
                    <p className="result-excerpt">
                      {start > 0 ? "…" : ""}
                      {span ? (
                        <>
                          {Array.from(doc.text)
                            .slice(start, span.start)
                            .join("")}
                          <mark>{span.text}</mark>
                          {Array.from(doc.text)
                            .slice(
                              span.end,
                              Math.max(start + 360, span.end + 100),
                            )
                            .join("")}
                        </>
                      ) : (
                        excerpt
                      )}
                      …
                    </p>
                    <div className="result-actions">
                      <button
                        className="text-action"
                        onClick={() => openDocument(doc)}
                      >
                        Read document <Icon name="arrow" size={14} />
                      </button>
                      <button
                        className="text-action"
                        aria-pressed={isSaved(doc)}
                        onClick={() => toggleSaved(doc)}
                      >
                        <Icon
                          name={isSaved(doc) ? "check" : "bookmark"}
                          size={15}
                        />
                        {isSaved(doc)
                          ? "Saved to working file"
                          : "Save document"}
                      </button>
                    </div>
                  </div>
                </article>
              );
            })}
          {scope !== "documents" &&
            type === "all" &&
            findings.map((f) => (
              <article className="search-result finding-result" key={f.id}>
                <div className="result-number">
                  <Icon name="scale" />
                </div>
                <div className="result-content">
                  <div className="issue-meta">
                    <span>Analysis finding</span>
                    <span>{data.messages.labels[f.status]}</span>
                  </div>
                  <a
                    className="result-title"
                    href={
                      routes[f.module] +
                      (f.module === "absence"
                        ? `?guarantee=${f.standard_id}`
                        : "")
                    }
                    onClick={(e) => {
                      e.preventDefault();
                      navigate(
                        routes[f.module] +
                          (f.module === "absence"
                            ? `?guarantee=${f.standard_id}`
                            : ""),
                      );
                    }}
                  >
                    {f.standard_label}
                  </a>
                  <p className="result-citation">{f.citation}</p>
                  <p>{f.message}</p>
                  <button
                    className="text-action"
                    onClick={() =>
                      openSource(f.evidence[0].span, f.standard_label)
                    }
                  >
                    Read supporting source <Icon name="arrow" size={14} />
                  </button>
                </div>
              </article>
            ))}
          {!(scope === "findings"
            ? findings.length
            : visible.length +
              (scope === "all" && type === "all" ? findings.length : 0)) && (
            <div className="empty">
              <h2>No matching results</h2>
              <p>
                Try fewer words, a different phrase, or another content filter.
              </p>
              <button onClick={() => navigate("/research")}>
                Browse the full record
              </button>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
export function SavedWorkspace({ context, navigate, tab }) {
  const { saved, recent, documents, openSource, openDocument, toggleSaved } =
    useContext(context);
  return (
    <>
      <div className="page-heading">
        <p className="eyebrow">Your workspace</p>
        <h1>{tab === "history" ? "Research history" : "Saved research"}</h1>
        <p>Resume your review with the sources you have collected.</p>
        <p className="caption">
          Stored in this browser only. Saved passages and history are not shared
          with other users.
        </p>
      </div>
      <div className="queue-tabs workspace-tabs">
        <button
          className={tab !== "history" ? "active" : ""}
          onClick={() => navigate("/saved")}
        >
          Working file <span>{saved.length}</span>
        </button>
        <button
          className={tab === "history" ? "active" : ""}
          onClick={() => navigate("/saved?tab=history")}
        >
          History <span>{recent.length}</span>
        </button>
      </div>
      {tab === "history" ? (
        recent.length ? (
          <div className="saved-list">
            {recent.map((item, i) => (
              <article className="search-result" key={item.key}>
                <Icon name={item.query !== undefined ? "search" : "file"} />
                <div className="result-content">
                  <div className="issue-meta">
                    <span>
                      {item.query !== undefined ? "Search" : "Document viewed"}
                    </span>
                    <span>{new Date(item.at).toLocaleString()}</span>
                  </div>
                  <button
                    className="result-title"
                    onClick={() =>
                      item.query !== undefined
                        ? navigate(
                            `/research?q=${encodeURIComponent(item.query)}`,
                          )
                        : documents[item.doc_id] &&
                          openDocument(documents[item.doc_id])
                    }
                  >
                    {item.query !== undefined
                      ? item.query || "Browse case record"
                      : documents[item.doc_id]?.title || "Source document"}
                  </button>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty">
            <h2>Your research history starts here.</h2>
            <p>Search the record or open a source to see it here.</p>
            <button onClick={() => navigate("/research")}>
              Explore the record
            </button>
          </div>
        )
      ) : saved.length ? (
        <div className="saved-list">
          {saved.map((item) => {
            const doc = documents[item.doc_id];
            return doc ? (
              <article className="search-result" key={item.key}>
                <Icon name="bookmark" />
                <div className="result-content">
                  <div className="issue-meta">
                    <span>
                      {item.span ? "Saved passage" : "Saved document"}
                    </span>
                    <span>Synthetic record</span>
                  </div>
                  <button
                    className="result-title"
                    onClick={() =>
                      item.span
                        ? openSource(item.span, "Saved passage")
                        : openDocument(doc)
                    }
                  >
                    {doc.title}
                  </button>
                  {item.span && (
                    <blockquote className="saved-quote">
                      “{item.span.text}”
                    </blockquote>
                  )}
                  <p className="result-citation">{item.citation}</p>
                  <button
                    className="text-action"
                    onClick={() => toggleSaved(doc, item.span)}
                  >
                    Remove from working file
                  </button>
                </div>
              </article>
            ) : null;
          })}
        </div>
      ) : (
        <div className="empty">
          <Icon name="folder" size={28} />
          <h2>Build your working file.</h2>
          <p>
            Save a document from search, or keep an exact passage from the
            source reader.
          </p>
          <button className="primary" onClick={() => navigate("/research")}>
            Browse source documents
          </button>
        </div>
      )}
    </>
  );
}
