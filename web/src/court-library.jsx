import React, { useEffect, useState } from "react";
import { requestJSON } from "./library-client.mjs";

export function CourtJudgments({ navigate }) {
  const [catalogue, setCatalogue] = useState(null),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    requestJSON("/api/precedents")
      .then((d) => {
        if (active) setCatalogue(d);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, []);
  if (error)
    return <p role="alert">Court reference library unavailable: {error}</p>;
  if (!catalogue) return <p role="status">Opening saved court judgments…</p>;
  const judgments = catalogue.documents.filter(
    (d) => d.kind === "court_judgment",
  );
  return (
    <section className="court-library" aria-labelledby="court-library-heading">
      <div className="collection-heading">
        <div>
          <p className="eyebrow">Original decisions · saved locally</p>
          <h2 id="court-library-heading">Court judgments</h2>
          <p>
            Start with the judgment. Read the charge, the consequences and the
            court’s reasoning.
          </p>
        </div>
      </div>
      {judgments.length ? (
        <div className="court-cards">
          {judgments.map((d) => (
            <article key={d.id}>
              <p className="eyebrow">
                {d.state} · {d.year}
              </p>
              <h3>
                <a
                  href={`/judgments/${d.id}`}
                  onClick={(e) => {
                    e.preventDefault();
                    navigate(`/judgments/${d.id}`);
                  }}
                >
                  {d.title}
                </a>
              </h3>
              {d.issue && <p className="caption">{d.issue}</p>}
              {d.charge_summary && (
                <p className="card-charge">
                  <strong>Charge</strong> {d.charge_summary}
                </p>
              )}
              {d.outcome && <p className="case-status">{d.outcome}</p>}
              <p className="caption">
                Court judgment · ECHR Article 6(2)
                {d.one_pager ? " · one-pager available" : ""}
              </p>
              <button onClick={() => navigate(`/judgments/${d.id}`)}>
                Open judgment &amp; case summary
              </button>
            </article>
          ))}
        </div>
      ) : (
        <p className="empty">
          No court judgments have been installed yet. Build the court source
          collection to add full decisions.
        </p>
      )}
      <p className="caption">
        UN Human Rights Committee decisions and Working Group opinions are in{" "}
        <a
          href="/international"
          onClick={(e) => {
            e.preventDefault();
            navigate("/international");
          }}
        >
          International review
        </a>
        . Their forum and legal standard stay distinct.
      </p>
    </section>
  );
}

export function BriefSections({ brief }) {
  if (!brief) return null;
  return (
    <div className="brief-sections">
      <p className="caption">{brief.method}</p>
      {brief.sections.map((s, i) => (
        <section key={i}>
          <h3>{s.label}</h3>
          <p>{s.summary}</p>
          <details>
            <summary>{s.pinpoint} · supporting source passage</summary>
            <blockquote className="corpus-excerpt">{s.quote}</blockquote>
          </details>
        </section>
      ))}
    </div>
  );
}

export function CourtJudgment({ id, navigate, onImport, refresh }) {
  const [doc, setDoc] = useState(null),
    [error, setError] = useState(""),
    [staging, setStaging] = useState(false);
  useEffect(() => {
    let active = true;
    setDoc(null);
    setError("");
    requestJSON(`/api/precedents/${encodeURIComponent(id)}`)
      .then((d) => {
        if (active) setDoc(d);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [id]);
  async function stage() {
    setStaging(true);
    setError("");
    try {
      await requestJSON(`/api/precedents/${encodeURIComponent(id)}/stage`, {});
      await refresh();
      onImport();
    } catch (e) {
      setError(e.message);
    } finally {
      setStaging(false);
    }
  }
  if (!doc)
    return (
      <div className="empty" role={error ? "alert" : "status"}>
        {error || "Opening the saved judgment…"}
        <p>
          <button onClick={() => navigate("/")}>Back to collection</button>
        </p>
      </div>
    );
  if (doc.kind !== "court_judgment")
    return (
      <p role="alert">
        This source is not a court judgment.{" "}
        <button onClick={() => navigate("/international")}>
          Open the reference library
        </button>
      </p>
    );
  return (
    <article className="judgment-review">
      <button onClick={() => navigate("/")}>← Case collection</button>
      <p className="eyebrow">
        Court judgment · {doc.state} · {doc.year}
      </p>
      <h1>{doc.title}</h1>
      <p>{doc.body}</p>
      <div className="focus-ribbon">
        <span>Presumption of innocence</span>
        <strong>ECHR Article 6(2)</strong>
      </div>
      <p className="caption">
        Historical reference case. This judgment does not determine a new
        client’s liability or prospects of international review.
      </p>
      {error && <p role="alert">{error}</p>}
      <div className="judgment-actions">
        {doc.local_pdf && (
          <a
            className="primary"
            href={`/api/precedents/${encodeURIComponent(id)}/original`}
          >
            Download full judgment
          </a>
        )}
        {doc.brief && (
          <a href={`/api/precedents/${encodeURIComponent(id)}/brief.pdf`}>
            Download one-pager
          </a>
        )}
        <a href={doc.url} target="_blank" rel="noreferrer">
          Official source copy
        </a>
      </div>
      <BriefSections brief={doc.brief} />
      {!doc.brief && (
        <p className="empty">
          A verified case summary is not available yet. Read the full judgment
          below.
        </p>
      )}
      <details className="corpus-reader">
        <summary>Read the complete saved judgment text</summary>
        <p className="caption">
          {doc.attribution} · source text checksum{" "}
          {doc.text_sha256.slice(0, 16)}
        </p>
        <pre tabIndex={0}>{doc.text}</pre>
      </details>
      <section className="judgment-next">
        <h2>Use this authority in a case review</h2>
        <p>
          Compare the facts and procedural stage with your client’s matter.
          Confirm the defendant, court and document role when adding the
          original PDF.
        </p>
        {doc.local_pdf && (
          <button disabled={staging} onClick={stage}>
            {staging ? "Adding original…" : "Add original to local case intake"}
          </button>
        )}
        <button onClick={() => navigate("/international")}>
          Research UN review decisions
        </button>
      </section>
    </article>
  );
}
