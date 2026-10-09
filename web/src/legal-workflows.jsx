import React, { useContext, useState } from "react";
import { Icon } from "./workspace.jsx";
import {
  allFindings,
  currentReviews,
  moduleNames,
  moduleRoutes,
  reviewError,
  makeReview,
  reviewWorksheet,
  downloadFile,
} from "./workflow.mjs";

const percent = (value) =>
  value == null ? "—" : `${Math.round(value * 100)}%`;
const date = (value) =>
  value
    ? new Date(value).toLocaleDateString("en-GB", {
        day: "numeric",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      })
    : "Not recorded";

function PageHeader({ title, kicker, description, children }) {
  return (
    <div className="legal-page-header">
      <div>
        <p className="eyebrow">{kicker || "Matter analysis"}</p>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {children}
    </div>
  );
}
export function LegalContext({ data, flag, question }) {
  const refs = flag
    ? data.jurisprudence.by_finding[flag.id] || []
    : data.jurisprudence.by_follow_up[question] || [];
  if (!refs.length) return null;
  return (
    <details className="authority-context">
      <summary>
        Linked jurisprudence <span>{refs.length}</span>
      </summary>
      <p className="caption">{data.messages.notes.jurisprudence_caveat}</p>
      {refs.map((ref) => (
        <article className="authority-quote" key={ref.id}>
          <h4>{ref.citation}</h4>
          <blockquote>{ref.shown}</blockquote>
        </article>
      ))}
    </details>
  );
}
export function StateReply({
  data,
  flag,
  context,
  Evidence,
  expanded = false,
}) {
  if (!flag) return null;
  const reply = data.analysis.steelman?.replies.find(
    (r) => r.flag_id === flag.id,
  );
  if (!reply) return null;
  const catalogue = data.steelman_catalogue.grounds;
  const grounds = Object.fromEntries(
    [...(catalogue.all || []), ...(catalogue[reply.standard_id] || [])].map(
      (g) => [g.id, g],
    ),
  );
  return (
    <details
      className="authority-context state-reply"
      open={expanded || undefined}
    >
      <summary>
        Possible State reply{" "}
        <span>
          {reply.arguments.length} supported argument
          {reply.arguments.length === 1 ? "" : "s"}
        </span>
      </summary>
      <div className="model-caveat">
        Model-generated · Unverified · Grounds need legal review
      </div>
      <p className="caption">{data.messages.notes.steelman_caveat}</p>
      {!reply.checked && (
        <p className="warning">
          {reply.note || "No usable model response was recorded."}
        </p>
      )}
      {reply.checked && !reply.arguments.length && (
        <p className="caption">
          No offered ground was supported by the record passages shown to the
          model.
        </p>
      )}
      {reply.arguments.map((a, i) => (
        <article className="state-argument" key={`${a.ground_id}-${i}`}>
          <h4>{grounds[a.ground_id]?.label || a.ground_id}</h4>
          {grounds[a.ground_id]?.quote && (
            <blockquote className="ground-quote">
              “{grounds[a.ground_id].quote}”{" "}
              <cite>{grounds[a.ground_id].pinpoint}</cite>
            </blockquote>
          )}
          <p>
            <strong>Local model:</strong> {a.argument}
          </p>
          <Evidence
            items={[{ role: "mention", span: a.span }]}
            heading="Source supporting the State’s possible reply"
          />
        </article>
      ))}
      {reply.unsupported_grounds.length > 0 && (
        <p className="caption">
          <strong>Not supported by the record:</strong>{" "}
          {reply.unsupported_grounds
            .map((id) => grounds[id]?.label || id)
            .join("; ")}
          .
        </p>
      )}
      {reply.dropped.length > 0 && (
        <details>
          <summary>
            Arguments dropped by the checks ({reply.dropped.length})
          </summary>
          <ul className="caption">
            {reply.dropped.map((reason, i) => (
              <li key={i}>{reason}</li>
            ))}
          </ul>
        </details>
      )}
    </details>
  );
}
export function FindingActions({ data, flag, navigate }) {
  if (!flag) return null;
  return (
    <div className="finding-actions">
      <button
        className="text-action"
        onClick={() => navigate(`/review?finding=${flag.id}`)}
      >
        <Icon name="check" size={15} />
        Review this finding
      </button>
      <button
        className="text-action"
        onClick={() => navigate(`/jurisprudence?finding=${flag.id}`)}
      >
        Research the standard <Icon name="arrow" size={14} />
      </button>
    </div>
  );
}
export function Renewals({ data, context, Evidence, Badge, navigate }) {
  const result = data.analysis.renewal;
  const { openDocument, documents } = useContext(context);
  return (
    <>
      <PageHeader
        title="Detention renewals"
        description={data.messages.notes.renewal_intro}
      />
      <div className="metrics">
        {[
          [
            "Detention orders",
            result.orders.length,
            "Original orders in the record",
          ],
          [
            "Repeated grounds",
            result.flags.filter((f) => f.status === "repeated_grounds").length,
            "A prompt to review re-examination",
          ],
          [
            "Gaps between orders",
            result.gaps.length,
            "Measured from the recorded dates",
          ],
        ].map(([label, value, note]) => (
          <div className="metric" key={label}>
            <div>{label}</div>
            <strong>{value}</strong>
            <small>{note}</small>
          </div>
        ))}
      </div>
      <section className="section">
        <h2>Order chronology</h2>
        <p className="section-caption">
          Dates and comparison measures come from the local analysis pipeline.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Order</th>
                <th>Made</th>
                <th>Detention until</th>
                <th>Repeated grounds</th>
                <th>New passages</th>
              </tr>
            </thead>
            <tbody>
              {result.orders.map((order) => (
                <tr key={order.doc_id}>
                  <td>
                    <button
                      className="text-action"
                      onClick={() => openDocument(documents[order.doc_id])}
                    >
                      {order.title}
                    </button>
                  </td>
                  <td>{date(order.date)}</td>
                  <td>{date(order.until)}</td>
                  <td>{percent(order.share_repeated)}</td>
                  <td>
                    {order.share_repeated == null
                      ? "—"
                      : `${order.passages_new} of ${order.passages_compared}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {result.notes.map((note, i) => (
          <p className="caption" key={i}>
            {note}
          </p>
        ))}
      </section>
      <section className="section">
        <h2>Gaps in the order record</h2>
        {result.gaps.length ? (
          result.gaps.map((gap) => {
            const flag = result.flags.find((f) => f.id === gap.flag_id);
            return (
              <article className="panel" key={gap.id}>
                <Badge status={flag.status} messages={data.messages} />
                <p>{flag.message}</p>
                <p className="caption">{flag.citation}</p>
                <Evidence items={flag.evidence} heading={flag.standard_label} />
                <LegalContext data={data} flag={flag} />
                <StateReply
                  data={data}
                  flag={flag}
                  context={context}
                  Evidence={Evidence}
                />
                <FindingActions data={data} flag={flag} navigate={navigate} />
              </article>
            );
          })
        ) : (
          <p>No gaps were found between the recorded orders.</p>
        )}
      </section>
      <section className="section">
        <h2>Compare the grounds</h2>
        <p className="section-caption">
          Quoted law, the prosecution’s request, captions and operative text are
          excluded. Repetition is not a legal conclusion.
        </p>
        <div className="stack">
          {result.orders
            .filter((o) => o.share_repeated != null)
            .map((order) => (
              <RenewalOrder
                key={order.doc_id}
                data={data}
                order={order}
                context={context}
                Evidence={Evidence}
                Badge={Badge}
                navigate={navigate}
              />
            ))}
        </div>
      </section>
    </>
  );
}
function RenewalOrder({ data, order, context, Evidence, Badge, navigate }) {
  const { documents } = useContext(context);
  const sources = [...new Set(order.pairs.map((pair) => pair.earlier.doc_id))];
  const [earlier, setEarlier] = useState(sources[0]);
  const flag = data.analysis.renewal.flags.find((f) => f.id === order.flag_id);
  return (
    <article className="panel renewal-order">
      <div className="row-heading">
        <h3>{order.title}</h3>
        {flag && <Badge status={flag.status} messages={data.messages} />}
      </div>
      <p>
        {flag?.message ||
          `${percent(order.share_repeated)} of its grounds repeat earlier orders; ${order.passages_new} of ${order.passages_compared} passages long enough to compare are new.`}
      </p>
      {sources.length > 1 && (
        <>
          <label htmlFor={`earlier-${order.doc_id}`}>
            Compare with an earlier order
          </label>
          <select
            id={`earlier-${order.doc_id}`}
            value={earlier}
            onChange={(e) => setEarlier(e.target.value)}
          >
            {sources.map((id) => (
              <option key={id} value={id}>
                {documents[id].title}
              </option>
            ))}
          </select>
        </>
      )}
      {sources.length > 0 && (
        <>
          <p className="legend">
            <mark className="ratio-verbatim">Repeated word for word</mark>
            <mark className="ratio-paraphrase">Close paraphrase</mark>
            <span>Grey text is excluded</span>
          </p>
          <div className="split">
            {[order.doc_id, earlier].map((id) => (
              <div key={id}>
                <h4>
                  {id === order.doc_id ? "Later order" : "Earlier order"} ·{" "}
                  {documents[id].title}
                </h4>
                <div
                  className="panel document-panel renewal-document"
                  dangerouslySetInnerHTML={{
                    __html: data.renewal_comparison[order.doc_id][id].grounds,
                  }}
                />
              </div>
            ))}
          </div>
        </>
      )}
      {flag && (
        <>
          <Evidence items={flag.evidence} heading={flag.standard_label} />
          <LegalContext data={data} flag={flag} />
          <StateReply
            data={data}
            flag={flag}
            context={context}
            Evidence={Evidence}
          />
          <FindingActions data={data} flag={flag} navigate={navigate} />
        </>
      )}
    </article>
  );
}
export function Jurisprudence({
  data,
  context,
  Evidence,
  Badge,
  navigate,
  selectedId,
}) {
  const findings = allFindings(data),
    [selected, setSelected] = useState(selectedId || "all"),
    [query, setQuery] = useState("");
  const selectedFlag = findings.find((f) => f.id === selected);
  const entries = (
    selectedFlag
      ? data.jurisprudence.by_finding[selectedFlag.id] || []
      : data.jurisprudence.entries
  ).filter((e) =>
    `${e.citation} ${e.shown}`.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <>
      <PageHeader
        kicker="Legal research"
        title="Jurisprudence"
        description={data.messages.notes.jurisprudence_intro}
      />
      <p className="authority-disclaimer">
        {data.messages.notes.jurisprudence_caveat}
      </p>
      <div className="authority-toolbar">
        <div>
          <label htmlFor="authority-finding">Standard or finding</label>
          <select
            id="authority-finding"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            <option value="all">
              Whole reviewed corpus · {data.jurisprudence.entries.length}{" "}
              entries
            </option>
            {findings.map((f, i) => (
              <option key={f.id} value={f.id}>
                {i + 1}. {f.standard_label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="authority-search">
            Search citations and passages
          </label>
          <input
            id="authority-search"
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="A communication number, a case, or exact words…"
          />
        </div>
      </div>
      {selectedFlag && (
        <section className="panel selected-finding">
          <Badge messages={data.messages} status={selectedFlag.status} />
          <h3>{selectedFlag.standard_label}</h3>
          <p>{selectedFlag.message}</p>
          {selectedFlag.module === "judges" && (
            <p className="warning">{data.messages.notes.selection_bias}</p>
          )}
          <Evidence
            items={selectedFlag.evidence.slice(0, 1)}
            heading={selectedFlag.standard_label}
          />
          <FindingActions data={data} flag={selectedFlag} navigate={navigate} />
        </section>
      )}
      <div className="section-head">
        <h2>
          {entries.length} linked entr{entries.length === 1 ? "y" : "ies"}
        </h2>
        <span className="caption">
          Corpus last checked: {data.jurisprudence.checked}
        </span>
      </div>
      <div className="stack">
        {entries.map((entry) => (
          <article className="panel authority-card" key={entry.id}>
            <p className="eyebrow">
              {entry.kind === "general_comment"
                ? "Human Rights Committee · General Comment"
                : "Committee decision · Cited by the General Comment"}
            </p>
            <h2>{entry.citation}</h2>
            <blockquote>{entry.shown}</blockquote>
            <p className="caption">
              {entry.kind === "general_comment"
                ? "Quoted in the General Comment’s own words."
                : "Shown for the proposition for which the General Comment cites the decision."}
            </p>
          </article>
        ))}
      </div>
      {!entries.length && (
        <p className="empty">No corpus entry matches this selection.</p>
      )}
      <section className="section">
        <h2>Questions for the monitor</h2>
        <p className="section-caption">
          Missing evidence remains a follow-up question. These are linked to
          General Comment paragraphs, rather than treated as findings.
        </p>
        {data.analysis.absence.follow_ups.map((question) => (
          <article
            className="panel follow-up-authority"
            key={question.rubric_id}
          >
            <h3>{question.question}</h3>
            <Evidence
              items={question.context}
              heading="Follow-up context; not evidence of a violation"
            />
            <LegalContext data={data} question={question.rubric_id} />
          </article>
        ))}
      </section>
    </>
  );
}
export function StateReplies({ data, context, Evidence, Badge, navigate }) {
  const replies = data.analysis.steelman?.replies || [],
    findings = allFindings(data);
  return (
    <>
      <PageHeader
        title="Possible State replies"
        description={data.messages.notes.steelman_intro}
      />
      <p className="authority-disclaimer">
        {data.messages.notes.steelman_caveat}
      </p>
      <div className="metrics">
        {[
          ["Findings contested", replies.length],
          [
            "With a supported reply",
            replies.filter((r) => r.arguments.length).length,
          ],
          [
            "Arguments that passed checks",
            replies.reduce((n, r) => n + r.arguments.length, 0),
          ],
        ].map(([label, value]) => (
          <div className="metric" key={label}>
            <div>{label}</div>
            <strong>{value}</strong>
            <small>Recorded model output · Unverified wording</small>
          </div>
        ))}
      </div>
      <div className="stack">
        {replies.map((reply) => {
          const flag = findings.find((f) => f.id === reply.flag_id);
          return flag ? (
            <article className="panel" key={reply.flag_id}>
              <div className="row-heading">
                <h3>{flag.standard_label}</h3>
                <Badge messages={data.messages} status={flag.status} />
              </div>
              <p>{flag.message}</p>
              <Evidence
                items={flag.evidence.slice(0, 1)}
                heading={flag.standard_label}
              />
              <StateReply
                data={data}
                flag={flag}
                context={context}
                Evidence={Evidence}
                expanded
              />
              <FindingActions data={data} flag={flag} navigate={navigate} />
            </article>
          ) : null;
        })}
      </div>
    </>
  );
}
function DecisionForm({ data, flag, review, history, context }) {
  const { saveReview } = useContext(context);
  const [decision, setDecision] = useState(review?.decision || "accepted"),
    [note, setNote] = useState(review?.note || ""),
    [wording, setWording] = useState(review?.edited_message || flag.message),
    [reviewer, setReviewer] = useState(""),
    [error, setError] = useState("");
  function save(e) {
    e.preventDefault();
    const problem = reviewError(
      flag,
      decision,
      note,
      wording,
      data.messages.block_list,
      data.review_transliteration,
    );
    if (problem) {
      setError(problem);
      return;
    }
    saveReview(makeReview(data, flag, decision, note, wording, reviewer));
    setError("");
  }
  return (
    <div className="decision-panel">
      <form onSubmit={save}>
        <fieldset>
          <legend>Your decision</legend>
          {[
            ["accepted", "Accept"],
            ["edited", "Reword"],
            ["rejected", "Reject"],
          ].map(([id, label]) => (
            <label className="radio-choice" key={id}>
              <input
                type="radio"
                name={`decision-${flag.id}`}
                value={id}
                checked={decision === id}
                onChange={() => setDecision(id)}
              />
              {label}
            </label>
          ))}
        </fieldset>
        <label htmlFor={`reason-${flag.id}`}>
          Reason {decision === "accepted" ? "(optional)" : "(required)"}
        </label>
        <textarea
          id={`reason-${flag.id}`}
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={3}
        />
        {decision === "edited" && (
          <>
            <label htmlFor={`wording-${flag.id}`}>Your revised wording</label>
            <textarea
              id={`wording-${flag.id}`}
              value={wording}
              onChange={(e) => setWording(e.target.value)}
              rows={4}
            />
          </>
        )}
        <label htmlFor={`reviewer-${flag.id}`}>Reviewer name (optional)</label>
        <input
          id={`reviewer-${flag.id}`}
          value={reviewer}
          onChange={(e) => setReviewer(e.target.value)}
          placeholder="For this synthetic demonstration only"
        />
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <div className="decision-actions">
          <button className="primary">Save decision</button>
          {review && (
            <button
              type="button"
              onClick={() =>
                saveReview(makeReview(data, flag, "reopened", "", ""))
              }
            >
              Reopen finding
            </button>
          )}
        </div>
      </form>
      {history.length > 0 && (
        <details>
          <summary>Decision history ({history.length})</summary>
          <ol className="review-history">
            {history.map((item, i) => (
              <li key={i}>
                <strong>
                  {data.messages.labels["review_" + item.decision] ||
                    "Reopened"}
                </strong>{" "}
                · {new Date(item.created_at).toLocaleString()}
                {item.note && <p>{item.note}</p>}
                {item.edited_message && (
                  <p>Reviewer’s wording: {item.edited_message}</p>
                )}
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}
export function Review({
  data,
  context,
  Evidence,
  Badge,
  navigate,
  selectedId,
}) {
  const { reviews, saveReview, missed, setMissed, documents, saved, notes } =
      useContext(context),
    current = currentReviews(reviews),
    findings = allFindings(data);
  const [selected, setSelected] = useState(
      findings.some((f) => f.id === selectedId) ? selectedId : findings[0].id,
    ),
    [filter, setFilter] = useState("all"),
    [issueModule, setIssueModule] = useState("absence"),
    [standard, setStandard] = useState(""),
    [description, setDescription] = useState(""),
    [issueError, setIssueError] = useState("");
  const flag = findings.find((f) => f.id === selected),
    review = current[selected];
  const visible = findings.filter(
    (f) =>
      filter === "all" || Boolean(current[f.id]) === (filter === "reviewed"),
  );
  function report() {
    const note =
      "> This export contains the recorded pipeline report followed by a separate browser demo review worksheet. Review decisions do not silently replace recorded findings.\n\n";
    downloadFile(
      note +
        data.report_markdown +
        "\n\n" +
        reviewWorksheet(data, reviews, missed, documents),
      `${data.record.case_id}-report-draft.md`,
      "text/markdown",
    );
  }
  function json() {
    downloadFile(
      JSON.stringify(
        {
          synthetic: true,
          case_id: data.record.case_id,
          reviews,
          missed,
          working_file: { saved, notes },
        },
        null,
        2,
      ),
      `${data.record.case_id}-review-worksheet.json`,
      "application/json",
    );
  }
  function addIssue(e) {
    e.preventDefault();
    if (!standard.trim() || !description.trim()) {
      setIssueError("Enter the standard and what the record shows.");
      return;
    }
    setMissed((previous) => [
      ...previous,
      {
        id: crypto.randomUUID(),
        case_id: data.record.case_id,
        module: issueModule,
        standard: standard.trim(),
        note: description.trim(),
        created_at: new Date().toISOString(),
      },
    ]);
    setStandard("");
    setDescription("");
    setIssueError("");
  }
  const stale = Object.values(current).filter(
    (r) => !findings.some((f) => f.id === r.flag_id),
  );
  return (
    <>
      <PageHeader
        kicker="Reviewer workspace"
        title="Review & report"
        description="Accept, reword, or reject a finding with a reason. Record issues the analysis missed."
      >
        <div className="report-actions">
          <button className="primary" onClick={report}>
            <Icon name="download" />
            Report draft (.md)
          </button>
          <button onClick={json}>Review worksheet (.json)</button>
        </div>
      </PageHeader>
      <p className="browser-review-note">
        Synthetic demo decisions are saved in this browser and kept in their
        history. The local Python app stores authoritative case reviews in
        SQLite and exports them within its report.
      </p>
      <div className="metrics">
        {[
          ["Review prompts", findings.length],
          ["Reviewed", findings.filter((f) => current[f.id]).length],
          [
            "Accepted",
            findings.filter((f) => current[f.id]?.decision === "accepted")
              .length,
          ],
          [
            "Reworded",
            findings.filter((f) => current[f.id]?.decision === "edited").length,
          ],
          [
            "Rejected",
            findings.filter((f) => current[f.id]?.decision === "rejected")
              .length,
          ],
        ].map(([label, value]) => (
          <div className="metric" key={label}>
            <div>{label}</div>
            <strong>{value}</strong>
            <small>Current decisions on this record</small>
          </div>
        ))}
      </div>
      {stale.length > 0 && (
        <details className="panel">
          <summary>
            Earlier decisions on findings no longer shown ({stale.length})
          </summary>
          {stale.map((r) => (
            <p key={r.flag_id}>
              {r.flag.standard_label}: {r.decision}. {r.note}
            </p>
          ))}
        </details>
      )}
      <div className="review-layout">
        <div className="review-finding-list">
          <label htmlFor="review-filter">Show findings</label>
          <select
            id="review-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="all">All findings and judge prompts</option>
            <option value="pending">Not reviewed</option>
            <option value="reviewed">Reviewed</option>
          </select>
          {visible.map((f, i) => (
            <button
              key={f.id}
              className={`review-finding ${selected === f.id ? "selected" : ""}`}
              aria-pressed={selected === f.id}
              onClick={() => setSelected(f.id)}
            >
              <span className="eyebrow">{moduleNames[f.module]}</span>
              <strong>{f.standard_label}</strong>
              {f.module === "reuse" && (
                <span className="caption">
                  {f.evidence[0]?.span.text.slice(0, 90)}…
                </span>
              )}
              <Badge
                messages={data.messages}
                status={
                  current[f.id]
                    ? "review_" + current[f.id].decision
                    : "review_pending"
                }
              />
            </button>
          ))}
          {!visible.length && (
            <p className="caption">No findings in this filter.</p>
          )}
        </div>
        <article className="panel review-detail" key={selected}>
          {flag.module === "judges" && (
            <p className="warning">{data.messages.notes.selection_bias}</p>
          )}
          <div className="row-heading">
            <h2>{flag.standard_label}</h2>
            <Badge messages={data.messages} status={flag.status} />
          </div>
          <div className="badges">
            <Badge
              messages={data.messages}
              status={review ? "review_" + review.decision : "review_pending"}
            />
            <Badge messages={data.messages} status={flag.review_status} />
          </div>
          <p>
            {review?.decision === "edited"
              ? review.edited_message
              : flag.message}
          </p>
          {review?.decision === "edited" && (
            <details>
              <summary>Original recorded wording</summary>
              <p>{flag.message}</p>
            </details>
          )}
          <p className="caption">{flag.citation}</p>
          <Evidence items={flag.evidence} heading={flag.standard_label} />
          <LegalContext data={data} flag={flag} />
          <StateReply
            data={data}
            flag={flag}
            context={context}
            Evidence={Evidence}
          />
          <DecisionForm
            data={data}
            flag={flag}
            review={review}
            history={reviews.filter((r) => r.flag_id === selected)}
            context={context}
          />
        </article>
      </div>
      <section className="section missed-issues">
        <h2>Issues Ratio did not flag</h2>
        <p className="section-caption">
          Add your own observation. Withdrawing it preserves the entry in the
          worksheet export.
        </p>
        {missed
          .filter((x) => !x.withdrawn_at)
          .map((issue) => (
            <article className="panel" key={issue.id}>
              <h3>
                {moduleNames[issue.module]} · {issue.standard}
              </h3>
              <p>{issue.note}</p>
              <button
                className="text-action"
                onClick={() =>
                  setMissed((previous) =>
                    previous.map((x) =>
                      x.id === issue.id
                        ? { ...x, withdrawn_at: new Date().toISOString() }
                        : x,
                    ),
                  )
                }
              >
                Withdraw missed issue
              </button>
            </article>
          ))}
        <form className="panel missed-form" onSubmit={addIssue}>
          <label htmlFor="issue-module">Analysis workstream</label>
          <select
            id="issue-module"
            value={issueModule}
            onChange={(e) => setIssueModule(e.target.value)}
          >
            {Object.entries(moduleNames).map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
          <label htmlFor="issue-standard">Standard or guarantee</label>
          <input
            id="issue-standard"
            value={standard}
            onChange={(e) => setStandard(e.target.value)}
            placeholder="For example, ICCPR Art. 14(3)(f)"
          />
          <label htmlFor="issue-description">What the record shows</label>
          <textarea
            id="issue-description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={3}
          />
          {issueError && (
            <p className="form-error" role="alert">
              {issueError}
            </p>
          )}
          <button className="primary">Record missed issue</button>
        </form>
      </section>
    </>
  );
}
