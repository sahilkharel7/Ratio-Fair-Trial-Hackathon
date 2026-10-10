import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import "./workspace.css";
import "./legal-workflows.css";
import "./case-library.css";
import "./theme.css";
import { startEffects } from "./effects.mjs";
import { CourtJudgment } from "./court-library.jsx";
import {
  CaseCollection,
  CaseReview,
  InternationalDecisions,
  IntakeDialog,
  useCaseWorkspace,
} from "./case-library.jsx";
import {
  Renewals,
  Jurisprudence,
  StateReplies,
  Review,
  LegalContext,
  StateReply,
  FindingActions,
} from "./legal-workflows.jsx";
import { allFindings, currentReviews } from "./workflow.mjs";
import {
  Icon,
  ResearchBar,
  MatterHeader,
  ReviewQueue,
  ReviewRail,
  Research,
  SavedWorkspace,
} from "./workspace.jsx";
import { sourceKey, sourceCitation } from "./research.mjs";

const Source = createContext(null);
const navigation = [
  ["/overview", "Full analysis record"],
  ["/coverage", "Rights coverage"],
  ["/timeline", "Procedural timeline"],
  ["/renewals", "Detention renewals"],
  ["/reuse", "Reasoning comparison"],
  ["/judges", "Judicial history"],
  ["/jurisprudence", "Jurisprudence"],
  ["/state-replies", "Possible State replies"],
  ["/review", "Review & report"],
];
// Match Python's displayed percentages, including ties rounded to the even digit.
const percent = (value) => {
  if (value == null) return "n/a";
  const n = value * 100,
    low = Math.floor(n);
  return `${n - low === 0.5 ? (low % 2 === 0 ? low : low + 1) : Math.round(n)}%`;
};
const date = (value) =>
  value
    ? new Date(value).toLocaleDateString("en-GB", {
        day: "numeric",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      })
    : "Undated";
const points = (text, start, end) =>
  Array.from(text).slice(start, end).join("");
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const flags = (analysis) =>
  [analysis.absence, analysis.clock, analysis.renewal, analysis.reuse].flatMap(
    (module) => module?.flags || [],
  );

function Badge({ status, messages }) {
  const tone = ["evidence_of_violation", "exceeds_benchmark"].includes(status)
    ? "adverse"
    : [
          "evidence_of_compliance",
          "within_benchmark",
          "addressed_argument",
          "review_accepted",
        ].includes(status)
      ? "positive"
      : /review|unaddressed|verbatim|repeated|order_gap|paraphrase/.test(status)
        ? "review"
        : "neutral";
  return (
    <span className={`badge ${tone}`}>{messages.labels[status] || status}</span>
  );
}
function Metrics({ items }) {
  return (
    <div className="metrics">
      {items.map(([label, value, note, tone]) => (
        <div key={label} className={`metric ${tone || ""}`}>
          <div>{label}</div>
          <strong>{value}</strong>
          <small>{note}</small>
        </div>
      ))}
    </div>
  );
}
function Section({ title, description, children }) {
  return (
    <section className="section">
      <h2>{title}</h2>
      {description && <p className="section-caption">{description}</p>}
      {children}
    </section>
  );
}
function Notice({ messages }) {
  return <div className="data-notice">{messages.notes.synthetic_banner}</div>;
}
function Header({ data, title, description, crossCase = false }) {
  return (
    <>
      <p className="eyebrow">
        {crossCase ? "Cross-case analysis" : "Case analysis"}
      </p>
      <h1>{title}</h1>
      <Notice messages={data.messages} />
      {!crossCase && (
        <p className="caption">
          {data.record.meta.title} · {data.record.meta.court}
        </p>
      )}
      {description && <p className="intro">{description}</p>}
    </>
  );
}
function Evidence({ items = [], heading = "" }) {
  const { documents, openSource, messages } = useContext(Source);
  return (
    <div className="evidence-list">
      {items.map((item, index) => (
        <div
          className="evidence"
          key={`${item.span.doc_id}-${item.span.start}-${index}`}
        >
          <blockquote>
            <cite>
              {messages.labels[`role_${item.role}`]} ·{" "}
              {documents[item.span.doc_id]?.title}
            </cite>
            “{item.span.text}”
          </blockquote>
          <button
            aria-label={`View source: ${documents[item.span.doc_id]?.title}, passage ${index + 1}`}
            onClick={() => openSource(item.span, heading)}
          >
            View source
          </button>
        </div>
      ))}
    </div>
  );
}

function Overview({ data, navigate, onAbout }) {
  const { record, analysis } = data;
  return (
    <>
      <MatterHeader
        data={data}
        navigate={navigate}
        download={() => navigate("/review")}
      />
      <div className="overview-lead">
        <div>
          <p className="eyebrow">Matter overview</p>
          <h2>Review summary</h2>
          <p>Recorded findings and supporting evidence from the matter file.</p>
        </div>
        <span className="review-label">
          <Icon name="scale" /> Awaiting legal review
        </span>
      </div>
      <Metrics
        items={[
          [
            "Source documents",
            record.documents.length,
            "The original matter record",
          ],
          [
            "Findings to review",
            allFindings(data).length,
            "Case findings and judge review prompts",
            "review",
          ],
          [
            "Monitor follow-ups",
            analysis.absence.follow_ups.length,
            "Questions where evidence is missing",
          ],
          [
            "Traceable reasoning",
            percent(analysis.reuse.score),
            "Judgment text matching the indictment",
          ],
        ]}
      />
      <div className="overview-grid">
        <div>
          <ReviewQueue data={data} navigate={navigate} context={Source} />
          <section className="record-collection">
            <div className="section-head">
              <div>
                <h2>Source record</h2>
                <p>Every document in this matter.</p>
              </div>
              <button
                className="text-action"
                onClick={() => navigate("/research?scope=documents")}
              >
                Browse all <Icon name="arrow" size={15} />
              </button>
            </div>
            <div className="record-shortcuts">
              {[
                ["judgment", "Judgment", "Court’s reasons and disposition"],
                [
                  "indictment",
                  "Indictment",
                  "Charges and prosecution’s position",
                ],
                [
                  "monitoring_note",
                  "Monitoring notes",
                  "Four hearings in the record",
                ],
                [
                  "detention_order",
                  "Detention orders",
                  "Three original orders and their grounds",
                ],
              ].map(([type, title, note]) => (
                <button
                  key={type}
                  onClick={() =>
                    navigate(
                      `/research?q=${encodeURIComponent(type === "monitoring_note" ? "Monitoring" : type === "detention_order" ? "detention" : title)}&scope=documents`,
                    )
                  }
                >
                  <Icon name="file" />
                  <span>
                    <strong>{title}</strong>
                    <small>{note}</small>
                  </span>
                  <Icon name="arrow" size={15} />
                </button>
              ))}
            </div>
          </section>
          <details className="method-record">
            <summary>Analysis record and method</summary>
            <p>
              {flags(analysis).length} findings, each linked to exact source
              text. {analysis.dropped_flags} dropped because their source text
              could not be found.
            </p>
            <p className="caption">
              Model: {analysis.llm_model}. Analysis was generated locally and
              exported to this preview.
            </p>
            <button onClick={onAbout}>About the synthetic preview</button>
          </details>
        </div>
        <ReviewRail data={data} navigate={navigate} context={Source} />
      </div>
    </>
  );
}

function Coverage({ data, selectedId, navigate }) {
  const { absence } = data.analysis;
  const [selected, setSelected] = useState(
    absence.assessments.some((a) => a.rubric_id === selectedId)
      ? selectedId
      : absence.assessments[0].rubric_id,
  );
  useEffect(() => {
    if (absence.assessments.some((a) => a.rubric_id === selectedId))
      setSelected(selectedId);
  }, [selectedId, absence]);
  const a = absence.assessments.find((a) => a.rubric_id === selected),
    flag = absence.flags.find((f) => f.id === a.flag_id);
  const statuses = absence.assessments.map((a) => a.status);
  return (
    <>
      <Header
        data={data}
        title="Rights coverage"
        description="Each guarantee is marked as evidence of compliance, evidence of violation, or no evidence. Missing evidence becomes a follow-up question for the monitor."
      />
      <Metrics
        items={[
          [
            "Evidence of violation",
            statuses.filter((s) => s === "evidence_of_violation").length,
            "Review the contrary monitoring notes",
            "review",
          ],
          [
            "Evidence of compliance",
            statuses.filter((s) => s === "evidence_of_compliance").length,
            "Supported by the monitoring record",
            "positive",
          ],
          [
            "Monitor follow-ups",
            statuses.filter((s) => s === "no_evidence").length,
            "No evidence recorded; follow up with the monitor",
          ],
        ]}
      />
      <Section
        title="Guarantee review"
        description="Select a guarantee to inspect its evidence and the parts covered by the record."
      >
        <div className="split coverage-split">
          <div className="guarantees" aria-label="Fair-trial guarantees">
            {absence.assessments.map((item) => (
              <button
                className={`guarantee panel ${item.rubric_id === selected ? "selected" : ""}`}
                aria-pressed={item.rubric_id === selected}
                onClick={() => setSelected(item.rubric_id)}
                key={item.rubric_id}
              >
                <strong>{item.provision}</strong>
                <span>{item.name}</span>
                <Badge messages={data.messages} status={item.status} />
              </button>
            ))}
          </div>
          <article className="panel sticky-detail" aria-live="polite">
            <h3>
              {a.provision}: {a.name}
            </h3>
            <div className="badges">
              <Badge messages={data.messages} status={a.status} />
              <Badge messages={data.messages} status={a.review_status} />
            </div>
            <p className="caption">{a.citation}</p>
            {flag ? (
              <>
                <p>{flag.message}</p>
                <Evidence items={flag.evidence} heading={flag.standard_label} />
                <LegalContext data={data} flag={flag} />
                <StateReply
                  data={data}
                  flag={flag}
                  context={Source}
                  Evidence={Evidence}
                />
                <FindingActions data={data} flag={flag} navigate={navigate} />
                {flag.model_note && (
                  <details>
                    <summary>Model note (unverified)</summary>
                    <p>{flag.model_note}</p>
                  </details>
                )}
              </>
            ) : (
              <>
                <div className="info">
                  <strong>Follow-up for the monitor</strong>
                  <p>{a.follow_up?.question}</p>
                </div>
                <h4>Not covered by the notes</h4>
                <ul>
                  {a.parts
                    .filter((p) => p.required && p.status === "no_evidence")
                    .map((p) => (
                      <li key={p.part_id}>
                        {p.label}
                        {p.hearings_missing.length
                          ? ` · ${p.hearings_missing.map(date).join(", ")}`
                          : ""}
                      </li>
                    ))}
                </ul>
                {a.unlabelled_notes > 0 && (
                  <p>
                    {a.unlabelled_notes} shortlisted notes were not labelled by
                    the model.
                  </p>
                )}
                <p className="caption">
                  Possibly relevant, not counted as evidence:
                </p>
                <Evidence items={a.follow_up?.context} heading="Context" />
                <LegalContext data={data} question={a.rubric_id} />
              </>
            )}
            <h4>Coverage by part</h4>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Part</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {a.parts.map((p) => (
                    <tr key={p.part_id}>
                      <td>
                        {p.label}
                        {p.required && (
                          <small className="required">Required</small>
                        )}
                      </td>
                      <td>
                        <Badge messages={data.messages} status={p.status} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </div>
      </Section>
    </>
  );
}

function TimelineChart({ data }) {
  const events = data.analysis.clock.timeline.filter((e) => e.date),
    start = Math.min(...events.map((e) => Date.parse(e.date))),
    end = Math.max(...events.map((e) => Date.parse(e.date)));
  const x = (value) =>
    250 +
    (((typeof value === "number" ? value : Date.parse(value)) - start) /
      (end - start)) *
      570;
  const names = data.messages.event_labels,
    rows = Object.keys(names).filter((type) =>
      events.some((e) => e.type === type),
    );
  const height = rows.length * 40 + 80;
  return (
    <div className="chart-scroll">
      <svg
        viewBox={`0 0 850 ${height}`}
        role="img"
        aria-label="Procedural events from February to July 2025"
      >
        {Array.from({ length: 6 }, (_, i) => {
          const time = start + ((end - start) * i) / 5;
          return (
            <g key={i}>
              <line
                x1={x(time)}
                x2={x(time)}
                y1="15"
                y2={height - 45}
                stroke="#e8edf1"
              />
              <text
                x={x(time)}
                y={height - 18}
                textAnchor="middle"
                className="chart-label"
              >
                {date(time).replace(" 2025", "")}
              </text>
            </g>
          );
        })}
        {rows.map((type, i) => (
          <g key={type}>
            <text
              x="232"
              y={37 + i * 40}
              textAnchor="end"
              className="chart-label"
            >
              {names[type]}
            </text>
            {events
              .filter((e) => e.type === type)
              .map((e) => (
                <circle
                  key={e.id}
                  cx={x(e.date)}
                  cy={32 + i * 40}
                  r="5"
                  fill="#28587d"
                >
                  <title>
                    {names[type]} · {date(e.date)} ·{" "}
                    {data.messages.labels[`timeline_${e.state}`]}
                  </title>
                </circle>
              ))}
          </g>
        ))}
      </svg>
    </div>
  );
}
function Timeline({ data, navigate }) {
  const clock = data.analysis.clock;
  return (
    <>
      <Header
        data={data}
        title="Procedural timeline"
        description="Only a confirmed benchmark can be marked as exceeded. Other intervals are cited and measured for legal review."
      />
      <Section
        title="Chronology"
        description="Distinct events from the source record. Hover over a point for its date and status."
      >
        <div className="panel">
          <TimelineChart data={data} />
        </div>
      </Section>
      <Section
        title="Benchmark review"
        description="Confirmed benchmarks appear first. Each interval includes the source dates and its citation."
      >
        <div className="stack">
          {[...clock.intervals]
            .sort(
              (a, b) =>
                (b.status === "exceeds_benchmark") -
                (a.status === "exceeds_benchmark"),
            )
            .map((i) => {
              const benchmark = data.benchmarks.benchmarks.find(
                  (b) => b.id === i.benchmark_id,
                ),
                flag = clock.flags.find((f) => f.id === i.flag_id);
              const duration =
                i.min_hours == null
                  ? "Cannot be measured"
                  : i.min_hours === i.max_hours
                    ? `${i.min_hours} hours`
                    : `Between ${i.min_hours / 24} and ${i.max_hours / 24} days (${i.min_hours}–${i.max_hours} hours; dates are day-level)`;
              const citation =
                flag?.citation ||
                `${benchmark.provision}; ${data.benchmarks.sources[benchmark.citation.instrument].symbol}, para. ${benchmark.citation.paras}`;
              return (
                <article key={i.id} className="panel">
                  <h3>
                    {data.messages.event_labels[benchmark.from_event]} →{" "}
                    {data.messages.event_labels[benchmark.to_event]}{" "}
                    <span className="caption">· {benchmark.provision}</span>
                  </h3>
                  <div className="badges">
                    <Badge messages={data.messages} status={i.status} />
                    <Badge messages={data.messages} status={i.review_status} />
                  </div>
                  <p>
                    {flag?.message ||
                      `${duration}. ${i.threshold_hours == null ? "No confirmed threshold; measured and shown only." : ""}`}
                  </p>
                  <p className="caption">{citation}</p>
                  <p className="caption">{benchmark.note}</p>
                  <details open={i.status === "exceeds_benchmark"}>
                    <summary>Sources ({i.evidence.length})</summary>
                    <Evidence items={i.evidence} heading={benchmark.name} />
                    {flag && (
                      <>
                        <LegalContext data={data} flag={flag} />
                        <StateReply
                          data={data}
                          flag={flag}
                          context={Source}
                          Evidence={Evidence}
                        />
                        <FindingActions
                          data={data}
                          flag={flag}
                          navigate={navigate}
                        />
                      </>
                    )}
                  </details>
                </article>
              );
            })}
        </div>
      </Section>
      <Section
        title="Event record"
        description="Open an event to read every source mention and any date conflicts."
      >
        <div className="stack">
          {[...clock.timeline]
            .sort((a, b) => Date.parse(a.date) - Date.parse(b.date))
            .map((e) => (
              <details className="panel" key={e.id}>
                <summary>
                  {date(e.date)} · {data.messages.event_labels[e.type]}{" "}
                  <span className="caption">
                    · {plural(e.mentions.length, "source")}
                  </span>
                </summary>
                {e.review_reasons.length > 0 && (
                  <p>{e.review_reasons.join("; ")}</p>
                )}
                <Evidence
                  items={e.mentions}
                  heading={data.messages.event_labels[e.type]}
                />
              </details>
            ))}
        </div>
      </Section>
    </>
  );
}

function Reasoning({ data, navigate }) {
  const reuse = data.analysis.reuse;
  const [whole, setWhole] = useState(false);
  const matchFlags = reuse.flags.filter(
    (f) => f.status !== "unaddressed_argument",
  );
  return (
    <>
      <Header
        data={data}
        title="Reasoning comparison"
        description="Quoted statutes, the recited charge, and party positions are excluded first. Highlighted text matches the indictment; its legal significance remains for the reviewing lawyer."
      />
      <Metrics
        items={[
          [
            "Traceable reasoning",
            percent(reuse.score),
            "Share matching indictment text",
          ],
          [
            "Verbatim",
            percent(reuse.verbatim_chars / reuse.reasoning_chars),
            "Word-for-word matches",
          ],
          [
            "Close paraphrase",
            percent(reuse.paraphrase_chars / reuse.reasoning_chars),
            "Similar wording in the reasoning",
          ],
          [
            "Unanswered arguments",
            reuse.arguments.filter((a) => a.flag_id).length,
            "Defence arguments with no response found",
          ],
        ]}
      />
      <Section
        title="Document comparison"
        description="Percentages measure characters of the court's reasoning, with quotations excluded. Match numbers connect both documents."
      >
        <div className="segmented" role="group" aria-label="Document view">
          <button
            aria-pressed={!whole}
            className={!whole ? "active" : ""}
            onClick={() => setWhole(false)}
          >
            Court's reasoning
          </button>
          <button
            aria-pressed={whole}
            className={whole ? "active" : ""}
            onClick={() => setWhole(true)}
          >
            Whole documents
          </button>
        </div>
        <p className="legend">
          <mark className="ratio-verbatim">Verbatim</mark>
          <mark className="ratio-paraphrase">Close paraphrase</mark>
          <span className="muted">
            Greyed text is quotation or not reasoning.
          </span>
        </p>
        <div className="split">
          {[reuse.judgment_doc_id, reuse.indictment_doc_ids[0]].map((docId) => (
            <div key={docId}>
              <h3>{data.record.documents.find((d) => d.id === docId).title}</h3>
              <div
                className="panel document-panel"
                dangerouslySetInnerHTML={{
                  __html: data.comparison[docId][whole ? "whole" : "reasoning"],
                }}
              />
            </div>
          ))}
        </div>
      </Section>
      <Section
        title="Matched passages"
        description="Read each match in context on either side of the comparison."
      >
        <div className="stack">
          {matchFlags.map((f, index) => (
            <article className="panel" key={f.id}>
              <div className="row-heading">
                <h3>{index + 1}. Matched passage</h3>
                <Badge messages={data.messages} status={f.status} />
              </div>
              <p>{f.message}</p>
              <Evidence
                items={f.evidence}
                heading={`Matched passage ${index + 1}`}
              />
              <LegalContext data={data} flag={f} />
              <StateReply
                data={data}
                flag={f}
                context={Source}
                Evidence={Evidence}
              />
              <FindingActions data={data} flag={f} navigate={navigate} />
            </article>
          ))}
        </div>
      </Section>
      <Section title="Defence arguments from the notes">
        <div className="stack">
          {reuse.arguments.map((a) => (
            <article className="panel" key={a.argument_id}>
              <Badge
                messages={data.messages}
                status={
                  !a.checked
                    ? "unchecked_argument"
                    : a.addressed
                      ? "addressed_argument"
                      : "unaddressed_argument"
                }
              />
              <Evidence
                items={[{ role: "argument", span: a.argument }]}
                heading="Defence argument"
              />
              {a.flag_id && (
                <>
                  <p>{reuse.flags.find((f) => f.id === a.flag_id)?.message}</p>
                  <LegalContext
                    data={data}
                    flag={reuse.flags.find((f) => f.id === a.flag_id)}
                  />
                  <StateReply
                    data={data}
                    flag={reuse.flags.find((f) => f.id === a.flag_id)}
                    context={Source}
                    Evidence={Evidence}
                  />
                  <FindingActions
                    data={data}
                    flag={reuse.flags.find((f) => f.id === a.flag_id)}
                    navigate={navigate}
                  />
                </>
              )}
              {a.responding.length > 0 && (
                <>
                  <p className="caption">
                    Answered in the reasoning ({a.passages_checked} of{" "}
                    {a.passages_total} passages checked):
                  </p>
                  <Evidence
                    items={a.responding.map((span) => ({
                      role: "judgment",
                      span,
                    }))}
                    heading="Response in the judgment"
                  />
                </>
              )}
              {a.model_note && (
                <details>
                  <summary>Model note (unverified)</summary>
                  <p>{a.model_note}</p>
                </details>
              )}
            </article>
          ))}
        </div>
      </Section>
    </>
  );
}

function Judges({ data }) {
  const profiles = data.judges.profiles;
  const [selected, setSelected] = useState(
    profiles.find((p) => p.case_ids.includes(data.record.case_id)).judge_id,
  );
  const profile = profiles.find((p) => p.judge_id === selected);
  const rate = (r) =>
    `${r.k} of ${r.n} cases, ${percent(r.rate)} (${percent(r.confidence)} interval ${percent(r.ci_low)} to ${percent(r.ci_high)})`;
  return (
    <>
      <Header data={data} title="Judicial history" crossCase />
      <div className="warning">{data.messages.notes.selection_bias}</div>
      <label htmlFor="judge">Judge</label>
      <select
        id="judge"
        value={selected}
        onChange={(e) => setSelected(e.target.value)}
      >
        {profiles.map((p) => (
          <option key={p.judge_id} value={p.judge_id}>
            {p.display_name} · {p.court}
          </option>
        ))}
      </select>
      <div className="matter">
        <p className="eyebrow">Selected profile</p>
        <h2>{profile.display_name}</h2>
        <p className="caption">
          {profile.court} · {profile.charge_types.join("; ")} ·{" "}
          {plural(profile.case_ids.length, "case")}
        </p>
        <p>
          Indicators compared: <strong>{profile.k_compared}</strong>. Rates
          count each case once.
        </p>
      </div>
      <Section
        title="Coded ruling indicators"
        description="Compared with other judges of the same court and charge type. Small samples are not shown."
      >
        <div className="stack">
          {profile.indicators.map((i) => (
            <article className="panel" key={`${i.rate_id}-${i.charge_type}`}>
              <div className="row-heading">
                <h3>{i.label}</h3>
                {i.pattern && (
                  <Badge
                    messages={data.messages}
                    status="pattern_warrants_review"
                  />
                )}
                {!i.shown && (
                  <Badge messages={data.messages} status="hidden_indicator" />
                )}
              </div>
              {i.shown && (
                <>
                  <div className="rate-comparison">
                    <div>
                      <h4>This judge</h4>
                      <strong>{percent(i.judge.rate)}</strong>
                      <p>{rate(i.judge)}</p>
                    </div>
                    <div>
                      <h4>
                        Baseline · {plural(i.baseline_judges, "other judge")}
                      </h4>
                      <strong>{percent(i.baseline.rate)}</strong>
                      <p>{rate(i.baseline)}</p>
                    </div>
                  </div>
                  <p className="caption">
                    Difference from baseline:{" "}
                    {i.difference_ci
                      .map(
                        (v) =>
                          `${Math.round(v * 100) > 0 ? "+" : ""}${Math.round(v * 100)}`,
                      )
                      .join(" to ")}{" "}
                    percentage points (
                    {(i.difference_confidence * 100).toFixed(1)}% interval).
                  </p>
                </>
              )}
              <p>{i.message}</p>
              {i.outcomes.length > 0 && (
                <details>
                  <summary>
                    Counted cases and coded rulings ({i.outcomes.length})
                  </summary>
                  {i.outcomes.map((o) => (
                    <div key={o.case_id}>
                      <h4>
                        {o.title} ·{" "}
                        {o.counted
                          ? "Counted in numerator"
                          : "Not counted in numerator"}
                      </h4>
                      <Evidence
                        items={[o.ruling]}
                        heading={`${o.title}: coded ruling`}
                      />
                    </div>
                  ))}
                </details>
              )}
            </article>
          ))}
        </div>
      </Section>
      <Section title="Descriptive observations">
        <div className="stack">
          {profile.descriptive.map((d) => (
            <article className="panel" key={`${d.code}-${d.charge_type}`}>
              <h3>{d.label}</h3>
              <p>{d.message}</p>
              {d.shown ? (
                <>
                  <p>
                    {plural(d.cases, "case")} · Range {Math.min(...d.values)} to{" "}
                    {Math.max(...d.values)}
                  </p>
                  <details>
                    <summary>
                      Values and coded rulings ({d.evidence.length})
                    </summary>
                    <Evidence items={d.evidence} heading={d.label} />
                  </details>
                </>
              ) : (
                <Badge messages={data.messages} status="hidden_indicator" />
              )}
            </article>
          ))}
        </div>
      </Section>
      <details className="panel">
        <summary>
          Names waiting for manual confirmation (
          {data.judges.manual_confirmations.length})
        </summary>
        <p className="caption">
          These rulings do not count in any judge's rates or baseline until a
          person confirms the name.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name as written</th>
                <th>Court</th>
                <th>Possible match</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {data.judges.manual_confirmations.map((c) => (
                <tr key={`${c.case_id}-${c.raw_name}`}>
                  <td>{c.raw_name}</td>
                  <td>{c.court}</td>
                  <td>{c.candidate_name || "None"}</td>
                  <td>{c.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

function SourceDialog({ value, onClose, messages }) {
  const ref = useRef(null);
  const { toggleSaved, isSaved, notes, setNotes } = useContext(Source);
  const [copyStatus, setCopyStatus] = useState("");
  useEffect(() => {
    setCopyStatus("");
    if (value) ref.current.showModal();
    else ref.current.close();
  }, [value]);
  const doc = value?.document,
    span = value?.span;
  const valid =
    doc && (!span || points(doc.text, span.start, span.end) === span.text);
  const rawBegin = span ? Math.max(0, span.start - 300) : 0;
  const contextBefore = span
      ? points(doc?.text || "", rawBegin, span.start)
      : "",
    boundary = contextBefore.indexOf("\n\n");
  const begin =
      boundary >= 0
        ? rawBegin + Array.from(contextBefore.slice(0, boundary + 2)).length
        : rawBegin,
    end = span
      ? Math.min(Array.from(doc?.text || "").length, span.end + 450)
      : 0;
  async function copy() {
    try {
      await navigator.clipboard.writeText(
        `${span ? "“" + span.text + "”\n" : ""}${sourceCitation(doc, span)}`,
      );
      setCopyStatus("Citation copied.");
    } catch {
      setCopyStatus(
        "Select and copy the citation below; clipboard is unavailable.",
      );
    }
  }
  return (
    <dialog
      ref={ref}
      className="source-dialog"
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
    >
      <div className="dialog-heading">
        <div>
          <p className="eyebrow">
            Original source ·{" "}
            {doc?.synthetic ? "Synthetic example" : "Public record"}
          </p>
          <h2>{doc?.title || "Source document"}</h2>
        </div>
        <button onClick={onClose} aria-label="Close source viewer">
          <Icon name="close" />
        </button>
      </div>
      {value &&
        (!valid ? (
          <p className="warning">
            This passage was not found in its source document, so it is not
            shown.
          </p>
        ) : (
          <>
            <div className="reader-toolbar">
              <span>
                <Icon name="check" size={15} />
                {span
                  ? "Passage verified · Exact source match"
                  : "Original source text"}
              </span>
              <div>
                <button
                  aria-pressed={isSaved(doc, span)}
                  onClick={() => toggleSaved(doc, span)}
                >
                  <Icon
                    name={isSaved(doc, span) ? "check" : "bookmark"}
                    size={15}
                  />
                  {isSaved(doc, span)
                    ? "Saved"
                    : span
                      ? "Save passage"
                      : "Save document"}
                </button>
                {doc.download_url && (
                  <a className="reader-download" href={doc.download_url}>
                    Download original file
                  </a>
                )}
                <button onClick={copy}>
                  {copyStatus === "Citation copied."
                    ? "Copied"
                    : "Copy citation"}
                </button>
              </div>
            </div>
            {copyStatus && (
              <p className="reader-copy-status" role="status">
                {copyStatus}
              </p>
            )}
            <div className="reader-layout">
              <div className="reader-document">
                {value.heading && (
                  <p className="reader-context">Opened from: {value.heading}</p>
                )}
                <p className="reader-citation">{sourceCitation(doc, span)}</p>
                <div className="source-text">
                  {span ? (
                    <>
                      {begin > 0 ? "…\n" : ""}
                      {points(doc.text, begin, span.start)}
                      <mark>{span.text}</mark>
                      {points(doc.text, span.end, end)}
                      {end < Array.from(doc.text).length ? "\n…" : ""}
                    </>
                  ) : (
                    doc.text
                  )}
                </div>
                {span && (
                  <details>
                    <summary>Read the whole document</summary>
                    <div className="source-text">
                      {points(doc.text, 0, span.start)}
                      <mark>{span.text}</mark>
                      {points(doc.text, span.end)}
                    </div>
                  </details>
                )}
              </div>
              <aside className="reader-notes">
                <p className="eyebrow">Your working notes</p>
                <label htmlFor="source-note">Note on this document</label>
                <textarea
                  id="source-note"
                  rows={7}
                  value={notes[doc.id] || ""}
                  onChange={(e) =>
                    setNotes({ ...notes, [doc.id]: e.target.value })
                  }
                  placeholder="Why does this source matter to your review?"
                />
                <p className="caption">
                  Reading notes are saved in this browser. Save case assessments
                  and next steps on the case page.
                </p>
                {span && (
                  <>
                    <h3>Passage reference</h3>
                    <dl>
                      <dt>Line</dt>
                      <dd>
                        {points(doc.text, 0, span.start).split("\n").length}
                      </dd>
                      <dt>Characters</dt>
                      <dd>
                        {span.start}–{span.end}
                      </dd>
                    </dl>
                  </>
                )}
                <p className="caption">
                  The original text is preserved. Interpretation and legal
                  conclusions remain with you.
                </p>
              </aside>
            </div>
          </>
        ))}
    </dialog>
  );
}

function AboutDialog({ open, onClose, local = false }) {
  const ref = useRef(null);
  useEffect(() => {
    if (open) ref.current.showModal();
    else ref.current.close();
  }, [open]);
  return (
    <dialog className="about-dialog" ref={ref} onCancel={onClose}>
      <div className="dialog-heading">
        <h2>{local ? "About this local workspace" : "About this preview"}</h2>
        <button aria-label="Close preview information" onClick={onClose}>
          ×
        </button>
      </div>
      <p>
        {local
          ? "This local React workspace stores case records, original uploaded documents, assessments and working notes in SQLite on this computer. Its primary workflow focuses on presumption of innocence."
          : "This preview contains three synthetic cases exported from the local pipeline. People, courts and laws in the case collection are fictional. International decision examples are separately sourced from official public UN material."}
      </p>
      <p>
        Source buttons open the stored source text and check the passage before
        highlighting it. Screening passages remain prompts for a lawyer to
        assess. The international outcomes are selected research examples, not a
        prediction of success.
      </p>
      <h3>New-case analysis runs locally</h3>
      <p>
        {local
          ? "Upload source documents from the case collection. Their originals, case records and focused assessments are saved on this computer. The full local-model analysis and Precedent library remain available in the Streamlit application:"
          : "For actual document uploads and persistent case records, use the local workspace server. For full local-model analysis, use the Streamlit application:"}
      </p>
      <pre>streamlit run app/main.py</pre>
      <p>Legal conclusions remain with the reviewing lawyer.</p>
      <a
        href="https://github.com/sahilkharel7/Ratio-Fair-Trial-Hackathon/tree/codex/court-case-workspace"
        target="_blank"
        rel="noreferrer"
      >
        View the deployment branch →
      </a>
    </dialog>
  );
}
function useStored(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const raw = localStorage.getItem(key);
      const parsed = raw ? JSON.parse(raw) : initial;
      if (Array.isArray(initial))
        return Array.isArray(parsed) ? parsed : initial;
      return parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? parsed
        : initial;
    } catch {
      return initial;
    }
  });
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      setFailed(true);
    }
  }, [key, value]);
  return [value, setValue, failed];
}
function App() {
  const [data, setData] = useState(null),
    [error, setError] = useState(""),
    [location, setLocation] = useState(
      window.location.pathname + window.location.search,
    ),
    [source, setSource] = useState(null),
    [selectedMatter, setSelectedMatter] = useState(null),
    [intakeOpen, setIntakeOpen] = useState(false),
    [about, setAbout] = useState(false),
    [drawer, setDrawer] = useState(false),
    [toast, setToast] = useState("");
  const [saved, setSaved, saveFailed] = useStored("ratio-demo-saved-v1", []),
    [recent, setRecent, historyFailed] = useStored("ratio-demo-history-v1", []),
    [notes, setNotes, noteFailed] = useStored("ratio-demo-notes-v1", {}),
    [reviews, setReviews, reviewFailed] = useStored(
      "ratio-demo-reviews-v1",
      [],
    ),
    [missed, setMissed, missedFailed] = useStored("ratio-demo-missed-v1", []);
  const {
    workspace,
    error: workspaceError,
    refresh: refreshWorkspace,
  } = useCaseWorkspace(data);
  const route = location.split("?")[0],
    params = new URLSearchParams(location.split("?")[1] || ""),
    query = params.get("q") || "";
  const primaryRoute =
    route === "/" ||
    route.startsWith("/cases/") ||
    route.startsWith("/judgments/") ||
    route === "/international";
  const pageName =
    route === "/"
      ? "Case collection"
      : route.startsWith("/judgments/")
        ? "Court judgment"
        : route.startsWith("/cases/")
          ? "Case review"
          : route === "/international"
            ? "International decisions"
            : [
                ...navigation,
                ["/research", "Matter research"],
                ["/saved", "Saved research"],
              ].find((n) => n[0] === route)?.[1] || "Trial review";
  useEffect(() => {
    document.title = `Ratio · ${pageName}`;
  }, [pageName]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 3500);
    return () => clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    fetch("/demo.json")
      .then((r) => {
        if (!r.ok) throw new Error("The demo record could not be loaded.");
        return r.json();
      })
      .then((d) => {
        if (
          d.synthetic !== true ||
          d.history.some((r) => !r.meta.synthetic) ||
          !d.record.meta.synthetic
        )
          throw new Error("This preview only accepts synthetic demo data.");
        setData(d);
      })
      .catch((e) => setError(e.message));
    const handler = () => {
      setLocation(window.location.pathname + window.location.search);
      setDrawer(false);
    };
    window.addEventListener("popstate", handler);
    return () => window.removeEventListener("popstate", handler);
  }, []);
  function remember(item) {
    setRecent((previous) =>
      [
        { ...item, at: new Date().toISOString() },
        ...previous.filter((p) => p.key !== item.key),
      ].slice(0, 30),
    );
  }
  function navigate(to) {
    window.history.pushState({}, "", to);
    setLocation(to);
    setDrawer(false);
    window.scrollTo({ top: 0, behavior: "instant" });
    if (to.startsWith("/research")) {
      const q = new URLSearchParams(to.split("?")[1] || "").get("q") || "";
      remember({ key: `search:${q}`, query: q });
    }
  }
  const documents = data
    ? Object.fromEntries(
        [
          data.record,
          ...data.history,
          ...(data.collection?.records || []).map((p) => p.record),
          ...(selectedMatter ? [selectedMatter.record] : []),
        ]
          .flatMap((r) => r.documents)
          .map((d) => [d.id, d]),
      )
    : {};
  function openSource(span, heading) {
    remember({ key: `doc:${span.doc_id}`, doc_id: span.doc_id });
    setSource({ document: documents[span.doc_id], span, heading });
  }
  function openDocument(document) {
    remember({ key: `doc:${document.id}`, doc_id: document.id });
    setSource({ document });
  }
  const context = {
    documents,
    messages: data?.messages,
    openSource,
    openDocument,
    saved,
    recent,
    notes,
    setNotes,
    reviews,
    missed,
    setMissed,
    saveReview: (review) => {
      setReviews((previous) => [...previous, review]);
      setToast(
        review.decision === "reopened"
          ? "Finding reopened; its history is retained."
          : "Demo review decision saved in this browser.",
      );
    },
    notify: setToast,
    isSaved: (doc, span) => saved.some((s) => s.key === sourceKey(doc, span)),
    toggleSaved: (doc, span) => {
      const key = sourceKey(doc, span);
      setSaved((previous) =>
        previous.some((s) => s.key === key)
          ? previous.filter((s) => s.key !== key)
          : [
              ...previous,
              {
                key,
                doc_id: doc.id,
                span,
                citation: sourceCitation(doc, span),
              },
            ],
      );
    },
  };
  const navIcons = [
    "grid",
    "scale",
    "timeline",
    "file",
    "compare",
    "scale",
    "bookmark",
    "compare",
    "check",
  ];
  return (
    <Source.Provider value={context}>
      <a className="skip-link" href="#main">
        Skip to review
      </a>
      <header className="product-header">
        <a
          className="product-brand"
          href="/"
          onClick={(e) => {
            e.preventDefault();
            navigate("/");
          }}
          aria-label="Ratio case collection"
        >
          <span className="brand-mark">r.</span>
          <span>
            ratio<small>LEGAL RESEARCH &amp; REVIEW</small>
          </span>
        </a>
        <nav className="product-nav" aria-label="Product">
          <a
            href="/"
            aria-current={
              route !== "/research" && route !== "/saved" ? "page" : undefined
            }
            onClick={(e) => {
              e.preventDefault();
              navigate("/");
            }}
          >
            Matter workspace
          </a>
          <a
            href="/research"
            aria-current={route === "/research" ? "page" : undefined}
            onClick={(e) => {
              e.preventDefault();
              navigate("/research");
            }}
          >
            Research
          </a>
        </nav>
        <div className="header-tools">
          <button
            aria-label={`Saved research (${saved.length})`}
            onClick={() => navigate("/saved")}
          >
            <Icon name="folder" /> <span>Saved</span>
            {saved.length > 0 && <b>{saved.length}</b>}
          </button>
          <button
            aria-label="Research history"
            onClick={() => navigate("/saved?tab=history")}
          >
            <Icon name="clock" />
            <span>History</span>
          </button>
          <button className="preview-chip" onClick={() => setAbout(true)}>
            {workspace?.mode === "local_sqlite"
              ? "On this computer"
              : "Demo workspace"}
          </button>
        </div>
      </header>
      {primaryRoute ? (
        <div className="collection-shell-band">
          <span>
            <Icon name="scale" />
            Court case workspace
          </span>
          <span>
            Presumption of innocence ·{" "}
            {route.startsWith("/judgments/")
              ? "ECHR Article 6(2)"
              : "ICCPR Article 14(2)"}
          </span>
        </div>
      ) : (
        <ResearchBar
          navigate={navigate}
          query={query}
          onSearch={(q) => navigate(`/research?q=${encodeURIComponent(q)}`)}
        />
      )}
      <button
        className="mobile-menu"
        aria-expanded={drawer}
        aria-controls="workspace-nav"
        onClick={() => setDrawer(!drawer)}
      >
        {drawer ? "Close navigation" : "Matter navigation"}
      </button>
      {drawer && (
        <button
          className="drawer-scrim"
          aria-label="Close navigation"
          onClick={() => setDrawer(false)}
        />
      )}
      <aside id="workspace-nav" className={`sidebar ${drawer ? "open" : ""}`}>
        {!primaryRoute && (
          <div className="side-matter">
            <p className="eyebrow">Current matter</p>
            <strong>
              {route.startsWith("/judgments/")
                ? "Historical court judgment"
                : route === "/international"
                  ? "International review library"
                  : route === "/"
                    ? "Court case collection"
                    : selectedMatter?.record.meta.title ||
                      data?.record.meta.title ||
                      "Case workspace"}
            </strong>
            <span>
              {primaryRoute
                ? "Presumption of innocence"
                : selectedMatter?.record.case_id || "VENN–2025"}
            </span>
          </div>
        )}
        <p className="eyebrow side-label">Review workspace</p>
        <nav aria-label="Primary workspace">
          <a
            href="/"
            aria-current={route === "/" ? "page" : undefined}
            onClick={(e) => {
              e.preventDefault();
              navigate("/");
            }}
          >
            <Icon name="folder" size={17} />
            Case collection
          </a>
          <a
            href={"/cases/" + (selectedMatter?.record.case_id || "venn-2025")}
            aria-current={route.startsWith("/cases/") ? "page" : undefined}
            onClick={(e) => {
              e.preventDefault();
              navigate(
                "/cases/" + (selectedMatter?.record.case_id || "venn-2025"),
              );
            }}
          >
            <Icon name="scale" size={17} />
            Case review
          </a>
          <a
            href="/international"
            aria-current={route === "/international" ? "page" : undefined}
            onClick={(e) => {
              e.preventDefault();
              navigate("/international");
            }}
          >
            <Icon name="bookmark" size={17} />
            International decisions
          </a>
        </nav>
        <details className="supporting-nav">
          <summary>Supporting analysis</summary>
          <nav aria-label="Matter pages">
            {navigation.map(([path, label], i) => (
              <a
                key={path}
                href={path}
                aria-current={route === path ? "page" : undefined}
                onClick={(e) => {
                  e.preventDefault();
                  navigate(path);
                }}
              >
                <Icon name={navIcons[i]} size={17} />
                {label}
                {path === "/coverage" && data && (
                  <span className="nav-count">
                    {data.analysis.absence.assessments.length}
                  </span>
                )}
              </a>
            ))}
          </nav>
        </details>
        <p className="eyebrow side-label">Research tools</p>
        <nav aria-label="Research tools">
          <a
            href="/research"
            aria-current={route === "/research" ? "page" : undefined}
            onClick={(e) => {
              e.preventDefault();
              navigate("/research");
            }}
          >
            <Icon name="search" size={17} />
            Search the matter
          </a>
          <a
            href="/saved"
            aria-current={route === "/saved" ? "page" : undefined}
            onClick={(e) => {
              e.preventDefault();
              navigate("/saved");
            }}
          >
            <Icon name="folder" size={17} />
            Working file<span className="nav-count">{saved.length}</span>
          </a>
        </nav>
        <div className="sidebar-note">
          <div className="provenance-seal">
            <Icon name="check" />
            <span>Built for the record</span>
          </div>
          <p>
            Every finding leads back
            <br />
            to its original source.
          </p>
          <small>
            ICCPR Articles 9 &amp; 14
            <br />
            {workspace?.mode === "local_sqlite"
              ? "Records stay on this computer"
              : "Synthetic demonstration collection"}
          </small>
        </div>
      </aside>
      <main id="main" tabIndex={-1}>
        <div className="workspace-breadcrumb">
          <span>Workspace</span>
          <span aria-hidden="true">/</span>
          {pageName !== "Case collection" && (
            <>
              <a
                href="/"
                onClick={(e) => {
                  e.preventDefault();
                  navigate("/");
                }}
              >
                Case collection
              </a>
              <span aria-hidden="true">/</span>
            </>
          )}
          <strong>{pageName}</strong>
        </div>
        {(saveFailed ||
          historyFailed ||
          noteFailed ||
          reviewFailed ||
          missedFailed) && (
          <p className="warning" role="status">
            Browser storage is unavailable. Your saved sources and notes will
            last only for this session.
          </p>
        )}
        {error ? (
          <div className="empty" role="alert">
            <h1>Demo unavailable</h1>
            <p>{error}</p>
            <button onClick={() => window.location.reload()}>Reload</button>
          </div>
        ) : !data ? (
          <div className="empty" role="status">
            Loading the synthetic trial record…
          </div>
        ) : (
          <>
            {route === "/" ? (
              <CaseCollection
                workspace={workspace}
                error={workspaceError}
                refresh={refreshWorkspace}
                navigate={navigate}
                onImport={() => setIntakeOpen(true)}
              />
            ) : route.startsWith("/judgments/") ? (
              <CourtJudgment
                key={route}
                id={decodeURIComponent(route.slice(11))}
                navigate={navigate}
                onImport={() => setIntakeOpen(true)}
                refresh={refreshWorkspace}
              />
            ) : route.startsWith("/cases/") ? (
              <CaseReview
                caseId={decodeURIComponent(route.slice(7))}
                workspace={workspace}
                bundle={data}
                context={Source}
                navigate={navigate}
                onOpen={setSelectedMatter}
                refresh={refreshWorkspace}
              />
            ) : route === "/international" ? (
              <InternationalDecisions
                key={params.get("pattern") || "all"}
                registry={workspace?.outcomes || data.outcomes}
                local={workspace?.mode === "local_sqlite"}
                onImport={() => setIntakeOpen(true)}
                refresh={refreshWorkspace}
                pattern={params.get("pattern") || ""}
                navigate={navigate}
              />
            ) : selectedMatter &&
              (selectedMatter.record.case_id !== data.record.case_id ||
                selectedMatter.record.documents.some(
                  (d) =>
                    d.sha256 !==
                    data.record.documents.find(
                      (original) => original.id === d.id,
                    )?.sha256,
                )) ? (
              <div className="empty">
                <h1>Supporting analysis</h1>
                <p>
                  This case has the focused worksheet and its original record.
                  The broader tools here contain the recorded Venn
                  demonstration.
                </p>
                <button
                  onClick={() =>
                    navigate("/cases/" + selectedMatter.record.case_id)
                  }
                >
                  Return to this case
                </button>
                <button
                  onClick={() => {
                    setSelectedMatter(null);
                    navigate("/overview");
                  }}
                >
                  Open the recorded demonstration
                </button>
              </div>
            ) : route === "/overview" ? (
              <Overview
                data={data}
                navigate={navigate}
                onAbout={() => setAbout(true)}
              />
            ) : route === "/coverage" ? (
              <Coverage
                data={data}
                selectedId={params.get("guarantee")}
                navigate={navigate}
              />
            ) : route === "/timeline" ? (
              <Timeline data={data} navigate={navigate} />
            ) : route === "/reuse" ? (
              <Reasoning data={data} navigate={navigate} />
            ) : route === "/judges" ? (
              <Judges data={data} />
            ) : route === "/renewals" ? (
              <Renewals
                data={data}
                context={Source}
                Evidence={Evidence}
                Badge={Badge}
                navigate={navigate}
              />
            ) : route === "/jurisprudence" ? (
              <Jurisprudence
                key={params.get("finding") || "all"}
                data={data}
                context={Source}
                Evidence={Evidence}
                Badge={Badge}
                navigate={navigate}
                selectedId={params.get("finding")}
              />
            ) : route === "/state-replies" ? (
              <StateReplies
                data={data}
                context={Source}
                Evidence={Evidence}
                Badge={Badge}
                navigate={navigate}
              />
            ) : route === "/review" ? (
              <Review
                key={params.get("finding") || "all"}
                data={data}
                context={Source}
                Evidence={Evidence}
                Badge={Badge}
                navigate={navigate}
                selectedId={params.get("finding")}
              />
            ) : route === "/research" ? (
              <Research
                data={data}
                query={query}
                navigate={navigate}
                context={Source}
                scope={params.get("scope") || "all"}
              />
            ) : route === "/saved" ? (
              <SavedWorkspace
                context={Source}
                navigate={navigate}
                tab={params.get("tab")}
              />
            ) : (
              <>
                <h1>Page not found</h1>
                <p>This review page does not exist.</p>
                <a
                  href="/"
                  onClick={(e) => {
                    e.preventDefault();
                    navigate("/");
                  }}
                >
                  Return to case overview
                </a>
              </>
            )}
            <footer>
              <span>Ratio · Evidence-led legal review</span>
              <span>
                {workspace?.mode === "local_sqlite"
                  ? "Local case workspace · Records stored on this computer"
                  : "Synthetic demonstration · Recorded local analysis"}
              </span>
            </footer>
          </>
        )}
      </main>
      {data && (
        <SourceDialog
          value={source}
          onClose={() => setSource(null)}
          messages={data.messages}
        />
      )}
      <IntakeDialog
        open={intakeOpen}
        onClose={() => setIntakeOpen(false)}
        workspace={workspace}
        refresh={refreshWorkspace}
        navigate={navigate}
      />
      <AboutDialog
        open={about}
        onClose={() => setAbout(false)}
        local={workspace?.mode === "local_sqlite"}
      />
      <div className={`toast ${toast ? "visible" : ""}`} role="status">
        {toast}
      </div>
    </Source.Provider>
  );
}

createRoot(document.getElementById("root")).render(<App />);
startEffects();
