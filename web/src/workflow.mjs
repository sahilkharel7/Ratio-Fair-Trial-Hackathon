import { sourceCitation } from "./research.mjs";

export const moduleNames = {
  absence: "Rights coverage",
  clock: "Procedural timeline",
  renewal: "Detention renewals",
  reuse: "Reasoning comparison",
  judges: "Judicial history",
};
export const moduleRoutes = {
  absence: "/coverage",
  clock: "/timeline",
  renewal: "/renewals",
  reuse: "/reuse",
  judges: "/judges",
};
export const allFindings = (data) =>
  data.review_findings ||
  ["absence", "clock", "renewal", "reuse"].flatMap(
    (name) => data.analysis[name]?.flags || [],
  );
export function currentReviews(history) {
  const current = {};
  for (const review of history) {
    if (review.decision === "reopened") delete current[review.flag_id];
    else current[review.flag_id] = review;
  }
  return current;
}
export function reviewError(
  flag,
  decision,
  note,
  wording,
  blockList = [],
  transliteration = {},
) {
  if (!["accepted", "edited", "rejected", "reopened"].includes(decision))
    return "Choose a review decision.";
  if (["edited", "rejected"].includes(decision) && !note.trim())
    return "A reason is required to reword or reject a finding.";
  if (decision === "edited") {
    if (!wording.trim()) return "Enter your revised wording.";
    if (wording.trim() === flag.message.trim())
      return "The wording is unchanged. Accept the finding or change your wording.";
    const normal = wording
      .normalize("NFKC")
      .replace(/[\u00ad\u200b-\u200d\u2060\ufeff]/g, "");
    const ascii = Array.from(normal)
      .map((char) => transliteration[char] ?? char)
      .join("")
      .normalize("NFKD")
      .replace(/\p{M}/gu, "");
    const blocked = new RegExp(
      `(?<![\\w-])(?:${blockList.join("|")})(?![\\w-])`,
      "iu",
    );
    if (blockList.length && (blocked.test(normal) || blocked.test(ascii)))
      return "Describe the facts or pattern rather than a person’s character or motives.";
  }
  return "";
}
export function makeReview(
  data,
  flag,
  decision,
  note,
  wording,
  reviewer = "",
  at = new Date().toISOString(),
) {
  return {
    case_id: data.record.case_id,
    flag_id: flag.id,
    decision,
    note: note.trim(),
    edited_message: decision === "edited" ? wording.trim() : null,
    reviewer: reviewer.trim() || null,
    created_at: at,
    flag,
  };
}
export function markdownLiteral(text = "") {
  return text
    .replace(/([\\`*_\[\]<>|~&])/g, "\\$1")
    .replace(/^(\s*)(#|>|[-+=]|\d+[.)])/gm, "$1\\$2");
}
export function reviewWorksheet(data, history, missed, documents) {
  const current = currentReviews(history),
    flags = allFindings(data),
    active = missed.filter((x) => !x.withdrawn_at);
  const lines = [
    "## Browser demo review worksheet",
    "",
    "Decisions and missed issues below were entered in this browser. They are not synchronized with the local Python application. The recorded analysis above preserves its original wording; this worksheet records the reviewer’s disposition separately.",
    "",
  ];
  for (const flag of flags) {
    const decision = current[flag.id];
    lines.push(
      `### ${markdownLiteral(flag.standard_label)}`,
      `Finding ID: ${flag.id}`,
      `Decision: ${decision ? data.messages.labels["review_" + decision.decision] || decision.decision : "Not reviewed"}`,
      "",
    );
    if (decision)
      lines.push(
        `Reason: ${markdownLiteral(decision.note) || "No reason entered."}`,
        decision.decision === "edited"
          ? `Reviewer’s wording: ${markdownLiteral(decision.edited_message)}`
          : "",
        `Recorded at: ${decision.created_at}${decision.reviewer ? ` · ${markdownLiteral(decision.reviewer)}` : ""}`,
        "",
      );
    lines.push(
      `Original recorded wording: ${markdownLiteral(flag.message)}`,
      "",
    );
    for (const e of flag.evidence) {
      const doc = documents[e.span.doc_id];
      lines.push(
        markdownLiteral(doc ? sourceCitation(doc, e.span) : e.span.doc_id),
        "",
        ...markdownLiteral(e.span.text)
          .split("\n")
          .map((line) => "> " + line),
        "",
      );
    }
  }
  if (active.length) {
    lines.push("### Issues recorded by the reviewer", "");
    for (const issue of active)
      lines.push(
        `- ${markdownLiteral(moduleNames[issue.module])}: ${markdownLiteral(issue.standard)} — ${markdownLiteral(issue.note)}`,
        "",
      );
  }
  const stale = Object.values(current).filter(
    (r) => !flags.some((f) => f.id === r.flag_id),
  );
  if (stale.length) {
    lines.push("### Earlier decisions whose finding is no longer shown", "");
    for (const r of stale)
      lines.push(
        `- ${markdownLiteral(r.flag.standard_label)}: ${r.decision} — ${markdownLiteral(r.note)}`,
      );
  }
  lines.push("### Decision history", "");
  for (const r of history)
    lines.push(
      `- ${r.created_at} · ${r.flag_id} · ${r.decision} · ${markdownLiteral(r.note)}`,
    );
  return lines.join("\n");
}
export function downloadFile(contents, filename, type) {
  const url = URL.createObjectURL(new Blob([contents], { type })),
    link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
