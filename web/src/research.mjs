// Search the matter record, preserving Unicode character offsets used by the pipeline.
export function searchTerms(query) {
  return [...query.matchAll(/"([^"]+)"|(\S+)/g)]
    .map((m) => (m[1] || m[2]).toLowerCase())
    .filter(Boolean);
}
export function matchesQuery(text, query) {
  const haystack = text.toLowerCase();
  return searchTerms(query).every((term) => haystack.includes(term));
}
export function firstMatch(document, query) {
  for (const term of searchTerms(query)) {
    const match = new RegExp(
      term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
      "iu",
    ).exec(document.text);
    if (!match) continue;
    const start = match.index,
      text = match[0];
    return {
      doc_id: document.id,
      start: Array.from(document.text.slice(0, start)).length,
      end: Array.from(document.text.slice(0, start + text.length)).length,
      text,
    };
  }
  return null;
}
export function sourceKey(document, span) {
  return span ? `${document.id}:${span.start}:${span.end}` : document.id;
}
export function sourceCitation(document, span) {
  return `${document.title}${span ? `, line ${Array.from(document.text).slice(0, span.start).join("").split("\n").length}` : ""} [${document.synthetic?'synthetic':'public'} record]`;
}
