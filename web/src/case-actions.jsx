import React, { useRef, useState } from "react";
import { encodeFiles, requestJSON } from "./library-client.mjs";

const TYPES = [
  ["monitoring_note", "Monitoring note"],
  ["judgment", "Judgment"],
  ["indictment", "Indictment"],
  ["detention_order", "Detention order"],
  ["transcript", "Transcript"],
];

// Add more documents to a stored case, or delete it with everything saved for it on this computer.
export function CaseFileActions({ caseId, record, navigate, refresh, onUpdated }) {
  const input = useRef(null);
  const [type, setType] = useState(() =>
      record.documents?.length && record.documents.every((d) => d.type === "judgment") ? "judgment" : "monitoring_note",
    ),
    [busy, setBusy] = useState(false),
    [message, setMessage] = useState("");
  async function addFiles(e) {
    const files = [...e.target.files];
    e.target.value = "";
    if (!files.length) return;
    setBusy(true);
    setMessage("");
    try {
      if (
        files.length > 100 ||
        files.some((f) => f.size > 5_000_000) ||
        files.reduce((n, f) => n + f.size, 0) > 50_000_000
      )
        throw new Error("Add at most 100 files, 5 MB each and 50 MB in total.");
      const staged = await requestJSON("/api/intake", {
        files: await encodeFiles(files),
        provenance: record.meta.data_provenance,
        source_note: record.meta.source_note || "",
      });
      const batch = staged.pending_files.filter((f) => f.batch_id === staged.batch_id);
      const problems = batch.filter((f) => f.problem);
      if (problems.length) throw new Error(problems.map((f) => `${f.name}: ${f.problem}`).join(" "));
      const fresh = await requestJSON(`/api/cases/${encodeURIComponent(caseId)}/files`, {
        file_ids: batch.map((f) => f.id),
        type,
      });
      onUpdated(fresh);
      refresh();
      setMessage(`Added ${batch.length} document${batch.length === 1 ? "" : "s"} to this case.`);
    } catch (err) {
      setMessage(err.message || "The documents could not be added.");
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    if (!window.confirm("Delete this case and its documents from this computer? This cannot be undone.")) return;
    setBusy(true);
    try {
      await requestJSON(`/api/cases/${encodeURIComponent(caseId)}/delete`, { confirm: true });
      await refresh();
      navigate("/");
    } catch (err) {
      setMessage(err.message || "The case could not be deleted.");
      setBusy(false);
    }
  }
  return (
    <div className="case-file-actions">
      <label className="case-file-type">
        Add as
        <select value={type} onChange={(e) => setType(e.target.value)}>
          {TYPES.map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <button disabled={busy} onClick={() => input.current?.click()}>
        {busy ? "Working…" : "Add files"}
      </button>
      <input ref={input} type="file" multiple accept=".pdf,.txt,.md" hidden onChange={addFiles} />
      <button className="case-delete" disabled={busy} onClick={remove}>
        Delete case
      </button>
      {message && (
        <p className="caption" role="status">
          {message}
        </p>
      )}
    </div>
  );
}
