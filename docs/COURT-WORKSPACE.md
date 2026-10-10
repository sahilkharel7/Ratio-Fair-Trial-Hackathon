# Court case workspace and focused demo

## The lawyer's primary path

1. Choose a court and find the defendant in the case collection.
2. Open the stored record. Read the charge, prosecution's requested penalty,
   sentence imposed, and quoted statutory range as separate source-backed facts.
3. Review Article 14(2) passages: possible burden reversal, public pre-judgment
   guilt statements, or prejudicial courtroom presentation. Each is a source prompt
   requiring the lawyer's assessment, not a confirmed violation.
4. Compare claim-specific international decisions and their sources.
5. Save an assessment and next step, retaining history, and export the worksheet.

Other analysis modules are retained under supporting analysis. Reused prosecution
wording, an accusation, a conviction, and missing monitoring evidence do not by
themselves establish a presumption-of-innocence violation.

## Two collections with different purposes

**Case collection** holds active matters in the existing `data/ratio.db`, using
`cases` plus an additive `matter_catalogue`. It opens the selected case from
SQLite rather than treating the last case as the homepage. Synthetic judge-history
records and past precedent documents do not automatically become active matters.

**Precedent library** is Sahil's latest reference-collection implementation
(`app/views/library.py`, `corpus_builder/collections.py`). It accepts uploaded
reference collections, reads private collections only with local Ollama, and
searches checked paragraphs using the local embedding/index implementations.
These reference files remain separate from active defendant case records.

The integration includes both latest main pushes: `378cb33` (Similar cases) and
`f820a0a` (Precedent library, uploaded collections and paragraph search).

## Actual React uploads with persistent SQLite

From the repository root, after the normal Python setup:

```sh
npm --prefix web ci
npm --prefix web run build
.venv/bin/python scripts/serve_workspace.py --demo
```

Open `http://127.0.0.1:8503`. `--demo` adds three synthetic examples without
overwriting existing cases. Omit it for an empty/new workspace, or use `--db PATH`
for a separate SQLite file. The server binds only to loopback, applies the network
guard, rejects cross-origin/rebinding hosts, and serves only the compiled interface.
It never serves the SQLite file or repository directory.

The existing Streamlit app uses the same default SQLite store:

```sh
streamlit run app/main.py
```

The public Vercel deployment is a synthetic collection preview. Actual uploaded
documents remain in the local workspace. The exporter constructs the preview in a
disposable SQLite file using committed synthetic inputs, not the user's database.

## Bulk-PDF ingestion contract

`LibraryStore.stage()` saves up to 100 PDF/TXT/MD originals (5 MB each, 50 MB per
batch) in SQLite with hashes, provenance, extraction status and assignment state.
Textless/unreadable documents are retained with a problem and cannot silently
become analysed cases. OCR is not fabricated; supply an extracted text version or
use a future OCR adapter.

The user groups files belonging to one case and confirms their document types.
`LibraryStore.import_manifest(CaseManifest, {path: staged_file_id})` reuses the
existing validated `build_base_record` pipeline, then atomically stores the case,
catalogue, original-file mappings and assignment. Duplicate case references,
reused files, mismatched provenance and undeclared documents are refused.

A bulk classifier can return that same manifest/file-id contract, or submit a
validated CaseRecord to the existing store followed by `LibraryStore.register()`.
It does not need to replace the analysis schema. This deliberately leaves document
grouping and classification as a reviewed boundary while Sahil's bulk work evolves.

Case assessments use an append-only `focus_reviews` table with the original prompt
snapshot. A changed source cannot receive a stale assessment. Working notes and
original uploads are persisted in SQLite; original files are downloadable only
through their owning case's mapping.

## Penalty and screening limits

The English screening is deterministic and source-led. Requested, imposed and
statutory statements remain separate. Compound numeric terms, life/death and
unquantified penalties are handled conservatively. Conflicting specific terms are
shown as an ambiguity instead of picking an arbitrary defendant's sentence. The
lawyer must check source context, especially joint trials and changed requests.

The existing Venn record says the prosecutor requested five years, the court
imposed four, and the quoted fictional statutory range is two to six. Two additional
synthetic matters demonstrate an explicit burden-of-proof concern and a normal
prosecution-burden statement. No existing recorded model prompts were changed.

## Historical outcomes and prospects

`ratio/config/innocence_outcomes.json` is a source-checked draft starter registry:
seven deliberately curated UN Human Rights Committee communications. Five have
explicit Article 14(2) merits outcomes (four violations found, one no violation);
two are inadmissibility decisions. Links distinguish official Views, official
annual-report summaries and official decision metadata. Excerpts are short and
are accompanied by citations and source type, not presented as full decisions.

The displayed observed fraction is 4/5 (80%) only for that selected merits cohort.
It is not a representative success rate or a case-specific probability. A filter
with fewer than five merits decisions shows counts without a percentage. Outcomes
under other articles, inadmissible/unexamined claims, other forums, and dissenting
opinions are excluded from that merits denominator. Remedy implementation,
release and acquittal are not inferred from a violation finding.

An individual prospects estimate would require a substantially larger, legally
coded and representative dataset with a defined outcome, forum/admissibility
variables, held-out validation and calibration. This version abstains from that
prediction and gives the lawyer sources and clearly scoped observations.

## Verification

The tests cover source-span integrity, compound and ambiguous sentencing terms,
non-violation traps, SQLite restart behavior, original PDF upload/download hashes,
atomic case assignment, assessment history, origin restrictions, claim-specific
outcome denominators and the new native homepage. Browser QA verifies actual local
upload → assignment → case creation → reload → source reading, plus persisted notes.
