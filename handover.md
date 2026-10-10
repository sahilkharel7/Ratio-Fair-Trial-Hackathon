# Ratio handover to Sahil and his coding agent

Prepared 10 October 2026. This document describes the actual implementation and local runtime snapshot, distinguishes completed work from prototypes, and gives a concrete integration order. It contains no credentials or confidential case passages.

**Public-document transfer update:** at the user's subsequent explicit request, all 49 public corpus originals, normalized source text/metadata, 49 reading aids and the public-only runtime index are now versioned under `public-references/`. Run `python scripts/install_public_references.py` after cloning. This supersedes the earlier statement below that the public corpus assets are unavailable through Git. Case workspace data, Maseko inputs/cached aid, reviewer work, credentials and model weights are still excluded. The three live-upload demo PDFs remain held out.

The installer restores runtime assets only and refuses to replace a different index or reading aid. It does not restore `build.db`. Do not run `build-db` with an empty build store over the restored snapshot; use the supplied normalized public JSON files for a deliberate maintainer restore first.

## 1. Read this first

We are building a human-rights lawyer's case-review workspace for the hackathon, with **presumption of innocence as the single demo focus**. The primary flow is: choose a court/case, identify the defendant and charge, understand the recorded consequences, inspect original source passages, compare appropriately scoped legal references, record the lawyer's assessment, and export a reading aid/worksheet.

The latest user instruction supersedes the earlier Groq demo arrangement:

> All uploaded case files, case-specific workspace content, notes, analyses, case-derived queries and comparisons stay local. Only explicitly designated public legal references may use a configured cloud model. There is never a confidential-to-cloud fallback.

**Do not restore the old case-to-Groq route during the merge.** `data_provenance="public"` means the source was published; it is not authorization to send a review matter to cloud. The case-workspace role takes precedence over its published origin.

The real Maseko matter is visible and its checked reading aid is saved locally. Ten focused full decisions are installed within a larger 49-document reference corpus. This is **not** a completed, fully validated AI legal-evaluation system: the Maseko comparison currently abstains, full model extraction/analysis has not been run for that case, and the small local model's legal interpretations proved unreliable.

## 2. Git state and how to sync

### Branch and integration history

The handover branch is **`codex/real-case-library`** in `sahilkharel7/Ratio-Fair-Trial-Hackathon`. It should be reviewed/merged into main, rather than reimplementing these changes on a separate divergent branch.

| Commit / PR | Meaning |
| --- | --- |
| `378cb33` | Sahil's Similar cases implementation. Already integrated before this work. |
| `f820a0a` | Sahil's Precedent library, uploaded collections and paragraph search. Already integrated. |
| PR #1, merge `c4fd454` | Combined legal research/workspace integration. |
| PR #2, merge `add71f8` | SQLite case collection and focused innocence workspace. |
| PR #3, merge `994454c` | Actual court-reference homepage, corpus search, original PDFs, reading sheets and separate demo upload pack. |
| `a3c1e57`, `a313389` | Sahil's Ratio sidebar logo and PNG browser-tab icon. Included. |
| PR #4, merge `dcd777e` | Earlier Groq case-brief demo. **Its case-cloud routing is superseded by this branch.** |
| `3242b38` | Real Maseko import, six additional official full decisions, local-only case routing and initial local comparisons. |
| `e433b50` | Helper-script formatting/bootstrap cleanup after the timebox. |
| `8445924` | Sahil's `ui-polish` branch: React restyling, Lato font, theme and effects. |
| `e98b0b0`, `025d88e` | Main's UI-polish merge and primary-interface documentation update. |
| `191dfef` | Merged that latest main into the handover branch. Both the UI polish and confidential-routing changes are present. |

The handover document and final verification changes are committed after `191dfef`; obtain their final SHA from the pushed branch. The table intentionally does not invent a self-referential handover commit hash.

### Suggested integration procedure

On a clean working branch, preserving any agent/user changes first:

```sh
git fetch origin
git status --short
git merge origin/codex/real-case-library
```

Use a normal merge or PR. Do not force-push, replace main wholesale, reset reviewer data, or copy only the React files. The privacy boundary, schema, API and UI changes are coupled.

If main advances again, inspect that delta before merging. During the most recent sync, the overlap areas were `README.md`, `web/src/main.jsx` and `web/src/case-library.css`; the merge completed without conflicts. Keep Sahil's theme/effects/self-hosted fonts while preserving the local-only AI controls, confidential upload default, provenance labels and local comparison component.

**Deployment:** automatic previews are disabled for this handover branch in `web/vercel.json`. Main's existing deployment settings are unchanged. No manual Vercel deployment is needed for this version. Git-connected main automation can still run when the PR is merged. The hosted static export is synthetic-only and cannot provide the local SQLite/Ollama workflow.

## 3. Product priorities, in order

### P0: a truthful, reproducible real-case demo

1. Keep Maseko visible in the matter catalogue, with originals/excerpts clearly attributed.
2. Make its source-checked reading aid reproducible from the importer, not dependent on the current machine's cached SQLite row. This is the largest immediate reproducibility gap.
3. Preserve charges, historical sentence and later procedural developments as separate facts. Do not fabricate a prosecutor's request or statutory maximum.
4. Repair the comparison selector so a clearly attributed allegation of prejudgment can be reviewed without pretending it is a judicial finding. Preserve abstention when there is no supported comparison.
5. Add explicit end-to-end routing tests: a private upload and a published-origin **review case** must both use Ollama; neither may construct a cloud client on error.
6. Check private labels throughout exports/native views, not just the React badges. `ratio/workspace.py::focus_report` still has an older non-synthetic => “Public record” header and needs a private-specific label.

### P1: useful, verified legal references

1. Keep the existing corpus; do not replace it with the ten focused decisions.
2. Improve source quality and interpretation checks before increasing quantity. Quotes and PDF-page checks are implemented; semantic correctness is a separate issue.
3. Prefer full official decisions over report summaries or metadata. Keep violation, no-violation, inadmissibility, unexamined claims and dissent distinct.
4. Tie the manual outcome registry to installed full documents without silently changing its historical coding or denominators.
5. General comparison factors may be generated from public reference material. Applying them to our case must remain local.

### P2: additional real matters and bulk intake

Prepare **Malaysiakini**, then **Tam Tak-chi**, only after the minimum workflow is reliable. They were not imported during the timeboxed pass. Reuse the available original files and existing intake contract. Do not invent an indictment, hearing record or appeal decision to fill an empty slot.

Bulk upload should classify/group documents into the existing manifest + file-ID contract. Preserve original bytes, provenance, document roles and case assignment. An unknown upload starts confidential; resemblance to a public PDF is not a cloud-processing permission.

### P3: polish the established workflow

Preserve the professional legal-research typography and restrained palette; Sahil's latest UI polish is now included. Avoid another broad redesign. The charge and consequences need to be visible before supporting analytics. Keep advanced modules available without turning the focused demo into five dashboards of scores.

Do not introduce a case-specific “percentage chance of success.” The current selected historical fraction is not a prediction or a representative success rate.

## 4. Actual local runtime snapshot

These counts were read from the actual SQLite store and reference index, not from README claims or the static bundle.

### Visible matters: 5

| Case ID | Type / purpose | Documents | Full stored analysis |
| --- | --- | ---: | --- |
| `real-maseko-makhubu-2014` | Published-origin real trial record; case processing local | 22 | No |
| `allenet-1995-reference` | Historical ECHR judgment prepared as a reference-demo review matter | 1 | No |
| `sorin-2025` | Synthetic burden-of-proof example | 2 | No |
| `maren-2025` | Synthetic normal prosecution-burden example | 2 | No |
| `venn-2025` | Original synthetic recorded analysis example | 6 | Yes |

Counts apply to the visible `matter_catalogue`, not every case/history row in the database. Reference decisions do not automatically become review matters. Decide whether the Allenet historical-demo matter should remain in the active ledger; do not delete it or reviewer work automatically.

### Reference corpus: 49 documents, 5,546 passages, 0 fact-pattern facets

| Kind | Count | Interpretation |
| --- | ---: | --- |
| ECHR court judgments | 3 | Actual regional judgments, Article 6(2) context |
| Human Rights Committee decisions | 12 | Includes broader existing fair-trial material, not all focused on Article 14(2) |
| WGAD opinions | 11 | Arbitrary-detention opinions, not court judgments |
| TrialWatch reports | 23 | Observer/report assessments, not judicial findings |

The six additions were deduplicated against existing document IDs and original-file hashes. Their 286 new passages were embedded locally with the existing MiniLM implementation. Existing corpus documents were preserved.

`extraction_model` remains `unknown` and the facet count is zero. Public-reference reading-aid generation does **not** populate the fact-pattern extraction/verification tables. Meaning/phrase research works; do not claim that the full shared-fact-pattern Similar cases pipeline was populated by these reading aids.

## 5. Maseko import: precisely what happened

### Source inputs

On the current machine, the supplied folder is `~/Downloads/Ratio-real-cases/`:

```text
Ratio-real-cases/
  maseko-case-for-ratio/
    case.yaml
    indictment.txt
    judgment.txt
    notes/background.txt
    notes/hearing_1.txt ... notes/hearing_19.txt
  maseko-downloads/
    icj_report.pdf
    judgment_exa.json
    judgment_exa_raw.txt
    judgment_jina.txt
    icj_answer_key.json
  maseko-ratio-results/
    record.json, analysis.json, report.json, calls.json
  other-candidate-cases/
    ... supplied report/judgment files for other candidates
```

`scripts/import_real_demo.py` reads the manifest and its declared files. It checks the hearing-excerpt paragraphs against the supplied 112-page ICJ original PDF, and the judgment/indictment against the longest text value in the supplied EswatiniLII capture `judgment_exa.json`.

The fidelity check normalizes Unicode, case, whitespace and punctuation by comparing canonical alphanumeric streams. It is a layout-tolerant text match, **not** byte identity, proof of independent transcription, or independent verification of the publisher's authenticity. Do not describe it more strongly.

Observed audit:

- All **289 judgment paragraphs** checked in the exploratory audit matched the supplied capture after layout normalization.
- **159 ICJ excerpt paragraphs** matched the original report using the importer’s canonical check.
- **14 unmatched report paragraphs were omitted** from the imported evidence. They are not silently passed to the model.
- The resulting record has **22 documents and 340 deterministic observations**.
- Hearing headers/dates and source attributions were retained. The notes are ICJ report excerpts, not independent verbatim courtroom transcripts.
- `indictment.txt` is a reproduction of charges from judgment paragraph [2], not a separately authenticated original indictment.

The importer stages bytes with `LibraryStore.stage()`, then calls `LibraryStore.import_manifest()`. That stores the base record, the visible matter row, focus worksheet, original-file mappings and assignment state. Inserting only into `cases` would not satisfy this workflow.

Excluded from evidence/model input: `icj_answer_key.json`, supplied `record.json`, supplied `analysis.json`, and the report's conclusions. The existing answer key/results may be inspected separately for validation; do not turn them into evidence or claim they are newly verified outputs.

### Idempotency and preservation

Running the importer a second time reported `Existing case preserved` and did not overwrite the stored matter, notes, assessments or originals.

There is an edge case to improve: the early return checks `cases`, not whether a legacy case is missing its `matter_catalogue` row. If an earlier workflow stored only a case row, repair its registration carefully without overwriting an existing catalogue/focus/reviewer state. The current Maseko import is correctly registered.

### Case description and limits

This is the 2014 Swaziland contempt-of-court trial involving Thulani Maseko, Bheki Makhubu and publisher defendants, arising from articles about the judiciary. It is a **historical trial record**, not a newly pending defendant matter.

The ICJ sentencing observation records two years on each count for Maseko/Makhubu, running concurrently. That is an observer's account of the imposed sentence. Do not present it as a prosecutor's request, a statutory range, or current exposure. The supplied manifest/report describes a later reversal in June 2015; no separate appeal judgment was imported in this pass.

No verified Article 14(2) merits finding is coded for Maseko. Its allegation of prejudgment in Annexure A is the defendant's statement, not a court holding or an independently confirmed violation.

## 6. Maseko reading aid and the local-model limitation

The first small-Ollama draft made unsupported statements: an incorrect jurisdiction, invented statutory/requested penalty details, and an assertion about the defendant having been presumed innocent. It was replaced, not accepted as a legal analysis.

The current ready aid has four checked sections:

| Section | Evidence location | What it says |
| --- | --- | --- |
| Charges, as reproduced in the judgment | `indictment.txt`, anchor `COUNT ONE` | Common-law contempt allegations; reproduced indictment, not a separate original |
| Sentence recorded by the ICJ observer | `notes/hearing_19.txt`, anchor `Sentences were handed down` | Two-year concurrent sentences as recorded by the observer; not a request/range |
| Presumption-of-innocence reading lead | `judgment.txt`, anchor `our fate has long been determined` | Maseko's allegation in Annexure A, not a judicial violation finding |
| Attribution and comparison limits | `notes/hearing_19.txt`, anchor `Judge Simelane considered` | Separate observer narrative, party allegations and court record; no success probability |

Supporting quotes were checked against the saved document text and their offsets. The PDF and sections are cached in `case_ai_briefs` under the case/source/model fingerprint, with a PDF checksum.

**Important reproducibility gap:** this curated aid was installed by a one-off local verification command; `import_real_demo.py` currently imports the case but does not recreate that curated cache entry. The ordinary worker now discards model-written labels/legal summaries and returns selected source excerpts instead. Pressing “Regenerate” can replace the curated aid with a less useful extractive aid. Move the deterministic Maseko aid into a source-bound, idempotent repo utility before transferring the demo to another machine. Do not overwrite an existing reviewer-approved/edited aid automatically.

Legacy Groq-produced rows for Allenet and Sorin remain physically in the local database from the earlier allowed demo. They are **not active under the new routing**: changed cache fingerprint/model returns `not_generated`. Do not relabel those old cloud rows as local or rely on them to demonstrate the new policy.

The installed small model is `qwen2.5:1.5b`. It can select numbered passages, but it was not reliable enough for trusted legal interpretation. Evaluate a stronger **local** model before using generated reasoning as more than a draft. Increasing model size alone is not validation. The core pipeline's default recorded model remains `qwen2.5:7b-instruct`; avoid accidentally invalidating recorded synthetic caches by globally changing `RATIO_MODEL`.

## 7. The ten focused full decisions

All ten currently have saved reading aids. The three court aids are source-bound editorial aids; the seven Committee aids are public-reference AI drafts with verified quotes/PDF pages and an explicit interpretation-review label.

| Installed ID | Decision | Focus/status |
| --- | --- | --- |
| `court-echr-15175-89` | Allenet de Ribemont v. France, 1995 | Public statements of guilt; Article 6(2) violation |
| `court-echr-42095-98` | Daktaras v. Lithuania, 2000 | Prosecutor's language in context; no Article 6(2) violation; separate Article 6(1) issue |
| `court-echr-10519-83` | Salabiaku v. France, 1988 | Statutory presumptions/defence rights; no Article 6(2) violation |
| `ccpr-1870-2009` | Sobhraj v. Nepal, 2010 | Burden-of-proof Article 14(2) example |
| `ccpr-1520-2006` | Mwamba v. Zambia, 2010 | Public guilt statements; Article 14(2) example |
| `ccpr-770-1997` | Gridin v. Russian Federation, 2000 | Public guilt statements; Article 14(2) example |
| `ccpr-964-2001` | Saidov v. Tajikistan, 2004 | Pre-trial public/media portrayal; Article 14(2) example |
| `ccpr-2526-2015` | Mirzakhodzhaev v. Kyrgyzstan, 2020 | Majority no-violation example; dissent kept distinct |
| `ccpr-3124-2018` | G.B. v. Latvia, 2021 | Inadmissibility; not a merits/no-violation outcome |
| `ccpr-2252-2013` | Khadzhiyev and Muradova v. Turkmenistan, 2018 | Existing full Committee decision, including Article 14(2) reasoning |

Six full Committee decisions were added; Khadzhiyev/Muradova and the three ECHR judgments already existed. The original six additions total 286 newly embedded passages.

`ccpr-2490-2014`, Hincapié Dávila v. Colombia, was not accepted by identity/language validation. Its manual-registry entry remains metadata-only. Do not fabricate a full text or claim it was installed. Also retain the earlier corpus-download limitation: the Lopez Mendoza/Tintori Parra source returned 404 in the earlier build.

### Source access and rebuilding

`corpus_builder/sources.yaml` now has `ccpr_innocence` with **six `local_files` and zero crawler seeds**. This is intentional: the existing source policy refuses automated access to certain UN download hosts. The six accepted original files are registered as local copies, with official publication URLs, symbols, attribution and hashes.

`scripts/fetch_innocence_references.py` has the explicit seven-symbol list and a fail-closed `REFUSED_HOSTS` check. It skips already-installed IDs, deduplicates hashes and does not bypass the host restriction. **It is not a promise that a clean checkout can automatically redownload the six files.** Use approved official JURIS download links that respect the existing policy or transfer the already-designated public reference copies. Never remove the host guard to make a demo look complete.

The six filenames expected under `data/corpus/local/ccpr_innocence/` are:

```text
ccpr-1870-2009.pdf
ccpr-1520-2006.pdf
ccpr-770-1997.pdf
ccpr-964-2001.pdf
ccpr-2526-2015.pdf
ccpr-3124-2018.pdf
```

### Quote/pinpoint guarantees and limits

`build_innocence_aids.py` sends only fixed-ID public reference excerpts to the configured cloud provider. The model selects numbered candidate passages; the program supplies their original wording. It validates each quote against the saved text and finds its page/page-range in the original PDF. Unsupported page matches are omitted. `load_brief()` checks source/text hashes and generated PDF hashes before serving the output.

These checks certify source location, **not** that every generated sentence is entailed by the quote. Normalization used for page mapping removes non-word characters; duplicate passages can resolve to their first occurrence. Improve this with explicit original offsets/page boundaries and a semantic review gate. Do not change a draft's status to “legally verified” based solely on substring matching.

The other existing reports/opinions retain their labelled extractive reading sheets. Do not describe all 49 as court decisions or all 49 sheets as substantive case summaries.

## 8. Model-routing contract

| Material/task | Allowed processing |
| --- | --- |
| Uploaded case PDF, even if apparently published | Local Ollama only; confidential by default |
| Review case whose `CaseMeta.data_provenance` is `public` | Local Ollama only; published origin does not change workspace role |
| Case facts, notes, analyses, comparisons, derived search queries/embeddings | Local only; no cloud fallback or cloud retrieval service |
| Explicit public reference decision/report/standard in the reference corpus | Configured cloud LLM allowed; local Ollama fallback when unavailable |
| Private uploaded reference collection | Local Ollama only |
| Public-reference retrieval for comparison with our case | Retrieve locally, then compare locally with Ollama |

### Implemented enforcement

- `DataProvenance` now accepts `synthetic`, `public`, `private`.
- `LibraryStore.stage()` defaults to `private`; React intake defaults to confidential; the API defaults an omitted provenance to private.
- `DemoBriefService` always selects a local `RATIO_MODEL`/Ollama model and refuses model names containing `cloud`.
- Its subprocess payload has `source_role="case_workspace"`; the worker refuses other roles.
- `scripts/groq_case_worker.py` is **now local-only despite its historical filename**. Production constructs `OllamaLLM`, installs the network guard, and has no cloud client/fallback.
- `OllamaLLM.check_local()` and `ratio.llm.OllamaClient` verify that a loopback model is not actually served by an Ollama cloud host. A loopback URL alone is insufficient.
- `ratio/local_comparisons.py` uses local MiniLM retrieval and `OllamaClient` only. The query is currently the fixed phrase `presumption of innocence`; it is not sent to cloud.
- `build_innocence_aids.py` is public-reference-only, checks `doc.private`, and uses its fixed ID list. Groq setup/inference failures can fall back to local Ollama for those public references.
- Credentials remain in ignored local `.env`, with owner-only permissions on this machine. Do not put keys in this handover, Git, Vite variables, screenshots or API responses.

Remaining hardening: centralize routing by **purpose/role**, not just a provenance string. Use `corpus_builder.extract.is_private()` for broader reference workflows so deleted/reclassified collections fail closed. The current fixed-ID public aid path is narrower than arbitrary uploaded-collection support; do not expand it without that audit. Add tests for production factories and unavailable-Ollama/no-cloud behavior, not only injected fake runners.

The previously supplied Gemini credential failed its check. A Groq key passed authentication and a small inference check. That says nothing about a paid plan or unlimited quota; no paid credentials or account changes are required for the local case workflow.

## 9. Comparison implementation: current behavior, not a demo claim

`ratio/local_comparisons.py`:

1. Loads the case locally.
2. Looks for an existing focus prompt, or literal `presumption of innocence`, `presumed innocent`, or `burden of proof` wording.
3. Retrieves Committee passages locally with MiniLM meaning search, deduplicating document IDs and selecting up to three non-private references.
4. Sends both sides to **local** Ollama for a short similarity/limits explanation.
5. Uses only supplied reference IDs and copies source passages directly from the stored records.
6. Caches by case contents + corpus file timestamp in `local_comparisons`.

The UI has a **Compare with references locally** action and displays the case passage beside the reference passage, with source links.

**Maseko currently returns zero comparisons**: the literal selector does not pick up its Annexure A allegation that the outcome was predetermined. This abstention was checked through the actual API. The known allegation is already in the curated reading aid; do not pretend the side-by-side comparison was successfully generated.

Next steps: distinguish party allegations/observer facts/court findings during selection; support judicial-prejudgment wording with exact source/context; constrain candidates to relevant full decisions and outcome/stage metadata; validate explanations for unsupported interpretation. If no passage/reference/model answer survives, return no interpretation with an explicit reason. Do not fill the empty state with a canned case or citation.

Other limitations: this endpoint is synchronous, no explicit model/schema version is in its cache fingerprint, and cached results are not revalidated as thoroughly as case-reading-aid spans. Improve those before treating this as production infrastructure. Keep work scoped; the user explicitly rejected a large migration.

## 10. Files to inspect before changing anything

| Area | Files / purpose |
| --- | --- |
| Case contracts and deterministic import | `ratio/schema.py`, `ratio/extraction/loader.py`, `ratio/extraction/build.py` |
| Persistent cases/reviews/intake | `ratio/store.py`, `ratio/library.py`, `ratio/workspace.py` |
| Narrow innocence screen / penalty distinctions | `ratio/focus.py` |
| Existing full analysis pipeline | `ratio/pipeline.py`, `ratio/llm.py`, `ratio/modules/`, `ratio/provenance.py` |
| Network/privacy protection | `ratio/netguard.py`, Ollama remote-model checks in `ratio/llm.py` and `corpus_builder/extract.py` |
| Case reading aids | `ratio/demo_briefs.py`, `scripts/groq_case_worker.py`, `web/src/ai-brief.jsx` |
| Actual local comparisons | `ratio/local_comparisons.py`, `web/src/local-comparisons.jsx` |
| Local HTTP API | `scripts/serve_workspace.py` |
| Public corpus/search | `corpus_builder/`, `ratio/precedent_store.py`, `ratio/corpus_research.py` |
| Reading sheets / editorial court aids | `ratio/case_briefs.py`, `ratio/config/court_briefs.json`, `scripts/build_case_briefs.py` |
| New focused public aids / explicit downloads | `scripts/build_innocence_aids.py`, `scripts/fetch_innocence_references.py`, `corpus_builder/sources.yaml` |
| Maseko source verification/intake | `scripts/import_real_demo.py` |
| Outcome coding | `ratio/config/innocence_outcomes.json`, `ratio/outcomes.py` |
| React workspace and routes | `web/src/case-library.jsx`, `web/src/court-library.jsx`, `web/src/corpus-research.jsx`, `web/src/main.jsx` |
| Sahil's latest visual work | `web/src/theme.css`, `web/src/effects.mjs`, `web/DESIGN.md`, `web/vite.config.mjs` |
| Static synthetic preview/export | `scripts/export_web_demo.py`, `web/public/demo.json`, `web/vercel.json` |
| Existing Streamlit collection/library | `app/views/case_collection.py`, `app/views/library.py`, `app/views/similar_cases.py` |

The `--groq-demo` flag and some historical filenames remain for compatibility. They must not be interpreted as permission to restore cloud case processing. Startup documentation on this branch has been corrected after merging main's older wording.

## 11. API and database contract to reuse

### Key endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/library` | Actual visible matter catalogue, pending originals, manual outcomes and reading-aid capabilities |
| `POST /api/intake` | Stage PDF/TXT/MD originals; default confidential |
| `POST /api/cases` | Confirm manifest/file-ID assignments and create a visible case |
| `GET /api/cases/{id}` | Stored record, focus, assessments, notes and analysis availability |
| `GET /api/cases/{id}/files/{intake_id}` | Download only an original mapped to that case |
| `GET /api/cases/{id}/brief` | Current reading-aid status/sections, subject to source/model fingerprint |
| `POST /api/cases/{id}/brief` | Start local reading-aid generation; cached by default, `force: true` regenerates |
| `GET /api/cases/{id}/brief.pdf` | Download a verified ready PDF |
| `GET /api/cases/{id}/comparisons` | Local comparison generation/cache or explicit abstention |
| `GET /api/precedents` | Actual installed reference catalogue |
| `GET /api/precedents/search` | Local phrase/meaning retrieval with type/state filters |
| `GET /api/precedents/{id}` | Full saved text, provenance and reading-aid sections |
| `GET /api/precedents/{id}/original` | Hash-checked saved public original |
| `GET /api/precedents/{id}/brief.pdf` | Source/PDF-checksum-bound reading sheet |
| `POST /api/precedents/{id}/stage` | Copy a verified reference original into case intake; case processing remains local |

Preserve Host/Origin/Sec-Fetch-Site checks, JSON validation, source path containment and no-store responses. Do not serve the repository or database as a static directory.

### Existing persistence

`data/ratio.db`: `cases`, `analyses`, `matter_catalogue`, `intake_files`, `matter_files`, `focus_reviews`, `case_notes`, `case_ai_briefs`, `local_comparisons`, plus existing reviewer/history tables. Originals are stored as BLOBs and linked to their owning case. Do not drop/recreate this database to make new code work.

`data/corpus/build.db`: acquisition metadata, normalized documents, model-extraction cache, verified facets and local vectors. `data/corpus/precedents.db`: separate read-only/searchable runtime reference index. Regenerate that index atomically with existing builder code; keep it out of the matter catalogue.

Intake limits remain 100 files/batch, 5 MB/file, 50 MB total. Textless PDFs need an OCR text layer; do not silently invent extracted text. Source documents assigned to a case are not reusable as another case's pending file IDs without a new explicit staging operation.

## 12. What Git does NOT transfer

The following are intentionally ignored and **are not present in a fresh clone just because this branch was pushed**:

```text
.env                          cloud credentials, never share through Git
data/ratio.db                 case workspace, originals, reviewer work and cached Maseko aid
data/corpus/                  49-document corpus, originals, local copies, vectors and reading aids
data/real-demo/                Maseko fidelity audit and public-reference download audit
models/                       MiniLM and local Ollama model files
output/pdf/                   generated reading-aid PDFs / QA images
output/demo-upload-pack/       separate demo originals and ZIP
tmp/pdfs/                     rendering intermediates
```

Current audits:

- `data/real-demo/real-maseko-makhubu-2014-source-audit.json`
- `data/real-demo/reference-download-audit.json`

The curated **public reference snapshot is now available through Git** in `public-references/`; install it with `python scripts/install_public_references.py`. Runtime `data/corpus/` remains ignored so later private collections/caches cannot enter Git accidentally. Do not send the entire workspace database, notes, analyses, embeddings of confidential content or `.env` as a shortcut. Case-specific data remains confidential even when some source documents originated in publication.

If Sahil's agent is on this same machine, the supplied folder is already available. On another machine, code alone cannot reconstruct missing original files. The agent should check for approved inputs, handle missing files explicitly and avoid fabricating content; the user does not want to be asked to annotate or prepare a new dataset manually.

### Separate live-upload pack

`scripts/fetch_demo_pack.py` creates `output/demo-upload-pack/Ratio-real-case-demo-pack.zip`, with unchanged official ECHR PDFs, `sources.json` (URLs/hashes/pages) and an upload guide. The three are:

- Butkevičius v. Lithuania, application 48297/99, 2002.
- Nešťák v. Slovakia, application 65559/01, 2007.
- Minelli v. Switzerland, application 8660/79, 1983.

They were verified absent from both installed collections by original-file hashes. Keep them held out for live intake; do not automatically seed them while rebuilding references. The archive's Butkevičius source URL has an incorrect 2003 label; the original judgment cover confirms 2002.

## 13. Run/rebuild instructions

From the repository root, with the existing Python setup:

```sh
.venv/bin/pip install -e '.[dev,corpus,demo]'
npm --prefix web ci
npm --prefix web run build
```

For a local Ollama server using this repo's already-downloaded model storage, in a separate terminal:

```sh
OLLAMA_MODELS="$PWD/models/ollama" OLLAMA_NO_CLOUD=1 ollama serve
```

If the model is genuinely absent, download it from the normal model registry; do not substitute a `:cloud` model. On this machine Ollama was installed through Homebrew and `qwen2.5:1.5b` was downloaded to the repo-local model directory. That binary/service/model is not distributed by the Git branch.

Start the primary interface:

```sh
RATIO_MODEL=qwen2.5:1.5b .venv/bin/python scripts/serve_workspace.py --port 8503 --local-ai
```

Use `--demo` only to add the three synthetic examples idempotently, and `--db PATH` for an isolated database. The environment assignment above applies to this command; do not globally change the core recorded-demo model accidentally. Ollama/web servers were started as local processes, not registered as guaranteed login services.

Import Maseko from approved available inputs:

```sh
.venv/bin/python scripts/import_real_demo.py --folder "$HOME/Downloads/Ratio-real-cases"
```

That imports/registers the case, **not** the one-off curated reading-aid cache entry. Address the reproducibility gap in section 6 rather than blindly trusting the weaker model to recreate it.

With public corpus source files already present:

```sh
.venv/bin/python -m corpus_builder normalize
.venv/bin/python -m corpus_builder embed
.venv/bin/python -m corpus_builder build-db
.venv/bin/python scripts/build_case_briefs.py
.venv/bin/python scripts/build_innocence_aids.py
```

`build_innocence_aids.py --only DOCUMENT_ID` can target one of its seven fixed IDs. It skips an existing source-bound aid whose method starts `Public-reference`. Check its logs: failed drafts are not published. Do not process the entire corpus immediately before a live demo and exhaust a free API quota.

If explicitly populating public-reference fact-pattern facets later, reuse `python -m corpus_builder extract --model groq --only DOCUMENT_ID`, followed by `verify`, `embed`, `build-db`. This is a separate operation from reading-aid generation. Private collections require the local model; never use our case as a “public reference” to make cloud extraction work.

The full Streamlit interface remains available via `streamlit run app/main.py`; the React court workspace is the primary demo interface. Both share the existing `ratio` engine and default store.

## 14. Verification evidence and acceptance checks

Latest Python-engine full suite on this branch: **1,656 passed, 6 skipped, 2 deselected**. This ran before the subsequent merge of Sahil's README/web-only UI changes; no Python engine code came from that merge. The earlier Groq baseline's 1,654 count is not the current count.

The timeboxed pass separately ran focused tests, fixed the new source registry expectations, and ended with 20 focused tests passing. Earlier focused API/citation checks also passed. Fresh frontend tests/build are rerun for the combined UI branch; see the final verification note below.

Checked against live data/API:

- Five visible matters; two real/published-origin, three synthetic.
- Maseko ID present in `/api/library`, 22 documents and 340 deterministic observations.
- Repeat import preserves the existing Maseko row.
- Maseko aid returns `ready`, provider `Ollama`, model `qwen2.5:1.5b`.
- Old Allenet/Sorin cloud caches return `not_generated` under the new routing.
- 49 reference documents, 5,546 passages, ten focused full decisions, ten saved focused aids.
- Maseko comparison returns zero items and an explicit abstention note.
- No supplied prior model results were trusted as current analysis.
- Runtime data and keys remain ignored.

Useful commands after integration:

```sh
.venv/bin/pytest -q
npm --prefix web run test
npm --prefix web run build
git diff --check
```

Focused subset:

```sh
.venv/bin/pytest tests/test_library.py tests/test_demo_briefs.py \
  tests/test_demo_briefs_api.py tests/test_workspace_api.py \
  tests/test_corpus_research.py tests/test_corpus_sources.py -q
```

Also inspect `tests/test_groq.py`, `tests/test_case_briefs.py`, `tests/test_architecture.py`, `tests/test_corpus_extract.py` and the network-guard/Ollama tests before extending routing. Tests isolate real corpus paths and use synthetic fixtures; do not point the suite at a user's case database.

Acceptance for the next demo milestone:

1. Clean clone + approved available inputs can reproduce Maseko's visible record and checked aid without user annotation.
2. Every displayed quote resolves to the stored text and, where asserted, the original PDF page/paragraph.
3. Published source origin and confidential processing role are distinguishable in labels, APIs and exports.
4. A private upload, a public-origin review matter and a case-derived comparison never construct a cloud client, including on local-model failure.
5. At least one carefully attributed real case/reference pair is shown, or the UI abstains honestly.
6. No report conclusion, answer key, dissent or general fair-trial finding is silently promoted to an Article 14(2) merits violation.
7. Existing originals, assessments, notes and history survive re-import/restart/merge.
8. Static hosting contains only synthetic demo material; local functionality is not represented as working on Vercel.

### Final combined-branch verification

Confirmed for the combined branch: all 13 frontend tests passed and the production build succeeded after installing Sahil's pinned UI/font dependencies. The Python full suite passed 1,656 tests (6 skipped, 2 deselected). The final changes after those checks are documentation/comment corrections, branch-specific preview configuration and formatting/bootstrap cleanup; no legal-analysis behavior was added during handover preparation.

## 15. First instructions for Sahil's coding agent

Read this file, the current source-routing docs and the implementation before editing. Merge the pushed handover branch with your existing work, preserving reviewer data. Inspect available local assets; distinguish code changes from the non-versioned dataset. Start the local app, confirm real counts, then work through P0/P1 in order.

Do not broaden cloud access, silently manufacture missing documents, trust precomputed case results, fill retrieval failures with invented interpretations, present selected precedents as a success predictor, or launch another large UI/infrastructure migration. The immediate objective is a small, coherent, source-backed demo that an actual human-rights lawyer can follow.
