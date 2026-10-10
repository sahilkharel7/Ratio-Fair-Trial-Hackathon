# Legal research workspace

The preview follows familiar legal research conventions: prominent matter search,
content and document-type filters, citations alongside results, saved research,
history, and an original-source reader. The overview presents the matter first,
then a source-linked review queue and a working file. Evidence of violation,
confirmed benchmark exceedances, and unanswered arguments precede compliance
findings; their underlying recorded statuses are preserved.

Official resources consulted:

- [Lexis result filters and document navigation](https://www.lexisnexis.com/en-us/products/lexis/feature-right-results.page)
- [Lexis search history and folders](https://www.lexisnexis.com/en-us/support/lexis/faqs/default.page)
- [Westlaw Precision folders and search within folders](https://www.thomsonreuters.com/en-us/help/westlaw-precision/setup/folders-in-westlaw)

The visual theme (`src/theme.css`, loaded last) is modelled on lexisnexis.com:
Lato (bundled from `@fontsource/lato`, so nothing is fetched remotely), a white
header with uppercase navigation, dark-navy headings in sentence case, a purple
hero on the collection page, white cards with 8px corners and soft shadows, 4px
buttons outlined in navy, and a pink-to-red gradient call to action. Quoted source
text keeps a Georgia reading serif. `src/effects.mjs` adds the header shadow on
scroll and sections that rise into view; with `theme.css` it supplies hover lifts
and a typed page title, all disabled under `prefers-reduced-motion`. Ratio keeps
its own name and mark. Status colors distinguish compliance, concerns, and unanswered review
questions. No remote assets are required.

Readability floor: body text is 15px, and labels, metadata and badges are at
least 12px (11px only for small counters). Text on light surfaces meets WCAG AA
(4.5:1); use `var(--muted)` for secondary text rather than a lighter grey.
Primary per-item actions (such as recording an assessment) look like buttons,
not small disclosure text. Empty sections collapse to a one-line note that says
how to fill them, so they never push the active work below the fold.

## Extend the preview

- `main.jsx` contains the application shell, shared Source context, recorded
  analysis views, validated source reader, and browser storage. Add a route to
  the navigation and renderer together. Treat legal review statuses as record
  data, rather than recomputing legal conclusions in the frontend.
- `workspace.jsx` contains reusable research and workspace components: Icon,
  ResearchBar, MatterHeader, ReviewQueue, ReviewRail, Research, and SavedWorkspace.
  Documents and exact passages use the shared context's open/save actions.
- `styles.css` contains the base typography and evidence/analysis primitives.
  `workspace.css` contains the shell, search, queue, reader, and responsive
  layout. Use shared color/font/spacing tokens and these primitives for new views.
- `research.mjs` contains literal multiword/quoted-phrase search, Unicode source
  offsets, stable saved-source keys, and source citations. All entered words
  must match. Search is confined to the exported matter; it is not a Boolean
  search engine or an external legal research service.

The source reader checks an exact span against the original document before
displaying it. Save only the original document identifier and span, and derive
source citations from that document. Preserve separate source, context, recorded
finding, and unverified model-note labels.

## Persistence and verification

The five versioned `ratio-demo-*` local-storage keys hold the working file,
notes, 30 recent unique activities, review history and missed issues. Storage failure is disclosed in the UI;
the session state remains usable. Review exports a Markdown draft and a JSON worksheet, including local
notes and saved sources. There is no collaboration or cloud note storage.

Run `npm test` and `npm run build`. The tests check the synthetic bundle, every
evidence span, preserved renewal/reply/jurisprudence outputs, quoted searches,
Unicode offsets, review validation, reopening and safe report text. Browser
QA covers source highlights, saved passage/note persistence, copy feedback,
specific guarantee links, desktop views, and a 390px mobile iframe with no page
overflow. The temporary responsive test harness is not deployed.

## Combined legal workflows

`legal-workflows.jsx` adds the original pipeline's detention renewal comparisons,
linked jurisprudence and possible State replies, together with a browser demo
review interface. `workflow.mjs` handles only the demonstration decisions, audit
history, validation and downloads; it does not calculate legal findings or scores.
`legal-workflows.css` extends the shared primitives. `scripts/export_web_demo.py`
produces the static source marks, standard-based reference selections and the
original Python report. Browser edits are shown in an explicit worksheet appended
to that report, preserving the recorded analysis as well as the reviewer wording.

The English wording guard removes invisible characters, applies NFKC and the
Python-exported Latin/Greek/Cyrillic transliteration map before applying the
existing block list. Full authoritative review validation remains in the offline
Python application. Judge prompts carry the case-selection caveat wherever their
rates appear. Neither browser decisions nor model replies silently replace a
recorded finding.

## Court-level primary workflow

`case-library.jsx`, `case-library.css` and `library-client.mjs` define the primary
collection, one-case Article 14(2) worksheet, claim-specific outcome reading and
document-intake UI. The primary shell has three navigation entries; the original
analysis views remain in a collapsed supporting section. `scripts/serve_workspace.py`
provides the actual loopback API against `LibraryStore` and the original `CaseStore`.
The Vercel preview uses the committed synthetic collection export.

Charges use recorded wording when it is found; requested, statutory and imposed
penalties remain separate. Ambiguous specific terms require source review. No
screening passage or missing passage is labelled as an automatic violation. Browser
assessment snapshots are validated against the current source prompt before a
local decision is saved. Historical rates are scoped to the coded Article 14(2)
merits cohort and are never shown as an individual success prediction.

Sahil's reference collection and paragraph-search code remains available in the
local Streamlit Precedent library. The active matter catalogue is a separate
collection: reference/private corpus documents are not exported to the public site
or silently registered as active defendants' cases.
