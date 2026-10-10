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

## Visual design

The theme (`src/theme.css`, loaded last) is modelled on lexisnexis.com while
keeping Ratio's own name and mark: Lato (bundled from `@fontsource/lato`, so no
remote assets are required), a white header with uppercase navigation, dark-navy
headings, a red to orange-red hero, cream surfaces, and soft rounded cards. Quoted
source text keeps a Georgia reading serif. Status colours distinguish compliance,
concerns and unanswered review questions.

- **Simple first.** The main view shows the person, the charge, what is at stake
  and the review. Further detail (penalty history, method notes, the supporting
  analysis pages) sits behind expandable sections; "Supporting analysis" opens
  automatically on its own pages. Safety caveats are always shown.
- **Type.** Titles, labels and current navigation use heavy Lato (700 or 900) with
  slightly tight tracking; body copy and sidebar links stay at regular weight.
  Body text is at least 15px and labels at least 12px; text on light surfaces
  meets WCAG AA (use `var(--muted)` for secondary text). Expandable sections use
  a rounded chevron.
- **Depth.** One warm elevation scale (`--elev-1`, `--elev-2`, `--elev-sm`,
  `--elev-sm-hover`, `--elev-inset`): cards sit just above the canvas, fields and
  quoted passages sit slightly into it, dialogs float highest. Use these tokens
  for new surfaces.
- **Interaction.** On hover, surfaces deepen their shadow and warm slightly but do
  not move; the only movements are the sidebar item's short slide and the arrow on
  a text link. Buttons press in with an inset shadow. Keyboard focus shows a soft
  warm ring.
- **Motion.** The collection hero greets by time of day, types its title with a
  terminal cursor that blinks and then disappears, and counts up its key numbers
  (`TypedTitle`, `CountUp` in `src/motion.jsx`). Case rows arrive in sequence,
  cases needing attention carry a pulsing amber dot, and sections rise into view
  (`src/effects.mjs`). All motion stops under `prefers-reduced-motion`.
- **Layout.** On desktop the sidebar collapses to an icon rail (stored as
  `ratio-ui-rail-v1`); link names remain as tooltips. The workspace status card at
  the foot of the sidebar opens the About panel. On a case page, the case chat
  (`src/case-chat.jsx`) is a launcher in the bottom-right corner that opens a
  floating panel; it keeps the local-only notice, the unverified-answer label and
  the source citations, and the conversation stays in memory only. Primary actions look like
  buttons, and empty sections collapse to a one-line note that says how to fill
  them.

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

The reference collection and paragraph-search code remains available in the
local Streamlit Precedent library. The active matter catalogue is a separate
collection: reference/private corpus documents are not exported to the public site
or silently registered as active defendants' cases.
