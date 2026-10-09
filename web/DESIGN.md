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

The application retains the Ratio identity: charcoal chrome, burgundy actions,
blue research links, serif matter titles and source text, and compact sans-serif
controls. Status colors distinguish compliance, concerns, and unanswered review
questions. Fonts are system fonts, with no remote assets required.

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

The three versioned `ratio-demo-*` local-storage keys hold the working file,
notes, and 30 recent unique activities. Storage failure is disclosed in the UI;
the session state remains usable. Export review produces JSON, including local
notes and saved sources. There is no collaboration or cloud note storage.

Run `npm test` and `npm run build`. The tests check the synthetic bundle, every
evidence span, preserved results, quoted searches, and Unicode offsets. Browser
QA covers source highlights, saved passage/note persistence, copy feedback,
specific guarantee links, desktop views, and a 390px mobile iframe with no page
overflow. The temporary responsive test harness is not deployed.
