# Interface design

Ratio uses a restrained legal research workspace: a persistent review navigation,
serif matter titles and document text, compact sans-serif controls, white reading
surfaces, and navy text with a burgundy action accent. The case overview leads with
the matter and its analysis; importing another matter and model diagnostics sit
below the review. All summaries are computed from the loaded record.

## References

The design draws on the task structure, overview panels, document comparisons,
and visible evidence signals in these official LexisNexis resources:

- [Lexis+ product overview](https://www.lexisnexis.com/en-us/products/lexis-plus.page)
- [Lexis+ Brief Analysis overview and interface screenshots](https://www.lexisnexis.com/pdf/lexisplus/experience-brief-analysis.pdf)
- [Lexis+ support and training](https://www.lexisnexis.com/en-us/support/lexis-plus/default.page)

Ratio retains its own identity. Georgia and Arial/Helvetica are local system-font
choices, not a claim to reproduce LexisNexis's proprietary typography. No remote
fonts, images, scripts, or stylesheets are loaded by the app.

## Extending the offline interface

The native page navigation, light/dark palettes, bundled serif headings,
contrast settings, focus outlines, underlined links, print styles and distinct
mark underlines remain in place. Both `.streamlit/config.toml` files must match.
The offline app uses native controls, status badges and review forms.

Use `widgets.header()` and `widgets.evidence()` for analysed-case pages, with
`widgets.jurisprudence()` and `widgets.state_reply()` for the linked context.
`style.panel()`, `style.metric()` and the product classes in `theme.css` support
matter summaries and workstreams while inheriting the native theme. Source readers
retain exact-span checks. New views must be registered in `app/main.py`.

## Two interfaces

The React court case workspace in `web/` is the main interface: `scripts/serve_workspace.py`
serves the built app with a loopback-only API over the same SQLite store (`ratio/library.py`,
`ratio/workspace.py`), including uploads, original files and saved assessments. The Streamlit
app keeps the full analysis pages, the Precedent library and the finding review. Without the
local API, the React app opens only the committed synthetic examples.

## Verification

Run `pytest tests/test_app.py tests/test_privacy_config.py tests/test_display.py
tests/test_config.py tests/test_architecture.py`. The app suite checks all review
pages, synthetic/public separation, evidence buttons, original-source validation,
document search, and the full document reader. It requires the local MiniLM model.
Also inspect the running dashboard at desktop and narrow widths after modifying
the native adapter; headless app tests do not validate browser layout.

## Focused case-collection entry point

The default page is now `app/views/case_collection.py`, with SQLite-backed matters
and staged PDF assignment. `app/views/focus.py` opens the selected record, source-backed
charge and penalty statements, Article 14(2) screening prompts and reviewer notes.
Supporting analysis contains the original pages and the Precedent library
(`app/views/library.py`); the two libraries are intentionally different workflows.
See [COURT-WORKSPACE.md](COURT-WORKSPACE.md) for API, ingestion and outcome-coding
contracts, and [web/DESIGN.md](../web/DESIGN.md) for the React components.
