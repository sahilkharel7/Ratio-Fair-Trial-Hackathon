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

Sahil's native page navigation, light/dark palettes, bundled serif headings,
contrast settings, focus outlines, underlined links, print styles and distinct
mark underlines remain in place. Both `.streamlit/config.toml` files must match.
The offline app uses native controls, status badges and review forms.

Use `widgets.header()` and `widgets.evidence()` for analysed-case pages, with
`widgets.jurisprudence()` and `widgets.state_reply()` for the linked context.
`style.panel()`, `style.metric()` and the product classes in `theme.css` support
matter summaries and workstreams while inheriting the native theme. Source readers
retain exact-span checks. New views must be registered in `app/main.py`.

## Framework decision

The current application is Streamlit. Upload ingestion, local analysis, SQLite
state, multipage navigation, and source dialogs all use its Python session model.
A React migration would require a separate local API and new upload, session,
and evidence-dialog implementations. The combined version keeps those tested workflows and the latest analysis engine.
A separate React preview in `web/` exports only the committed synthetic record; it
provides browser-local demonstration reviews without moving case uploads to a cloud service.

## Verification

Run `pytest tests/test_app.py tests/test_privacy_config.py tests/test_display.py
tests/test_config.py tests/test_architecture.py`. The app suite checks all review
pages, synthetic/public separation, evidence buttons, original-source validation,
document search, and the full document reader. It requires the local MiniLM model.
Also inspect the running dashboard at desktop and narrow widths after modifying
the native adapter; headless app tests do not validate browser layout.
