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

## Extending the interface

1. Add a view under `app/views/` and register it in `app/main.py`.
2. Add its navigation entry in `style.sidebar()` in `app/ratio_ui/style.py`.
3. Use `widgets.header()` for analysed-case pages. Use `style.section()` for
   sections, `style.panel(unique_key)` for reading and review panels, and
   `style.metric(label, value, note, tone)` for summaries. Panel keys must be
   unique within a rendered page, including any source dialog.
4. Use `widgets.badge()` for the reviewed status vocabulary and
   `widgets.evidence()` for evidence rows. Keep findings, missing evidence,
   and legal-review status distinct. Burgundy denotes actions, not a legal finding.
5. Use `viewer.show_source()` for an exact evidence span and
   `viewer.show_document()` for the complete original document. The source viewer
   continues to check spans against their original documents before displaying them.
6. Adjust colors, font stacks, radii, and spacing in the `--ratio-*` tokens in
   `app/ratio_ui/theme.css`. Keep both `.streamlit/config.toml` files identical;
   their palette also styles native controls and tables across theme preferences.

The stylesheet separates product classes (`ratio-*`) from the native Streamlit
adapter (`data-testid` and explicit container keys). It avoids generated framework
class names. Review the adapter after upgrading Streamlit. Responsive rules adjust
summary grids and reading layouts; keyboard focus and reduced-motion preferences
are retained.

## Framework decision

The current application is Streamlit. Upload ingestion, local analysis, SQLite
state, multipage navigation, and source dialogs all use its Python session model.
A React migration would require a separate local API and new upload, session,
and evidence-dialog implementations. This redesign keeps those tested workflows
and introduces a shared presentation layer without changing the analysis engine.

## Verification

Run `pytest tests/test_app.py tests/test_privacy_config.py tests/test_display.py
tests/test_config.py tests/test_architecture.py`. The app suite checks all review
pages, synthetic/public separation, evidence buttons, original-source validation,
document search, and the full document reader. It requires the local MiniLM model.
Also inspect the running dashboard at desktop and narrow widths after modifying
the native adapter; headless app tests do not validate browser layout.
