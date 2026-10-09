# Combined Ratio workspace

This integration merges Sahil's main at `a4bc765` (including stretch goals,
public-case fixes and the accessible native interface) with the redesigned
`codex/vercel-preview` at `2f8a60c`. Both histories are retained.

## Decisions based on the use case

| Area | Combined behavior |
|---|---|
| Analysis | The latest Python extraction, legal rubrics, renewal detector, provenance enforcement and conservative scoring from main remain unchanged. |
| Review ownership | The lawyer can inspect, accept, reword, reject, reopen, and record missed issues. Original findings and decision history are retained. |
| Offline app | Native nine-page navigation, accessible light/dark themes, status labels, distinct mark underlines and print support, combined with a matter-first overview and literal source search/full document reading. |
| React workspace | Compact legal typography, unified document/finding search, saved source passages, notes and history, with all nine case/review workflows visible. |
| Legal context | Jurisprudence is selected by the backend's standard/status rules. Model-generated State replies carry their original checks, quotes, unsupported grounds and caveats. |
| Reports | Local reports use the authoritative Python renderer and SQLite decisions. The public demo downloads that renderer's recorded report with an explicitly separate browser decision worksheet. |
| Hosted data | Only the committed synthetic nine-document matter and synthetic judicial history are exported. There is no cloud ingestion, model inference, SQLite access, or review synchronization. |

The web overview contains 13 case findings plus one judicial pattern prompt;
the latter remains a prompt for review and carries the selection-bias caveat.
The record contains three detention orders, one flagged repeated-ground order,
one gap between orders, 26 jurisprudence entries, and 11 possible-State-reply
records (two contain supported model arguments). These are recorded pipeline
outputs; the React interface does not reconstruct their legal assessments.

## Merge resolution

The main versions of the analysis views were retained so their legal-context
features and accessible source controls were preserved. The native configuration
retains main's contrast colors, bundled fonts, underlined links and light/dark
support; burgundy identifies actions. Custom overview components inherit that
native theme. A subprocess configuration test explicitly loads this source tree
so that it exercises the privacy settings independently of editable-install hooks.

## Validation

- Python suite: 774 passed; the two live-Ollama tests are excluded by the standard test configuration.
- Recorded demo evaluation: 18/18 expected outputs, 8/8 events and dates, no must-not-flag outputs, and exact source spans for all 13 case findings and the judicial prompt.
- Web tests cover bundle provenance, retained results, search, Unicode offsets, decision validation/history and exported worksheets.
- Browser checks cover source reading, the four new workflows, persisted/reopened demo decisions, report downloads and narrow-screen layouts.

See [INTERFACE.md](INTERFACE.md) for the local app and [web/DESIGN.md](../web/DESIGN.md)
for React extension patterns.
