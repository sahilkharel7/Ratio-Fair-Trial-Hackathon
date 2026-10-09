# Vercel demonstration

A React/Vite legal workspace combined with Sahil’s latest `main`, using analysis
exported from the committed synthetic case and judicial history. The original
Python application remains the place to import cases and run the local model.

This preview includes all nine case/review views, a unified document/finding search,
filtered research results, saved documents and exact passages, research history,
browser-local working notes, source highlights, judgment/indictment comparison,
coded judicial history, detention renewals, linked jurisprudence, possible State replies,
and reviewer decisions with an append-only history.
It does not accept uploaded cases or run a model in the cloud. The hosting status
and About dialog explain that distinction to visitors.

Saved research, notes, and the last 30 unique search/document activities persist
in this browser's local storage. Review decisions and missed issues use separate
versioned storage keys. They do not synchronize across devices or users.

**Report draft (.md)** contains the Python pipeline report followed by the browser
review worksheet, with original wording, decisions, reasons and exact source quotes.
The original pipeline report is preserved; browser decisions are clearly labelled
separately. **Review worksheet (.json)** contains decision snapshots, missed issues,
notes and saved sources. For authoritative reports with decisions applied directly
to the findings, use the offline Python application.
See [DESIGN.md](DESIGN.md) for interface references and reusable view patterns.

## Refresh the synthetic record

From the repository root, with Python dependencies and MiniLM installed:

```sh
python scripts/export_web_demo.py
```

The exporter reads only `data/demo/`, validates every case as synthetic, and
runs the existing provenance-checked analysis pipeline. It never opens SQLite,
uploaded files, or the user's runtime model cache.

## Build and preview

```sh
cd web
npm ci
npm run build
npm run preview
```

## Deploy on Vercel

Use a separate project (for example `ratio-trial-review-preview`) with root
directory **web**, framework **Vite**, build command **npm run build**, and output
directory **dist**. Use the branch **main** after the integration is merged. No environment
variables, database, Python runtime, analytics, or model downloads are needed.

The `vercel.json` SPA rewrite supports direct links to review pages. The committed
`package-lock.json` pins the web dependencies. Deployment state in `.vercel/` and
build output in `dist/` are ignored.
