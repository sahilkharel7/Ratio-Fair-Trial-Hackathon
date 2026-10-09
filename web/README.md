# Vercel demonstration

A React/Vite preview of the `frontend-improvements` dashboard, using analysis
exported from the committed synthetic case and judicial history. The original
Python application remains the place to import cases and run the local model.

This preview includes all five review views, searchable original documents,
source highlights, judgment/indictment comparison, and coded judicial history.
It does not accept uploaded cases or run a model in the cloud. The hosting status
and About dialog explain that distinction to visitors.

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
directory **dist**. Use the branch **codex/vercel-preview**. No environment
variables, database, Python runtime, analytics, or model downloads are needed.

The `vercel.json` SPA rewrite supports direct links to review pages. The committed
`package-lock.json` pins the web dependencies. Deployment state in `.vercel/` and
build output in `dist/` are ignored.
