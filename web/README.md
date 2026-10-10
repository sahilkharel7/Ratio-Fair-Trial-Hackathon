# React court case workspace

The primary interface is a court-filtered case collection, followed by a focused
Article 14(2) review. Charge, requested penalty, imposed sentence and quoted law
remain separate, each with a checked source. The original broader analysis views
remain under Supporting analysis.

## Real local uploads and SQLite

```sh
npm --prefix web ci
npm --prefix web run build
.venv/bin/python scripts/serve_workspace.py --demo
```

Open http://127.0.0.1:8503. The loopback API uses the existing `data/ratio.db`.
Upload PDFs/TXT/MD, assign the originals to one case with confirmed roles, then open
that stored case. Assessments retain their source snapshots and history; working
notes and originals survive a restart. `--db PATH` selects a separate database.
See [the court-workspace guide](../docs/COURT-WORKSPACE.md) for limits and the adapter
contract used by future bulk grouping/OCR/classification work.

## Vercel synthetic preview

The same interface detects the absence of the local API and opens three committed
synthetic examples. Actual uploads use the local server. Focused demo assessments
and notes stay in browser storage. Official UN decision examples are separately
marked as research references; they are not synthetic case evidence.

The international outcomes show a selected merits fraction, explicit denominator
and source coding. Inadmissibility is separate; a narrow sample below five merits
decisions shows counts without a percentage. A case-specific success probability
is not estimated. These selected examples do not establish release or acquittal.

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
