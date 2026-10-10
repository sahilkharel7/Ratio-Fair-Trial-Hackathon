# Local Groq demo

Sahil's two latest main commits add the Ratio sidebar logo and PNG tab icon. They are included in this version.

The demo uses `openai/gpt-oss-120b`, a production model supporting structured JSON. Groq's [free plan table](https://console.groq.com/docs/rate-limits) currently lists 30 requests/minute, 1,000/day, 8,000 tokens/minute and 200,000/day. Actual account limits can differ. Nothing here upgrades a plan or enables paid service tiers.

## Start

Install the optional demo dependencies with `pip install -e '.[demo]'`. Configure `GROQ_API_KEY` only in ignored local `.env`; it must never become a Vite variable or browser setting. Keep that file readable only by its owner (`chmod 600 .env`).

```sh
npm --prefix web run build
python scripts/serve_workspace.py --port 8503 --groq-demo
```

Open a stored public case or synthetic example. Choose **Draft case brief**. The case stays usable while its draft is queued. When it is ready, open the supporting passages and download the one-page PDF. The generated sections and PDF are saved in the local SQLite database; unchanged cases reuse the saved draft. Regeneration is explicit.

The default server, without `--groq-demo`, remains offline. In demo mode a fixed separate worker sends selected source excerpts, the public case title and court to Groq. It does not receive the SQLite path, working notes, assessments, the whole database or arbitrary browser-supplied text. The server's loopback, origin and network guards remain installed. The key stays in the worker, and API/browser responses never include it. Private provenance is refused before starting a model request.

## What the brief means

Drafts use bounded selected excerpts, **not a complete reading of the case file**. Charges, recorded holdings, operative provisions and sentencing history are prioritized. Groq selects a numbered supporting passage for each section; it cannot write or combine its own quotations. The worker supplies the original words and offsets, checks them against the saved source, and the server verifies them again. Pinpoints are computed stored-text line references. These checks do not prove that the summary correctly interprets a quotation. A lawyer must review attribution, procedural stage, appeal history and the full document.

Requested penalties, statutory ranges, imposed sentences, compensation and costs must stay separate. A draft does not determine a violation or estimate the probability of international review succeeding. Model results do not overwrite confirmed case metadata or screening assessments.

Three drafts can be queued; one worker runs at a time. Groq rate limits receive at most one bounded retry. Failed, refused, oversized or unverified responses do not publish a PDF. A stopped job can be retried after restarting the server. The separate [held-out upload pack](DEMO-UPLOAD-PACK.md) stays outside the installed case and reference collections.

## Maintainer corpus commands

```sh
python -m corpus_builder extract --model groq --only DOCUMENT_ID
python -m corpus_builder verify
python -m corpus_builder embed
python -m corpus_builder build-db
python scripts/build_case_briefs.py --provider groq --only DOCUMENT_ID --force
```

Corpus extraction reads document windows and caches answers with a `groq:` model identity. A private collection still requires the local Ollama model. Groq reading sheets use explicitly selected excerpts; existing reviewed court summaries retain their editorial source checks. Do not exhaust the free daily token budget by processing the entire corpus immediately before a live demo.
