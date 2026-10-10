# Model routing

Case workspace content, uploaded PDFs, notes, analyses and case comparisons use local Ollama only. Uploaded files default to confidential. --local-ai enables local reading aids; the legacy --groq-demo flag is now a local-only alias. There is no case-to-cloud fallback.

Groq is used only by maintainer builders run by hand, on designated public references: `scripts/build_case_briefs.py --provider groq`, `scripts/build_innocence_aids.py` and `python -m corpus_builder` with `--model groq` (refused for private collections). The workspace and the analysis app never call it; `scripts/groq_case_worker.py` is local-Ollama-only despite its name. `build_innocence_aids.py` falls back to local Ollama (qwen2.5:1.5b) when `GROQ_API_KEY` is not set or a Groq draft fails. Quote and PDF-page checks do not certify a model interpretation.

Two local models are installed (see the README's Setup). qwen2.5:1.5b is the default for the workspace's reading aids and reference comparisons (`ratio/demo_briefs.py`, `ratio/local_comparisons.py`); Triangulation, Ask about this case and the analysis app use qwen2.5:7b-instruct (`ratio/config/settings.yaml`). `RATIO_MODEL` overrides both. Generated legal interpretations are not trusted as findings. The Maseko aid uses checked source facts instead.
