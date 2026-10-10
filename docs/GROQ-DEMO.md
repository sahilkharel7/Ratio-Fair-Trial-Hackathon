# Model routing

Case workspace content, uploaded PDFs, notes, analyses and case comparisons use local Ollama only. Uploaded files default to confidential. --local-ai enables local reading aids; the legacy --groq-demo flag is now a local-only alias. There is no case-to-cloud fallback.

Only explicitly designated public legal references in the reference corpus may use the configured Groq provider. Public reading aids fall back to local Ollama when cloud processing is unavailable. Quote and PDF-page checks do not certify a model interpretation.

The installed local model is qwen2.5:1.5b. It is suitable for selecting reading passages; generated legal interpretations are not trusted as findings. The Maseko aid uses checked source facts instead.
