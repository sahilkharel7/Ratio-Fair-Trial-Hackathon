"""Builds the precedent corpus for "Similar cases": online, run by a maintainer, never by the app.

    python -m corpus_builder fetch|normalize|extract|verify|embed|build-db|index-opensearch|status|snapshot|restore

Only public documents from the sources in sources.yaml are fetched (robots.txt and rate limits
respected) and sent to the cloud model. This package never reads the case store, never imports the
app's model client, and is the only code that reads an API key. Downloads are untrusted data: they
are parsed, never executed.
"""
