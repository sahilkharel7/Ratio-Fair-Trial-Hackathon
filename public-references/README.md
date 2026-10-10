# Public legal reference snapshot

This intentionally versioned snapshot contains **49 public source PDFs**, their normalized text/metadata, **49 reading-aid PDFs and manifests**, and the matching public-only local search index. It was added at the user's explicit request so collaborators receive the actual documents through Git.

The source inventory is 3 ECHR judgments, 12 Human Rights Committee decisions, 11 WGAD opinions and 23 TrialWatch reports. Ten full decisions are the focused presumption-of-innocence set described in `../handover.md`. A report or WGAD opinion is not a court judgment; an inadmissibility decision is not a merits finding.

`manifest.json` records publication URLs, document identifiers, attribution, retrieval dates, original/text hashes and every payload-file checksum. Original PDFs are unchanged. Reading aids remain labelled editorial, extractive or AI draft as appropriate; verified quotes do not certify a model's interpretation. Third-party documents retain their authors' rights and source terms; the repository's software license does not relicense them or imply publisher endorsement.

No case workspace database, uploaded case documents, notes, assessments, confidential embeddings, API credentials or model weights are included. The separate three-PDF live-upload pack also remains held out.

From the repository root:

```sh
python scripts/install_public_references.py
```

The installer verifies checksums, restores the runtime layout expected by the existing app, and does not replace a different existing corpus. For an isolated installation:

```sh
python scripts/install_public_references.py --destination /path/to/reference-corpus
RATIO_CORPUS_DB=/path/to/reference-corpus/precedents.db python scripts/serve_workspace.py --port 8503 --local-ai
```

Local meaning search still requires the existing MiniLM model. Phrase search, saved text, original PDF downloads and reading aids require no cloud credentials.

This is a runtime snapshot, not the maintainer `build.db` or a complete builder-workspace backup. Do not overwrite the installed runtime index by running `build-db` against an empty build store. The public normalized JSON files are supplied for an explicit future builder restore; workspace case data is deliberately absent.
