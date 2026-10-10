# Court judgments, international review and the local corpus

The homepage begins with actual court judgments. Opening a judgment shows its charge, recorded consequences, focused issue, conclusion and source pinpoints. A historical judgment is reference material; creating a new client matter still requires confirming the defendant, court and document roles in intake.

The local installation contains 43 documents and 5,260 paragraph vectors:

| Source type | Documents | Use |
| --- | ---: | --- |
| ECHR court judgments | 3 | Original judicial reasoning and contrasting Article 6(2) outcomes |
| UN Human Rights Committee Views | 6 | Treaty-body review decisions; inspect admissibility and the specific article examined |
| Working Group on Arbitrary Detention opinions | 11 | Arbitrary detention review; an opinion is not a court judgment |
| TrialWatch monitoring reports | 23 | Context and factual comparisons; a monitor assessment is not a judicial finding |

Forty documents from Sahil's source list downloaded successfully. One listed Lopez Mendoza/Tintori Parra PDF returned 404 and is absent. The three new court judgments were downloaded from official UN reproductions after HUDOC's PDF conversion endpoint returned 403. The issuing court remains the ECHR; hosting on a UN website does not turn a judgment into a UN decision. Downloads preserve original PDF bytes and raw/text checksums.

## Focus for the demonstration

Keep presumption of innocence as the single review focus. An audit of the first forty normalized documents found that 24 contain the exact phrase across whitespace: 19 monitoring reports, four WGAD opinions and one Committee decision. This is a wording audit, not 24 established violations. Matches can occur in headings, legal quotations, party submissions and procedural descriptions.

The useful questions are whether a public official declared guilt before adjudication, whether the court shifted the burden of establishing innocence, and how the defendant was presented. Compare facts and procedural stage before applying an authority. The corpus also contains substantial detention and expression material; those topics provide context without adding another dashboard of scores.

The three starter judgments intentionally show different outcomes:

- **Allenet de Ribemont v. France**, application 15175/89, 10 February 1995: public official statements; Article 6(2) breach. Charge and custody history are in paragraph 12; the reasoning is in paragraphs 35-41. Compensation is not a criminal penalty.
- **Daktaras v. Lithuania**, application 42095/98, 10 October 2000: prosecutor's language considered in its procedural context; no Article 6(2) breach. The separate Article 6(1) breach must not become a presumption-of-innocence success. Paragraphs 17 and 19 record the domestic sentence, not a prosecution request.
- **Salabiaku v. France**, application 10519/83, 7 October 1988: statutory presumptions and defence rights; no Article 6(2) breach. Paragraphs 13-14 distinguish the initial conviction from the appeal; the initial two-year prison term must not be presented as current exposure.

These are regional Article 6(2) cases. Committee review under ICCPR Article 14(2) remains a separate forum and outcome cohort. The existing seven manually coded Committee outcomes remain separate from this uncoded reference corpus. No wording match, semantic score or selected precedent produces an individual probability of success.

## Official directories

- [OHCHR JURIS](https://juris.ohchr.org/): treaty-body decisions; select CCPR and Article 14(2), then distinguish merits from inadmissibility.
- [HUDOC](https://hudoc.echr.coe.int/): filter Article 6-2, English and judgment/decision type, then verify violation or non-violation.
- [OHCHR collection of full ECHR judgments](https://cambodia.ohchr.org/en/rule-of-law/echr-decisions?items_per_page=60): a larger directory of original decisions reproduced by OHCHR Cambodia, with language and date labels.
- [ECHR presumption of innocence guide](https://ks.echr.coe.int/documents/d/echr-ks/presumption-of-innocence): a Registry reading guide, updated February 2026, with leading case references. The guide itself is not a judgment.

## One-page reading aids

Run `pip install -e ".[pdf]"` and `python scripts/build_case_briefs.py` after building the corpus. The maintainer script creates saved one-page PDFs and manifests in ignored `data/corpus/briefs/`; the three court summaries are also copied to ignored `output/pdf/` for review. It validates each citation against the exact saved source text and verifies each PDF is one page. The runtime serves these saved outputs without importing a cloud SDK or PDF renderer.

Editorial court summaries are bound to the exact source hashes checked in `ratio/config/court_briefs.json`. Other documents receive extractive reading sheets labelled as reading leads, without inferred outcomes. Extracts retain stored text line references and surrounding source context is available in the reader.

Gemini drafts are an explicit maintainer operation: configure a working `GEMINI_API_KEY` in ignored `.env`, install `.[corpus,pdf]`, then run `python scripts/build_case_briefs.py --model-id MODEL_ID --force --only DOCUMENT_ID`. Choose a supported model using the existing builder's model-list method. The draft prompt distinguishes allegations, holdings, requests, sentences and appeal history. Invented supporting quotes are rejected. Exact quote matching does not validate the meaning of a model summary: legal review is still required. Private documents are refused before any model call. Full documents exceeding the input bound are refused rather than silently truncated.

The supplied credential failed the initial Google models check with `API_KEY_INVALID`. It was not saved and no case text was transmitted for that check. Automated model extraction and model summaries have not been claimed as completed.

## Sahil's integration

The latest published main contained `378cb33` (Similar cases) and `f820a0a` (Precedent library), already included in the merged workspace. This update exposes their downloaded corpus, local paragraph vectors and original PDFs in React. Normalized documents with paragraph embeddings can now enter the runtime index even before model facets are available. Phrase search preserves original character offsets; meaning search reuses the existing local ranking and filtering. Fact-pattern matching still requires the separate extract/verify/embed pipeline.

Uploaded matter PDFs stay in the SQLite intake store; source PDFs can be copied into intake while retaining their publication URL and checksum. The lawyer confirms grouping before creating a case. The server remains loopback-only and rejects cross-origin requests. Nothing in this update deploys the real corpus or an API key to Vercel.

The local integration branch has automatic Vercel deployment disabled in `web/vercel.json` using the [documented branch setting](https://vercel.com/docs/project-configuration/git-configuration). The published static preview remains a separate synthetic demonstration.

## Fresh uploads for the demo

[The held-out demo pack](DEMO-UPLOAD-PACK.md) contains three additional original court judgments. Its downloader uses a temporary download store, never registers corpus seeds, and never inserts case rows. Actual PDF uploads were checked in a temporary SQLite database; the default case collection and reference corpus remain untouched. The pack includes source URLs, page counts and original-file checksums.

PDF line wrapping and mixed year/month prison terms are handled by the focused screen. A review judgment that explicitly discusses the presumption of innocence can produce up to three general reading leads when no specific screening pattern is found. Such discussion is not treated as a violation or a fact pattern for filtering the outcome cohort. For historical judgments, inspect the full appeal history and the subject of each sentencing statement before using a term as current exposure.
