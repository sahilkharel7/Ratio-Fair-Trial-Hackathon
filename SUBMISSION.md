# Ratio: written description

## What Ratio is

Ratio is an open-source tool that turns a trial-monitoring record into findings a TrialWatch lawyer can check line by line, each linked to the exact sentence it rests on. It checks the monitor's notes and the court documents against ICCPR Articles 9 and 14. The legal evaluation stays with the lawyer.

## The problem it addresses for TrialWatch

Turning monitoring notes into a fairness report is slow. The notes are long and unstructured, violations hide in omissions (an interpreter never mentioned), and delays must be worked out by hand from scattered dates. Every conclusion must also be traceable to the record.

## How TrialWatch would use it

1. **Add the case.** In the court case workspace (http://127.0.0.1:8503), upload the documents (scanned PDFs are read with on-device OCR), record whether they are confidential (the default), public (with the source) or synthetic, and assign them to one case.
2. **Check the facts and Article 14(2).** The charge, requested penalty, sentence and quoted statutory penalty appear separately, each with its source passage when one is found. For each Article 14(2) passage Ratio surfaces, the lawyer records an assessment with a reason, compares international decisions and exports a worksheet. The local model can also set the case's passages beside public references.
3. **Run the full analysis.** In the full analysis app (http://127.0.0.1:8501), upload the case folder; the local model reads a case the size of the demo in a few minutes.
4. **Find context.** Similar cases lists past cases that share the case's fact patterns, from a library of public decisions and reports built on this computer; the Precedent library searches that library by facts. A team can add its own collections to it; a private one is read only by the local model. In the workspace, Triangulation sets the case beside the most similar past public cases on a chosen right, with the facts that matter linked to their sources on both sides, and a local chat answers questions about the case with cited passages.
5. **Review and draft.** On the review page the lawyer accepts, rewords or rejects each finding (with a reason to reword or reject) and adds issues Ratio missed. The Markdown report draft shows each decision and quotes each finding's source text in an annex.

## What it checks

- **Rights coverage**, Art. 14(3)(a)–(g): evidence for or against each guarantee; where the notes are silent, a follow-up question for the monitor, not a finding.
- **Delays**: the only confirmed benchmark is General Comment 35's 48 hours from arrest to a judge (para. 33); other intervals are measured, never marked as exceeded.
- **Detention renewals**, Art. 9: orders that mostly repeat earlier grounds, and days with no order in the record.
- **Reasoning copied from the indictment**, Art. 14(1)–(2): copied or paraphrased passages, their share of the court's reasoning, and unanswered defence arguments. Quoted law, the recited charge and party arguments are not counted.
- **Presumption of innocence**, Art. 14(2): possible burden reversal, public statements of guilt before judgment, a defendant restrained in court (prompts for review, not findings).
- **Judge patterns**: rates from rulings coded by people, with sample sizes and 95% intervals, against the same court and charge type; hidden below 5 cases; "pattern that warrants review" only if significant after correcting for the number of indicators. Every judge page warns that TrialWatch's cases are not representative.

## Responsible AI

- **Provenance.** Every finding carries an exact source span, re-checked before display; one that fails is dropped.
- **No verdicts about people.** Findings use fixed wording templates; model text is labelled unverified or as a draft needing legal review.
- **Offline.** The model runs locally, both screens listen only on this computer, and a network guard blocks outside connections. Case content never goes to a cloud model; Gemini and Groq are used only by maintainer tools, on public documents.
- **Data.** Uploaded case documents are confidential by default and stay on this computer, in a git-ignored store read only by the local model. The repository holds only public or synthetic material, and synthetic material is marked SYNTHETIC.
- **Review.** Team-drafted standards are labelled "needs legal review". The lawyer decides.

## Fit with TrialWatch's infrastructure

Ratio runs on one laptop (a Mac with Apple Silicon, 16 GB of memory, about 8 GB of free disk), with no separate server and no cloud account for case data. A one-time online setup installs Python 3.11 packages, Ollama with an open Qwen2.5 model and a small embedding model, and builds the workspace with Node.js. Ratio's code is Apache-2.0; on macOS everything it installs is open source.

## Evidence and limitations

On the synthetic demo case, the evaluation (run on 10 October 2026) found 18 of 18 expected outputs, 0 violations of 11 must-not-flag traps, all 8 events of the hand-checked timeline with correct dates, and exact source spans for all 13 findings. 1,672 automated tests pass.

- The evaluation is in-sample: the team wrote the case and its expected outputs. Recall on published TrialWatch reports is not yet measured. One local run on a real public trial exposed two bugs, since fixed.
- English only; scanned pages are read with on-device OCR, which can misread poor scans.
- The rubric, the unconfirmed benchmarks and the ruling codes await legal review.
- The small local model makes mistakes; an event it misses is absent.
- Judge statistics have run only on synthetic history.

## Team and links

Komal Neupane, Sahil Kharel, Hardik Kafle and Emeric Chang built Ratio for the FairTrial AI Hackathon, Track 2.

Code: https://github.com/sahilkharel7/Ratio-Fair-Trial-Hackathon. Setup and usage: [README.md](README.md).
