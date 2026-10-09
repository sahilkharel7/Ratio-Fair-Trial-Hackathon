# Ratio

Ratio reads a trial's monitoring record and gives the reviewing lawyer findings to check line by line, each linked to the sentence it rests on. The legal evaluation stays with the lawyer. We built it for the FairTrial AI Hackathon (Track 2: Observation to Legal Evaluation), run by Columbia Law School's Human Rights Institute with the TrialWatch project.

It runs four checks against the international fair-trial standards (ICCPR Articles 9 and 14):

| Module | Question it answers | What you see |
|---|---|---|
| Absence Detector | Which guarantees of ICCPR Art. 14(3)(a)–(g) have no evidence of compliance? | A rights coverage grid. Where the notes say nothing, a follow-up question for the monitor instead of a finding |
| Procedural Clock | Were detention and trial delays excessive? | A timeline with measured intervals, red only where the one confirmed benchmark (48 hours to see a judge) is exceeded |
| Reasoning Reuse Detector | Did the court reason independently of the prosecution? | The judgment beside the indictment, with copied and paraphrased passages highlighted, the share of the court's reasoning traceable to the indictment, and defence arguments the judgment never answers |
| Judicial History Tracker | Do a judge's rulings across monitored cases show a pattern that warrants review? | A judge profile: rates with sample size and confidence intervals, compared with other judges of the same court and charge type |

Every finding links to the exact sentence it rests on: click it and the source document opens with that passage highlighted. Everything runs on one laptop with open-source software, and nothing is sent to another computer.

The name comes from *ratio decidendi*, the reasoning behind a decision. Ratio's central question is whether the court did its own reasoning.

![The Ratio case overview, with matter details, source summaries, and evidence-led review workstreams for the synthetic demo](docs/screenshots/08-case-redesign.jpg)

## The problem

TrialWatch lawyers review large volumes of monitoring notes, transcripts and court documents, and the hackathon's Track 2 brief calls that review a bottleneck. Before judging whether a trial was fair, a lawyer has to reconstruct what happened from that record: the notes are long and unstructured, and the documents arrive in mixed formats. Three things make the review slow and inconsistent:

- **Violations hide in omissions.** A missing interpreter or late access to counsel never appears as a sentence in the notes. A reviewer has to notice what isn't there.
- **Delay is arithmetic buried in prose.** Deciding whether detention or trial delay was excessive means pulling dates from scattered entries and comparing them by hand.
- **There is no memory across cases.** Each case is reviewed on its own, so patterns by the same judge or court are hard to see.

## How it works

```
case folder: monitoring notes, indictment, judgment, case.yaml, rulings.yaml (optional)
   │
   ▼
Extraction layer
   in code:          read and normalise the text; split it into sentences and passages with
                     character offsets; citations; "Hearing date:" headers
   local model:      dated events and party arguments, returned as exact quotes (Ollama,
                     structured JSON output, answers cached)
   alignment:        every quote is found in the source text (exact, then normalised, then
                     fuzzy for long quotes); anything not found is dropped and counted
   dates:            parsed in code with dateparser; if the model and the parser disagree,
                     both are kept and the event is marked for review
   │
   ▼
Case record: every item carries its source span (document id, start, end, exact text)
   │                                                            stored in SQLite
   ├─► Absence Detector ────┐
   ├─► Procedural Clock ────┼─► provenance check: each span is re-read from its document,
   ├─► Reuse Detector ──────┘   and a finding whose span does not match is dropped
   └─► Judicial History Tracker (all stored cases, coded rulings)
                                    │
                                    ▼
                   Streamlit app on 127.0.0.1, and the evaluation script
```

The four modules read only the case record, never raw text, so each can be built and tested on its own (a test enforces this). The model never does arithmetic, never grades a person, and never decides a status on its own:

- Intervals are computed in code.
- The model's labels count only when they name the rubric indicator they rely on.
- The judge statistics come from rulings coded by people.

## Setup

You need a Mac with Apple Silicon, Python 3.11, about 8 GB of free disk space, and 16 GB of memory for the live model. The locked setup targets macOS. On Linux it would also install closed-source libraries (NVIDIA's CUDA, and Intel's MKL inside PyTorch on x86-64 computers), several GB more, and it is untested. Do the setup once, while online. After that, Ratio needs no network.

```bash
git clone https://github.com/sahilkharel7/Ratio-Fair-Trial-Hackathon.git
cd Ratio-Fair-Trial-Hackathon

# Python environment with the pinned dependencies (uv: https://docs.astral.sh/uv/)
uv venv -p 3.11
source .venv/bin/activate
uv pip install -r requirements-lock.txt -e ".[dev]"
# without uv: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements-lock.txt -e ".[dev]"

# The embedding model (about 90 MB, saved in models/)
python scripts/fetch_models.py

# The local language model (about 4.7 GB)
brew install ollama
brew services start ollama
ollama pull qwen2.5:7b-instruct

# Stop the Ollama server from contacting ollama.com: this adds "disable_ollama_cloud": true to
# ~/.ollama/server.json, creating the file if needed and keeping any other settings in it
python3 -c 'import json, pathlib; p = pathlib.Path.home() / ".ollama" / "server.json"; s = json.loads(p.read_text().strip() or "{}") if p.exists() else {}; s["disable_ollama_cloud"] = True; p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(s, indent=2) + "\n")'
brew services restart ollama

# Check that everything is ready
python -m ratio preflight
```

To use a different local model, set `RATIO_MODEL` (for example `RATIO_MODEL=mistral:7b-instruct`). The recorded demo answers belong to `qwen2.5:7b-instruct`.

## Run

```bash
streamlit run app/main.py
```

Open http://127.0.0.1:8501. The app only accepts connections from this computer.

On the **Case** page you can:

- **Load the demo case.** This replays the model's recorded answers, takes about 4 seconds, and works with Ollama stopped and Wi-Fi off.
- **Upload your own case folder.** The local model reads it live, which takes a few minutes for a case of this size.
- **Run one note live.** This shows that the recorded answers are real by comparing a fresh answer with the recorded one. It takes about 11 seconds with Ollama running.

The other pages are:

- **Rights coverage**
- **Timeline**
- **Reasoning reuse**
- **Judge profile**

Any evidence button opens the source viewer.

The **Case overview** leads with the current matter, review summaries, and searchable
source documents. Use **Open document** to read an original document; when a search
term is found in its text, the reader opens that exact passage. Imports and live
model checks are available below the record. The shared interface components,
design tokens, and design references are documented in [docs/INTERFACE.md](docs/INTERFACE.md).

**Live demo:** follow [docs/DEMO.md](docs/DEMO.md), a three-minute script with fallback screenshots.

Command line:

```bash
python -m ratio preflight              # is this computer ready for the offline demo?
python -m ratio ingest [CASE_DIR]      # build a case record (add --live to call the local model)
python -m ratio build-demo-cache       # record the demo's model answers with Ollama
python -m ratio check-ollama           # is the local model pulled and served on 127.0.0.1?
```

## Evaluation

```bash
python -m eval.run_eval          # replays the recorded answers; about 4 seconds, no Ollama needed
python -m eval.run_eval --live   # asks the local model wherever no answer is recorded
```

It prints the results and writes `eval/out/report.json`. It exits with an error if any finding, shown or dropped, lacked an exact source span. The current results on the demo case:

| Measure | Result |
|---|---|
| Events found, against the hand-checked timeline | 8 of 8 |
| Dates correct, among the events found | 8 of 8 |
| Expected outputs found (`data/demo/gold/expected_flags.json`: all 7 statuses, the 2 follow-up questions with their language context, and the 7 planted findings) | 16 of 16 |
| Must-not-flag violations, out of 7 planted traps (an appearance before a prosecutor, a quoted statute, the recited charge, the prosecution's argument, the caption, the court's own reasoning on the same facts, and a defence argument the court did answer) | 0 |
| Findings that point to an exact source span | 11 of 11, none dropped |
| Judge pattern indicators that point to exact source spans | 1 of 1 |

These results are **in-sample**: the team wrote both the case and the expected outputs. They show that the pipeline works end to end, not how well it generalises. The main metric we planned, recall against published TrialWatch reports, has not been measured yet.

Tests:

```bash
pytest                                     # every test except the 2 that call the local model; needs models/, not Ollama
pytest -m live                             # only those 2: the local model reads a note again (start Ollama first)
pytest --cov --cov-report=term-missing     # coverage of the ratio package
```

## Responsible AI safeguards

The code and its comments refer to these as hard rules 1 to 5.

| Rule | How Ratio enforces it | How it is tested |
|---|---|---|
| **1. Runs offline, on this computer.** No cloud services, no third-party APIs, no telemetry. | The model client only talks to Ollama on 127.0.0.1 and refuses cloud-hosted models. A network guard blocks every other connection in the app, the command line, the evaluation and the tests. Hugging Face runs in offline mode, and the embedding model loads from `models/`. Streamlit runs with usage statistics off, on 127.0.0.1 only, with no error-page links and no web fonts or remote themes. The app refuses to start if any of these settings is not in effect. | The full pipeline runs under the network guard. A global Streamlit config that tries to override the settings is ignored. The browser checks found every request going to 127.0.0.1. `python -m ratio preflight` checks these settings, any `STREAMLIT_*` environment variables that would override them, and whether the running Ollama server (or its settings file) has its cloud features disabled. |
| **2. Provenance on everything.** | The model returns exact quotes, and Ratio locates each one in the source text; quotes it can't find are dropped. A finding must carry at least one source span (document id, start, end, exact text). Before anything is stored or shown, every span is re-read from its document, and a finding whose span does not match is dropped. | The evaluation reports provenance before and after this check and fails on any unsourced finding. App tests check that every piece of evidence on every page has a button that opens its source. |
| **3. No verdicts about people.** | The messages of findings, and the short status and evidence labels shown next to them, come from fixed templates (`ratio/config/messages.yaml`). Model notes appear only under "Model note (unverified)", after words that characterise a person are removed. Judge indicators say "pattern that warrants review" only when the confidence interval for the difference from the baseline excludes zero, with the significance level divided across the indicators shown. Otherwise they say the judge is not distinguishable from the baseline at this sample size. Every rate shows n and a 95% Wilson interval. Indicators are hidden below 5 cases. The baseline is the same court and charge type only. Rulings are coded by people, never by the model. A name counts for a judge only at the same court, and doubtful names wait for a person to decide. | Known-value tests for the Wilson and Newcombe intervals. App tests check the judge page for blocked terms and for the fixed note, n and intervals. |
| **4. Public or synthetic data only.** | Every synthetic text file starts with a `SYNTHETIC:` line, and every YAML and JSON file declares `synthetic: true`. Public material must cite where it was published. Uploading requires a declaration. A SYNTHETIC banner appears on every page and in the source viewer. | Every demo data file is checked for the marker. App tests check the banner on every page. |
| **5. No invented legal thresholds.** | Only one benchmark is confirmed: General Comment 35's 48 hours to bring a detainee before a judge (para. 33). Every other benchmark has a citation and no threshold, is marked "needs legal review", and is measured but never coloured. Only confirmed benchmarks can be marked as exceeded. | Config tests check that only the 48-hour benchmark is confirmed and that an unconfirmed benchmark is never coloured. |

On every judge page, above everything else: *TrialWatch monitors cases already suspected of unfairness, so these rates are not representative.*

## Data sources

- **The demo case** (`data/demo/case`): *Republic of Calderra v. Daro Venn*. It is entirely synthetic: the country, court, people, minority language and laws are invented. It has four hearing notes, the indictment and the judgment, with planted issues:
  - no interpreter is ever mentioned, though the defendant's first language is a minority language;
  - the defendant first appears before a judge five days after arrest, after an earlier appearance before a prosecutor that does not stop the clock;
  - four judgment passages are copied from the indictment and one is lightly paraphrased;
  - one defence argument is never answered.
- **Judicial history** (`data/demo/history`): 20 synthetic cases generated from `seed.yaml`: 19 from the demo case's court and one from another invented court. On the presiding judge's profile:
  - one indicator fires;
  - two show without a badge;
  - two are hidden as too small;
  - a name written with initials only waits for confirmation;
  - a judge with the same name at another court stays separate.
- **Standards:** the ICCPR, and Human Rights Committee General Comments [No. 32](https://documents.un.org/doc/undoc/gen/g07/437/71/pdf/g0743771.pdf) (Article 14) and [No. 35](https://documents.un.org/doc/undoc/gen/g14/244/51/pdf/g1424451.pdf) (Article 9). Paragraph numbers were checked against the official UN documents. The rubric, benchmarks and ruling codes are in `ratio/config/`.
- **Your own cases:** use public or synthetic material only. A case folder has a `case.yaml` listing its documents (.txt, .md or .pdf), with the type, court and charge type, and whether the material is synthetic or public (with the source). It can also have a `rulings.yaml` of coded rulings for the Judicial History Tracker. `data/demo/case` is a complete example.

No real monitoring notes and no published TrialWatch reports are in this repository.

## Limitations

- **The evaluation is in-sample, on one synthetic case.** Ratio has not been tested on published TrialWatch reports yet.
- **Legal review is pending.** The rubric indicators, the three unconfirmed benchmarks, the reuse standards and the ruling codes were drafted by the engineering team and need review by a lawyer. Until then they are marked "needs legal review", and only the 48-hour benchmark is applied.
- **English only.** Sentence splitting, date parsing and the patterns that recognise statutes, charges and party positions are tuned for English. Multilingual input was cut from the hackathon scope.
- **A small local model makes mistakes.**
  - Every model output is checked against the source text, and dates are parsed in code, but an event the model misses is simply absent.
  - Reading a whole case live takes minutes.
- **Reuse counting is conservative.** A judgment sentence that quotes the law and then applies it is excluded as a statute quotation, so any copying in it is not counted.
- **The Judicial History Tracker runs on synthetic history only.**
  - Coded rulings must be written by hand.
  - TrialWatch's case selection makes the rates unrepresentative, as every judge page says.
  - A judge whose surname is also a title word (for example "Justice") waits for a person to confirm the name.
- **The recorded demo answers match exact prompts.** In the demo, one similarity comparison that selects notes for the model is decided by a margin of 0.00009. On a computer whose arithmetic differs slightly (another operating system or processor), that selection could change, and with it one prompt; the replay would then stop with "no cached labels reply for this request". `python -m ratio preflight` detects this and prints the command that records the missing answer with Ollama running.
- **Platforms:** tested on macOS with Apple Silicon. PyTorch no longer publishes wheels for Intel Macs.
- **Not built yet** (stretch goals):
  - the detention-renewal mode, which compares successive extension orders;
  - a report draft export;
  - the steelman-the-state agent;
  - the jurisprudence linker;
  - a feedback loop from lawyers' corrections.

## Project layout

```
ratio/
  extraction/     reading, sentence splitting, chunking, model extraction, span alignment, dates
  modules/        absence.py, clock.py, reuse.py, judges.py (with the exclusion rules, name registry and statistics)
  config/         rubric.yaml, benchmarks.yaml, ruling_codes.yaml, standards.yaml, messages.yaml, settings.yaml
  schema.py       the case record (frozen pydantic models)
  results.py      module outputs
  provenance.py   every finding re-checked against its source text
  llm.py          local Ollama client with a replayable answer cache
  netguard.py     blocks every connection that does not stay on this computer
  store.py        SQLite case store
  preflight.py    pre-demo checks
app/              Streamlit app: main.py, views/ (one file per page), ratio_ui/ (source viewer, widgets, privacy check)
data/demo/        synthetic demo case, expected outputs, judicial history, recorded model answers
eval/             evaluation script
scripts/          one-time model download and the history generator
tests/
docs/             demo script and screenshots
```

## Team

Komal Neupane, Sahil Kharel, Hardik Kafle, Emeric Chang.

## License

Ratio's own code, synthetic data and documentation are licensed under the [Apache License 2.0](LICENSE). Copyright 2026 Komal Neupane, Sahil Kharel, Hardik Kafle and Emeric Chang.

`ratio/config/` quotes the ICCPR and passages from the UN Human Rights Committee's General Comments No. 32 and No. 35, with their citations. Those are United Nations texts, not the team's work, so the Apache license does not cover them.

### Third-party software

This repository contains no third-party code. The setup steps download Ratio's dependencies from PyPI, the embedding model from Hugging Face and the language model from Ollama's library, each under its own license.

- **unidecode is licensed under the GNU GPL (version 2 or later).** Ratio uses it to match names and words written with accents or with look-alike letters from other alphabets, and almost every part of Ratio loads it.
  - Installing, using and changing Ratio, and copying it within your own organisation, create no obligations. The simplest way to share Ratio is to send the link to this repository.
  - If you give people Ratio *together with* its installed dependencies (a container, a disk image, a packaged app or a prepared laptop), the Free Software Foundation's view is that they form one program. That program must be passed on under the GNU GPL version 3, with its complete source code. Ratio's Apache license allows this.
  - The Foundation counts copies given to contractors, such as external monitors, as passing Ratio on.
  - Separate programs on the same computer, such as Ollama and its model, are not affected.
- **Other copyleft parts.**
  - certifi, tqdm, hf-xet, setuptools, Pillow and SciPy contain code under the Mozilla Public License, the GNU Lesser GPL, or the GPL with the GCC Runtime Library Exception. On some platforms NumPy, scikit-learn and PyTorch do too.
  - Streamlit's built-in fonts are under the SIL Open Font License.
  - Pass these licenses, and the source code they require, on with any bundle.
  - Everything else uses permissive licenses such as MIT, BSD and Apache 2.0.
- **Closed-source parts outside macOS.** On macOS, which the setup targets, everything Ratio installs is open source. Elsewhere:
  - On Linux, PyTorch's standard build installs NVIDIA's closed-source CUDA libraries: most of the `nvidia-*` packages (brought in by the `cuda-toolkit` metapackage), NVIDIA tools inside `triton`, and, on ARM computers, NVIDIA math libraries inside PyTorch itself.
  - On x86-64 Linux and Windows, PyTorch contains Intel's closed-source MKL library, even in its CPU-only build.
  - Ratio does not need a GPU, so PyTorch's CPU-only build leaves out NVIDIA's libraries. On ARM Linux that build is entirely open source.
  - A GPL bundle for x86-64 Linux or Windows would need a PyTorch built without MKL.
- **Models.** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (`qwen2.5:7b-instruct` in Ollama) are licensed under Apache 2.0, and Ollama under MIT. If you set `RATIO_MODEL` to another model, check its license: some models, including other sizes of Qwen2.5, are not open source.

This is a summary, not legal advice.
