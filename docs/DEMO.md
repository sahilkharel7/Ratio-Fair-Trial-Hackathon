# Ratio demo script (3 minutes)

The live demo follows the six demo steps in our hackathon proposal (not published): from the case record to the source line behind each finding. Everything on screen is the **synthetic** demo case *Republic of Calderra v. Daro Venn*.

Measured on 8 October 2026 (MacBook with Apple Silicon):

- Loading the demo took 3.5 seconds with Ollama stopped.
- The optional live-note step took 10.6 seconds, and the model gave the same answer as the recorded one.
- The browser made 364 requests, all to 127.0.0.1, and the app's own connections all stayed on 127.0.0.1.

## Before you present (10 minutes)

1. Check the laptop, with the same case store the demo will use:
   ```bash
   source .venv/bin/activate
   RATIO_DB=/tmp/ratio-demo.db python -m ratio preflight
   ```
   It must end with `Ready for the demo`. Fix any `fail` line with the command it prints. It replays the whole demo and checks that it still shows what this script says.
2. If you will show the live-note step, start Ollama and warm the model once. The first answer after a start is slower.
   ```bash
   brew services start ollama
   ```
   Otherwise stop it (`brew services stop ollama`); the demo works without it.
3. Turn Wi-Fi off.
4. Start the app with an empty case store, so that "Load the demo case" is the first thing the audience sees:
   ```bash
   rm -f /tmp/ratio-demo.db
   RATIO_DB=/tmp/ratio-demo.db streamlit run app/main.py
   ```
5. Open http://127.0.0.1:8501 in a full-screen browser window and zoom to 125% for the projector.
6. If you started Ollama, warm the model: click **Load the demo case**, then **Run this note live** once. Restart the app (step 4) to clear the screen.
7. Open `docs/screenshots/` in a second window, in case you need the fallback.

## The demo

| Time | Step | Click | Say |
|---|---|---|---|
| 0:00 | **1. Case record** | **Case** page → **Load the demo case** (about 4 s) | "Four hearing notes, the indictment and the judgment become one case record: 46 note sentences, 13 dated events and 2 party arguments. That gives 11 findings, and each one is linked to the exact text it rests on. The model's answers are replayed from a recording, and this laptop is offline right now." |
| 0:25 | *(optional)* live model | **Run this note live** (about 11 s) | "To show that the recorded answers are real, the local model reads hearing 1 again now, on this laptop. It gives the same answer." |
| 0:40 | **2. Rights coverage** | **Rights coverage** → **Open** on *ICCPR Art. 14(3)(f) Free assistance of an interpreter* | "Violations hide in omissions. No note ever mentions an interpreter. Ratio does not invent a finding. It gives the monitor a follow-up question, and next to it a note it does not count as evidence: the defendant's first language is Ostric." |
| 1:05 | **3. Timeline** | **Timeline**, at the top: the red interval | "Dates are pulled from three documents and the interval is computed in code: 4 to 6 days from arrest to the first appearance before a judge. General Comment 35, paragraph 33, treats 48 hours as ordinarily sufficient. The appearance before the prosecutor the day after the arrest does not stop the clock: a prosecutor is not a judge. This is our only confirmed benchmark; the others are measured, cited and marked 'needs legal review', never coloured." |
| 1:35 | **4. Reasoning reuse** | **Reasoning reuse**: the score and the side-by-side view, then **Whole documents** and scroll the judgment to sections II to IV | "60% of the court's own reasoning can be traced to the indictment: 48% word for word and 12% as a close paraphrase. The same number marks both sides of each match. Legitimate quotation doesn't count: the recited charge, the parties' arguments and the quoted statute are greyed out, each tagged with the reason, and excluded before scoring. One defence argument, about how the phone messages were extracted, gets no response in the judgment." |
| 2:05 | **5. Judge profile** | **Judge profile** | "This note is always at the top: TrialWatch monitors cases already suspected of unfairness, so these rates are not representative. Detention at the first appearance is 9 of 9 cases against 4 of 10 for the other judges of the same court and charge type. It is a 'pattern that warrants review', never a verdict, and n and the intervals are always shown. Two indicators are not distinguishable from the baseline, and two are hidden because they have fewer than 5 cases. A name written only as 'I. Varda' waits for a person to confirm it." |
| 2:35 | **6. Source** | Any **Source** button, for example under the red interval on the **Timeline** | "Every finding opens its source: the document, the line and the exact characters, checked character for character against the text Ratio read. A finding without a source is dropped. This is the moment that earns trust." |
| 3:00 | end | | |

Tip: the Case page's summary links to each page, so you can click through in order from there.

## If something goes wrong

| What you see | What to do |
|---|---|
| The app does not start, or a page is blank | Switch to the screenshots, in the order of the steps above: `01-case.png`, then `07-live-note.png` if you show the live step, then `02-coverage.png` to `06-source.png` (with `04b-reuse-excluded.png` after `04-reuse.png`) |
| "Load the demo case" shows **Stopped** and "no cached labels reply for this request (replay-only mode …)" | This laptop computes one similarity slightly differently (see the README's limitations). Use the screenshots now. After the demo, run `python -m ratio preflight` with Ollama running and use the command it prints |
| The live-note step says "The local model is not available" | Say "the rest of the demo runs without it" and move on. This is by design |
| A page says to load a case first | Go to **Case** and click **Load the demo case** |
| "Ratio will not start because these privacy settings are not in effect" | The app was started with changed settings. Run `git checkout -- .streamlit app/.streamlit` and start it again from the repository root |

## Proving it is offline (for questions)

With the app running:

```bash
lsof -nP -iTCP -a -p "$(lsof -t -iTCP:8501 -sTCP:LISTEN)"
```

Every line shows 127.0.0.1. During the live-note step you also see a connection to 127.0.0.1:11434, the local Ollama server.

## If someone checks a line number in the raw file

The line and character numbers in the source viewer count positions in the text Ratio read, which starts after the file's first line (the SYNTHETIC marker). In the raw file the same sentence is one line lower.

## Fallback screenshots

| File | Shows |
|---|---|
| `screenshots/01-case.png` | The demo case loaded: 11 findings, each linked to its source |
| `screenshots/02-coverage.png` | The coverage grid, with the interpreter follow-up question open |
| `screenshots/03-timeline.png` | The red interval: 4 to 6 days from arrest to the first appearance before a judge, against GC35's 48 hours |
| `screenshots/04-reuse.png` | The reuse score and the side-by-side view |
| `screenshots/04b-reuse-excluded.png` | Whole documents: the recited charge, the parties' arguments and the statute quote greyed out, each tagged with the reason |
| `screenshots/05-judge.png` | The judge profile: the selection-bias note and the pattern that warrants review, with n and intervals |
| `screenshots/06-source.png` | The source viewer: the indictment sentence highlighted, with its line and characters in the text Ratio read |
| `screenshots/07-live-note.png` | The live model's answer, the same as the recorded one |
