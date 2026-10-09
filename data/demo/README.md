<!-- SYNTHETIC: fictional case; all people, courts, country, language and laws are invented. -->
# Demo data

**SYNTHETIC: fictional case; all people, courts, country, language and laws are invented.** Nothing here comes from a real trial.

| Folder | What it holds |
|---|---|
| `case/` | The demo case, *Republic of Calderra v. Daro Venn*: four hearing notes, the indictment, the judgment, `case.yaml` and the coded rulings in `rulings.yaml`. This is the folder the app loads, and the one you can upload to try the upload path. |
| `gold/` | The hand-checked answers the evaluation compares against. Written by hand in `annotations.yaml`; `python -m ratio.gold` generates `expected_flags.json` (what the modules must and must not find), `gold_timeline.json` (one checked date per event) and `mock_record.json` (the record an ideal extractor would produce, which the modules are tested against). |
| `history/` | 20 synthetic cases that give the presiding judge a history for the Judicial History Tracker. See [history/README.md](history/README.md). |
| `cache/` | The local model's recorded answers for the demo case, so the demo runs with Ollama stopped. Each file in `cache/llm/` is one answer, named by a hash of its prompt; `manifest.json` records the model and Ollama version. `python -m ratio build-demo-cache` records them again. |
