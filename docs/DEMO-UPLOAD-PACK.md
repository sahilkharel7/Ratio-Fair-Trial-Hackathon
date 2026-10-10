# Three real cases for a fresh upload demonstration

These are full English **European Court of Human Rights judgments**, reproduced by OHCHR Cambodia. They are court decisions about historical criminal proceedings, including the domestic charges and sentencing history. They are not the original domestic indictment or first-instance judgment. Keep that distinction visible during the demonstration.

The downloader saves them only to `output/demo-upload-pack/`. They are deliberately absent from the installed reference corpus and the SQLite case collection. Upload one live through **Add case documents**; do not run these PDFs through the default corpus builder beforehand.

## 1. Butkevičius v. Lithuania — start here

- Application **48297/99**, judgment **26 March 2002**, 16 pages.
- Charge: attempting to cheat / obtain property by deception; see §§ 9 and 21. Do not label the charge bribery merely because officials used that word publicly.
- Recorded domestic sentence: **five years and six months**, LTL 50,000 fine and confiscation of half his property (§ 21). This is the sentence imposed, not a prosecutor's requested sentence. Release on licence is recorded in § 25.
- Presumption issue: the Chairman of Parliament's public declarations of guilt. The Court distinguished those remarks from the prosecutor's statements and found an Article 6 § 2 breach (§§ 49–54).
- Useful contrast: the already-installed **Daktaras** judgment found no Article 6 § 2 breach in its different procedural context.
- The archive URL incorrectly includes “2003”; the judgment's cover identifies **2002**.

## 2. Nešťák v. Slovakia — judicial prejudgment

- Application **65559/01**, judgment **27 February 2007**, 24 pages.
- Indictment: conspiracy and robbery, with the cited weapon provision (§ 14).
- Recorded domestic sentence: **five and a half years** at first instance (§ 23), reduced to **five years** on appeal (§ 28). The statute's range in § 36 is a different category; it is not the sentence or a requested penalty.
- Presumption issue: a remand decision stated that the accused had committed the offence before conviction. The Court found an Article 6 § 2 violation (§§ 88–91), explaining that the later conviction did not erase the earlier breach.

## 3. Minelli v. Switzerland — harm without imprisonment

- Application **8660/79**, judgment **25 March 1983**, 21 pages.
- Charge: criminal defamation through the press (§ 10).
- Proceedings expired under the limitation rule, without the applicant's conviction. Nevertheless, he was ordered to bear court costs and the private prosecutors' expenses (§§ 12–13). Do not assign him the fine imposed on the other journalist, Fust (§ 10).
- Presumption issue: the reasoning used to allocate those costs implied guilt. The Court found an Article 6 § 2 violation (§§ 37–41 and operative point 1).
- This demonstrates why harm needs to include financial and reputational consequences, as well as imprisonment.

## Upload walkthrough

1. Open the local dashboard, choose **Add case documents**, and select one original PDF. Mark the document **Public** and paste its publication URL from `sources.json`.
2. Use a new case reference, for example `butkevicius-demo`. Confirm the title, defendant, charge and document role **Judgment**. For this PDF, use the court label **European Court of Human Rights · historical review**; the PDF itself is a regional review judgment.
3. Open the resulting SQLite case entry. Inspect the original source, charge and sentencing passages; confirm who made each statement and when. The local screening produces reading leads, which can include quoted allegations or judicial explanations. A lawyer must assess their relevance.
4. Compare the facts against the installed court judgments and the separate UN review materials. Recorded findings are not a probability that a new application will succeed.

The original PDFs, application numbers, page counts, publication URLs and SHA-256 checksums are listed in `sources.json`. `Ratio-real-case-demo-pack.zip` contains all three PDFs and this guide. Gemini is optional; these upload and source-reading steps work locally without a valid API key.

To recreate the separate download pack: `.venv/bin/python scripts/fetch_demo_pack.py`.
