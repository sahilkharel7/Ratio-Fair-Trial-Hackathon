"""The report draft export: every finding of the replayed demo reaches the annex with its exact
source text, the fixed notes are present, no wording characterises a person, and document text is
escaped so it is shown literally."""

import re

import pytest

from ratio import __main__ as cli
from ratio.config import load_config
from ratio.extraction.build import load_case
from ratio.history import load_alias_decisions, load_history
from ratio.messages import contains_blocked_term
from ratio.paths import DEMO_CASE_DIR, MINILM_DIR
from ratio.pipeline import analysis_context, analyze, analyze_judges, demo_llm, ingest
from ratio.provenance import resolver_for, span_is_valid
from ratio.report import _quote, draft_report, md
from ratio.results import CaseAnalysis
from ratio.testing import make_record



def needs_embedder(test):
    """The demo replay embeds passages with the local MiniLM model."""
    skip = pytest.mark.skipif(not (MINILM_DIR / "modules.json").exists(), reason="run scripts/fetch_models.py once")
    return pytest.mark.embed(skip(test))


@pytest.fixture(scope="module")
def demo():
    config = load_config()
    llm = demo_llm(config)
    record, _ = ingest(load_case(DEMO_CASE_DIR), llm, config)
    analysis = analyze(record, analysis_context(config, llm))
    history = [r for r in load_history() if r.case_id != record.case_id]
    judges = analyze_judges([*history, record], {record.case_id: analysis}, load_alias_decisions(), config)
    profile = next(p for p in judges.profiles if record.case_id in p.case_ids)
    report = draft_report(record, analysis, judges, config, history=history)
    return record, analysis, profile, history, report, config


@needs_embedder
def test_every_finding_reaches_the_annex_with_its_exact_source(demo):
    record, analysis, profile, history, report, _config = demo
    flags = analysis.all_flags() + profile.flags
    annex = report.split("## Annex", 1)[1]
    assert len(re.findall(r"^### F\d+ · ", annex, re.MULTILINE)) == len(flags) == 14
    resolve = resolver_for([record, *history])
    for flag in flags:
        for span in flag.spans:
            assert span_is_valid(span, resolve)
            assert _quote(span.text) in annex
    cited = re.findall(r"\*\*\[(F\d+)\]\*\*", report.split("## Annex", 1)[0])
    assert cited == [f"F{n}" for n in range(1, len(flags) + 1)]  # numbered in the order the body cites them


@needs_embedder
def test_the_fixed_notes_and_case_header_are_present(demo):
    *_, report, config = demo
    notes = config.messages.notes
    assert report.startswith(f"# {notes['report_title']}: Republic of Calderra v. Daro Venn\n")
    for key in ("synthetic_banner", "selection_bias", "report_disclaimer"):
        assert md(notes[key]) in report
    assert "Republic of Calderra v. Aren Holt: Monitoring summary, line 6" in report  # history rulings name their case


@needs_embedder
def test_no_wording_outside_quoted_source_characterises_a_person(demo):
    *_, report, config = demo
    for line in report.splitlines():
        if not line.startswith(">"):
            assert not contains_blocked_term(line, config.messages.block_list), line


def test_document_text_is_escaped_so_it_shows_literally():
    hostile = "# Heading\n> quote\n- item\n1. first\n*bold* _it_ `code` [link](http://x) <script>alert(1)</script> a | b &amp;"
    escaped = md(hostile)
    assert "<script>" not in escaped and "\\<script\\>" in escaped
    assert "\\[link\\](http://x)" in escaped and "\\*bold\\*" in escaped and "\\&amp;" in escaped and "a \\| b" in escaped
    assert all(line.startswith("\\") for line in escaped.splitlines())  # no line opens a heading, quote or list
    assert "Republic of Calderra v. Daro Venn" == md("Republic of Calderra v. Daro Venn")  # ordinary prose is untouched


def test_a_case_with_no_findings_still_reports_every_section():
    config = load_config()
    record = make_record([("notes/one.txt", "monitoring_note", "Hearing date: 2 June 2025\nThe hearing opened.")])
    record = record.model_copy(update={"meta": record.meta.model_copy(update={"title": "<b>Title</b> *x*"})})
    report = draft_report(record, CaseAnalysis(case_id=record.case_id), None, config)
    assert report.startswith(f"# {config.messages.notes['report_title']}: \\<b\\>Title\\</b\\> \\*x\\*\n")
    for heading in ("report_coverage", "report_timeline", "report_renewal", "report_reuse", "report_judge", "report_annex"):
        assert config.messages.notes[heading] in report
    assert md(config.messages.notes["report_no_judge"]) in report
    assert "0 findings" in report


@needs_embedder
def test_the_cli_writes_the_demo_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RATIO_DB", str(tmp_path / "ratio.db"))  # the command reads reviewers' decisions from the store
    out = tmp_path / "report.md"
    assert cli.main(["report", "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Report draft: Republic of Calderra v. Daro Venn") and "### F14 · " in text
    assert "written to" in capsys.readouterr().err
    assert "Similar cases to check" not in text  # no corpus installed (tests/conftest.py): the command still succeeds


@needs_embedder
def test_the_renewal_section_tabulates_the_orders_and_cites_both_findings(demo):
    _record, analysis, *_, report, config = demo
    section = report.split(f"## 3. {config.messages.notes['report_renewal']}", 1)[1].split("## 4.", 1)[0]
    assert "| Detention extension order (14 May 2025) | 2025-05-14 | 2025-07-14 | 56 | 99% | 0 of 3 |" in section
    assert "| Detention extension order (19 March 2025) | 2025-03-19 | 2025-05-12 | 28 | 21% | 3 of 4 |" in section
    assert section.count("**[F") == 2 and "Whole days with no order in the record: 1." in section


@needs_embedder
def test_each_finding_in_the_annex_lists_its_jurisprudence_with_the_caveat(demo):
    *_, report, config = demo
    annex = report.split("## Annex", 1)[1]
    notes = config.messages.notes
    assert md(notes["jurisprudence_caveat"]) in annex
    clock = annex.split("Longer than the benchmark", 1)[1].split("### F", 1)[0]
    assert "Terán Jijón v. Ecuador, communication No. 277/1988 (CCPR/C/GC/35, footnote 96, to para. 33)" in clock
    body = report.split("## Annex", 1)[0]
    assert f"{notes['jurisprudence_heading']}: CCPR/C/GC/32, para. 40." in body  # the interpreter question


@needs_embedder
def test_the_annex_gives_the_states_reply_under_each_contested_finding(demo):
    *_, report, config = demo
    notes = config.messages.notes
    annex = report.split("## Annex", 1)[1]
    assert md(notes["steelman_caveat"]) in annex
    renewal = annex.split("Grounds repeated", 1)[1].split("### F", 1)[0]
    assert f"*{notes['steelman_heading']}:*" in renewal
    assert "- **The investigation was still incomplete, which justified more time.**" in renewal
    clock = annex.split("Longer than the benchmark", 1)[1].split("### F", 1)[0]
    assert f"- {notes['steelman_unsupported']}: The delay was exceptional and justified by the circumstances" in clock
    pattern = annex.split("Pattern that warrants review", 1)[1].split("### F", 1)[0]
    assert notes["steelman_heading"] not in pattern  # never a reply for a judge
