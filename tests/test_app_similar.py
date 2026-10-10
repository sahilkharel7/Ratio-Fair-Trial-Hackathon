"""Similar cases in the app, the report draft and the preflight, on the SYNTHETIC fixture corpus
(tests/precedent_fixtures.py) read from the corpus file.

The page links the replayed demo to the precedents that share its fact patterns, best first; both
sides of every shared pattern open their exact source, checked against it; a missing corpus or a
stopped OpenSearch never breaks a page; and the report draft and the preflight mention similar cases
only as a reading aid, never as a finding.
"""

import html
import logging
import re
import threading
import time
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import ratio.__main__ as cli
from ratio import feedback, precedent_opensearch, preflight
from ratio import precedents as precedents_module
from ratio import report as report_module
from ratio.config import PrecedentSettings, load_config
from ratio.feedback import Review
from ratio.messages import contains_blocked_term
from ratio.paths import CORPUS_DIR, MINILM_DIR, REPO_ROOT, corpus_db_path
from ratio.precedent_schema import PrecedentQuote
from ratio.precedent_store import SqlitePrecedentIndex
from ratio.precedents import link, profile
from ratio.report import draft_report, md
from ratio.results import CaseAnalysis
from ratio.schema import Evidence, Flag, SourceSpan
from ratio.store import CaseStore
from ratio.testing import make_record
from tests.precedent_fixtures import BUILT_AT, DEMO_LIKE, LINKED, PRECEDENTS, TAMPERED, FakeEmbedder, build_fixture_corpus

APP = REPO_ROOT / "app" / "main.py"
APP_DIR = APP.parent
PAGE = "views/similar.py"
CASE_PAGE = "views/case.py"
SQLITE = PrecedentSettings(backend="sqlite")
OPENSEARCH = PrecedentSettings(backend="opensearch")
CONFIG = load_config()
SQLITE_CONFIG = CONFIG.model_copy(update={"settings": CONFIG.settings.model_copy(update={"precedents": SQLITE})})
TITLES = {precedent.id: precedent.title for precedent in PRECEDENTS}
SLOW_LINK_SECONDS = 20.0  # a stalled index answers after this long; the Case page must not wait for it
STALL_SECONDS = 6.0  # a slow index or fact profile, longer than the Case page may wait
WAIT_SLACK_SECONDS = 2.0  # the Case page's own work, on top of its longest wait for Similar cases
JOB_DEADLINE_SECONDS = 60.0  # how long a test waits for a released background computation to finish
LONG_FINDING_CHARS = 574  # longer than the card's clip for facts, within the build's bound on a finding
BROKEN = "the index broke"
SHOWN_LINKS = "Similar cases: 3 past cases share at least 2 fact patterns"


def needs_minilm(test):
    """Loading the demo embeds its passages with the local MiniLM model."""
    skip = pytest.mark.skipif(not (MINILM_DIR / "modules.json").exists(), reason="run scripts/fetch_models.py once")
    return pytest.mark.embed(skip(test))


@pytest.fixture(scope="module")
def taxonomy():
    return CONFIG.fact_patterns


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, taxonomy):
    return build_fixture_corpus(tmp_path_factory.mktemp("corpus") / "precedents.db", FakeEmbedder(), taxonomy)


@pytest.fixture(scope="module")
def fixture_links(corpus, taxonomy):
    return link(DEMO_LIKE, SqlitePrecedentIndex(corpus, taxonomy), FakeEmbedder(), SQLITE, taxonomy)


def opensearch_down(patch) -> None:
    patch.setattr(precedent_opensearch, "available", lambda url, timeout=1.0: False)


def plain(markdown: str) -> str:
    """Markdown source as the reader sees it: no bold markers, no backslashes before punctuation."""
    return re.sub(r"\\(.)", r"\1", markdown.replace("**", ""))


def page_text(at: AppTest) -> list[str]:
    elements = [*at.markdown, *at.caption, *at.info, *at.warning, *at.error, *at.title, *at.header, *at.subheader]
    return [plain(element.value) for element in elements]


def html_bodies(at: AppTest) -> list[str]:
    return [element.proto.body for element in at.get("html")]


def source_buttons(at: AppTest):
    return [button for button in at.button if button.label == "View source"]


def page_links(at: AppTest) -> list[str]:
    return [element.proto.label for element in at.get("page_link")]


def similar_lines(at: AppTest) -> list[str]:
    return [label for label in page_links(at) if label.startswith("Similar cases")]


def wait_for_similar_line(at: AppTest) -> list[str]:
    """Runs the Case page again until its Similar cases line shows: the links finish in the background."""
    deadline = time.monotonic() + JOB_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        at.run()
        if similar_lines(at):
            break
        time.sleep(0.05)
    return similar_lines(at)


def timed_run(run) -> float:
    started = time.monotonic()
    run()
    return time.monotonic() - started


def failure_logs(records) -> list[logging.LogRecord]:
    return [r for r in records if r.levelno >= logging.WARNING and r.exc_info and BROKEN in str(r.exc_info[1])]


def make_flag(flag_id: str) -> Flag:
    span = SourceSpan(doc_id="case-1/judgment.txt", start=0, end=4, text="The ")
    return Flag(
        id=flag_id, case_id="case-1", module="reuse", standard_id="s", standard_label="Standard",
        status="verbatim_reuse", message="Ratio's message.", evidence=(Evidence(role="judgment", span=span),),
    )  # fmt: skip


def decision(flag_id: str, made: str) -> Review:
    note = "The record does not support it." if made == "rejected" else ""
    return Review(case_id="case-1", flag_id=flag_id, decision=made, note=note, flag=make_flag(flag_id))


def test_the_rejections_in_force_are_the_latest_decisions_and_shared_by_the_app_and_the_cli(ui):
    reviews = [
        decision("a", "rejected"),
        decision("b", "rejected"), decision("b", "reopened"),
        decision("c", "accepted"),
        decision("d", "accepted"), decision("d", "rejected"),
        decision("e", "rejected"), decision("e", "accepted"),
    ]  # fmt: skip
    rejected = feedback.rejected_flag_ids(reviews)
    assert rejected == frozenset({"a", "d"}) and isinstance(rejected, frozenset)
    assert feedback.rejected_flag_ids([]) == frozenset()
    assert ui.rejected_flag_ids is feedback.rejected_flag_ids and cli.rejected_flag_ids is feedback.rejected_flag_ids


# --- the report draft ------------------------------------------------------------------------------


def test_the_fixture_links_are_the_ones_the_page_shows(fixture_links):
    assert [found.precedent.id for found in fixture_links] == list(LINKED)


def test_the_report_lists_similar_cases_to_check_only_when_given(fixture_links):
    record = make_record([("notes/one.txt", "monitoring_note", "Hearing date: 2 June 2025\nThe hearing opened.")])
    analysis = CaseAnalysis(case_id=record.case_id)
    without = draft_report(record, analysis, None, CONFIG)
    assert CONFIG.messages.notes["report_similar"] not in without
    assert draft_report(record, analysis, None, CONFIG, similar=()) == without

    report = draft_report(record, analysis, None, CONFIG, similar=fixture_links)
    heading = f"## {CONFIG.messages.notes['report_similar']}"
    section = heading + report.split(heading, 1)[1].split("## Annex", 1)[0]
    assert report.replace(section, "") == without  # the section adds to the report and changes nothing else
    assert md(CONFIG.messages.notes["similar_caveat"]) in section
    entries = section.split("\n- ")[1:]
    assert len(entries) == len(LINKED)
    for entry, found in zip(entries, fixture_links, strict=True):
        assert entry.startswith(f"**{md(found.precedent.title)}") and md(found.precedent.body) in entry
        assert all(md(shared.label) in entry for shared in found.shared)
        assert md(found.precedent.url) in entry


def test_the_suite_never_reads_a_real_corpus_unless_a_test_opts_in():
    path = corpus_db_path()  # tests/conftest.py points RATIO_CORPUS_DB at a file that does not exist
    assert path != CORPUS_DIR / "precedents.db" and not path.exists()


@needs_minilm
def test_the_report_command_lists_similar_cases_when_a_corpus_is_installed(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("RATIO_DB", str(tmp_path / "ratio.db"))
    monkeypatch.setenv("RATIO_CORPUS_DB", str(corpus))
    monkeypatch.setattr(cli, "default_config", lambda: SQLITE_CONFIG)  # the corpus file, never OpenSearch
    out = tmp_path / "report.md"
    assert cli.main(["report", "-o", str(out)]) == 0
    heading = f"## {CONFIG.messages.notes['report_similar']}"
    section = out.read_text(encoding="utf-8").split(heading, 1)[1].split("## Annex", 1)[0]
    entries = section.split("\n- ")[1:]
    assert len(entries) == len(LINKED)
    assert all(entry.startswith(f"**{md(TITLES[pid])}") for pid, entry in zip(LINKED, entries, strict=True))


def test_the_report_section_never_states_an_outcome(fixture_links):
    record = make_record([("notes/one.txt", "monitoring_note", "Hearing date: 2 June 2025\nThe hearing opened.")])
    report = draft_report(record, CaseAnalysis(case_id=record.case_id), None, CONFIG, similar=fixture_links)
    section = report.split(f"## {CONFIG.messages.notes['report_similar']}", 1)[1].split("## Annex", 1)[0]
    findings = [shared.precedent_finding.span.text for found in fixture_links for shared in found.shared if shared.precedent_finding]
    assert findings and not any(md(text) in section for text in findings)  # citations and fact patterns only
    assert not any(contains_blocked_term(line, CONFIG.messages.block_list) for line in section.splitlines())


# --- the preflight -----------------------------------------------------------------------------


def test_preflight_reports_the_installed_corpus(corpus, monkeypatch):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(corpus))
    check = preflight.precedent_corpus(CONFIG, SQLITE)
    assert check.status == "ok" and check.name == preflight.SIMILAR
    assert check.detail.startswith(f"{len(PRECEDENTS)} documents, built {BUILT_AT[:10]}") and "Corpus file" in check.detail


def test_preflight_warns_when_opensearch_is_down_and_names_the_fallback(corpus, monkeypatch):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(corpus))
    opensearch_down(monkeypatch)
    check = preflight.precedent_corpus(CONFIG, OPENSEARCH)
    assert check.status == "warn" and "Corpus file (OpenSearch not running)" in check.detail
    assert "docker compose up -d opensearch" in check.fix and "index-opensearch" in check.fix


def test_preflight_reports_opensearch_when_it_serves_the_corpus(corpus, monkeypatch, taxonomy):
    index = SqlitePrecedentIndex(corpus, taxonomy)
    check = preflight.precedent_corpus(CONFIG, OPENSEARCH, opener=lambda settings, taxonomy: (index, "OpenSearch on 127.0.0.1:9200"))
    assert check.status == "ok" and check.detail.endswith("OpenSearch on 127.0.0.1:9200")


def test_preflight_only_warns_when_there_is_no_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(tmp_path / "missing.db"))
    for settings in (SQLITE, OPENSEARCH):
        check = preflight.precedent_corpus(CONFIG, settings)
        assert check.status == "warn" and "missing.db" in check.detail
        assert "python -m corpus_builder" in check.fix


def test_preflight_never_fails_on_something_unexpected(monkeypatch):
    def broken(settings, taxonomy):
        raise RuntimeError("odd corpus")

    check = preflight.similar_cases(CONFIG, opener=broken)
    assert check.status == "warn" and "RuntimeError: odd corpus" in check.detail


# --- the source viewer for a precedent --------------------------------------------------------------


@pytest.fixture
def ui(corpus, monkeypatch):
    """The app's session module, reading the fixture corpus file (never OpenSearch)."""
    monkeypatch.setenv("RATIO_CORPUS_DB", str(corpus))
    monkeypatch.syspath_prepend(str(APP_DIR))
    from ratio_ui import session

    monkeypatch.setattr(session, "precedent_settings", lambda: SQLITE)
    session.clear_precedent_index()
    return session


def show(precedent_id: str, quote: str, forged: str | None = None) -> AppTest:
    """Open the viewer on a quote of a fixture precedent (or on a forged text at the same place)."""

    def script(precedent_id, quote, forged):
        from ratio.precedent_schema import precedent_doc_id
        from ratio.schema import SourceSpan
        from ratio_ui import session, viewer

        index, _ = session.precedent_index()
        [(doc, _facets)] = [(d, f) for d, f in index.candidates([f.id for f in session.config().fact_patterns.facets]) if d.id == precedent_id]
        start = doc.text.index(quote)
        span = SourceSpan(doc_id=precedent_doc_id(doc.id), start=start, end=start + len(quote), text=forged or quote)
        viewer.show_precedent(span, doc, heading="More than 48 hours before a judge")

    return AppTest.from_function(script, args=(precedent_id, quote, forged), default_timeout=60).run()


def test_the_precedent_viewer_shows_the_exact_passage_checked_against_its_text(ui):
    quote = "Ms. Kostova was arrested at her home and was brought before a judge only five days later."
    at = show("wgad-synthetic-c", quote)
    assert not at.exception and not at.error
    captions = [plain(caption.value) for caption in at.caption]
    assert any(TITLES["wgad-synthetic-c"] in c and "characters" in c and "checked against the source: exact match" in c for c in captions)
    assert any(c.startswith("Synthetic test document: ") for c in captions)  # a fixture is never called a real public case
    address = "`https://example.invalid/precedents/wgad-synthetic-c`"
    assert any(caption.value.startswith(f"Source: {address} · ") for caption in at.caption)  # code: plain text, never a link
    assert any(f'<mark class="ratio-span">{quote}</mark>' in body for body in html_bodies(at))


def test_the_precedent_viewer_shows_the_checked_excerpt_and_never_the_whole_document(ui):
    at = show("wgad-synthetic-c", "Ms. Kostova was arrested at her home and was brought before a judge only five days later.")
    assert not at.exception and not at.error
    assert not at.expander  # the public document is quoted, never republished in full
    assert sum('class="ratio-span"' in body for body in html_bodies(at)) == 1


@pytest.mark.parametrize(
    ("kind", "finding_kind", "expected"),
    [
        ("ccpr_views", "violation_found", "Committee found a violation"),
        ("wgad_opinion", "violation_found", "Working Group found the detention arbitrary"),
        ("trialwatch_report", "monitor_assessment", "TrialWatch assessment"),
        ("ccpr_views", "no_violation", "No violation found"),
        ("wgad_opinion", "not_examined", "Not examined"),
        ("trialwatch_report", "violation_found", None),  # a monitor assesses; it does not find a violation
        ("trialwatch_report", "no_violation", None),
        ("ccpr_views", "monitor_assessment", None),
        ("wgad_opinion", "monitor_assessment", None),
    ],
)
def test_a_finding_is_labelled_only_for_the_pairs_ratio_expects(ui, kind, finding_kind, expected):
    from ratio_ui import widgets

    assert widgets.finding_label(SimpleNamespace(kind=kind), SimpleNamespace(finding_kind=finding_kind)) == expected


def test_a_finding_of_an_unexpected_kind_is_not_shown(ui, fixture_links):
    found = next(f for f in fixture_links if f.precedent.kind == "trialwatch_report")
    shared = next(s for s in found.shared if s.precedent_finding is not None).model_copy(update={"finding_kind": "violation_found"})

    def script(doc, shared):
        from ratio_ui import widgets

        widgets.precedent_evidence(doc, shared, key="odd")

    at = AppTest.from_function(script, args=(found.precedent, shared), default_timeout=60).run()
    assert not at.exception
    quotes = [body for body in html_bodies(at) if 'class="ratio-quote"' in body]
    assert len(quotes) == 1 and shared.precedent_finding.span.text not in quotes[0]  # the facts, never an unlabelled outcome
    assert len(source_buttons(at)) == 1


def long_quote(like: PrecedentQuote, text: str) -> PrecedentQuote:
    return PrecedentQuote(span=SourceSpan(doc_id=like.span.doc_id, start=0, end=len(text), text=text), match="exact")


def test_a_long_finding_is_shown_whole_with_its_negation_and_a_long_fact_is_clipped(ui, fixture_links):
    from ratio_ui import widgets

    found, shared = next((f, s) for f in fixture_links for s in f.shared if s.precedent_finding and widgets.finding_label(f.precedent, s))
    filler = "The source describes each hearing, each order and each period in custody in detail. "
    ending = "The Working Group finds that she was not brought before a judge within 48 hours."
    finding = (filler * 8)[: LONG_FINDING_CHARS - len(ending)] + ending
    fact = filler * 6
    assert len(finding) == LONG_FINDING_CHARS and len(fact) > widgets.QUOTE_CHARS
    shared = shared.model_copy(update={"precedent_fact": long_quote(shared.precedent_fact, fact), "precedent_finding": long_quote(shared.precedent_finding, finding)})

    def script(doc, shared):
        from ratio_ui import widgets

        widgets.precedent_evidence(doc, shared, key="long")

    at = AppTest.from_function(script, args=(found.precedent, shared), default_timeout=60).run()
    assert not at.exception
    fact_html, finding_html = [body for body in html_bodies(at) if 'class="ratio-quote"' in body]
    assert html.escape(finding) in finding_html and "not brought before a judge" in finding_html  # never cut before its "not"
    assert html.escape(fact) not in fact_html and "…" in fact_html


def test_the_precedent_viewer_refuses_a_forged_passage(ui):
    at = show("wgad-synthetic-c", "Ms. Kostova is a reporter", forged="Ms. Kostova is a criminal")
    assert any("not found in its source document" in error.value for error in at.error)
    assert not any('class="ratio-span"' in body for body in html_bodies(at))


def test_the_precedent_viewer_refuses_a_document_edited_after_its_quotes_were_checked(ui):
    at = show(TAMPERED, "Her lawyer was not present at the remand hearing.")
    assert any("not found in its source document" in error.value for error in at.error)


# --- the page, on the replayed demo --------------------------------------------------------------


@pytest.fixture(scope="module")
def app(tmp_path_factory, corpus):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RATIO_DB", str(tmp_path_factory.mktemp("similar-app") / "ratio.db"))
        patch.setenv("RATIO_CORPUS_DB", str(corpus))
        patch.syspath_prepend(str(APP_DIR))
        from ratio_ui import session

        patch.setattr(session, "precedent_settings", lambda: SQLITE)
        session.clear_precedent_index()
        at = AppTest.from_file(str(APP), default_timeout=180)
        at.run()
        at.button(key="load_demo").click().run()
        assert not at.exception
        yield at


@pytest.fixture(scope="module")
def demo_links(app, corpus, taxonomy):
    store = CaseStore()
    case_id = store.last_case()
    case_profile = profile(store.load_case(case_id), store.load_analysis(case_id), taxonomy)
    return link(case_profile, SqlitePrecedentIndex(corpus, taxonomy), FakeEmbedder(), SQLITE, taxonomy)


@pytest.fixture
def page(app):
    app.switch_page(PAGE).run()
    assert not app.exception and not app.error
    return app


@needs_minilm
def test_the_page_shows_the_banner_the_caveat_and_where_it_reads_from(page):
    assert any('class="ratio-banner"' in body and "SYNTHETIC DATA" in body for body in html_bodies(page))
    text = page_text(page)
    assert CONFIG.messages.notes["similar_caveat"] in text
    assert any(t == "Precedent index: Corpus file." for t in text)
    assert [h.value for h in page.title] == ["Similar cases"]


@needs_minilm
def test_the_page_lists_the_linked_precedents_best_first(page, demo_links):
    assert [found.precedent.id for found in demo_links] == list(LINKED)
    assert [plain(s.value) for s in page.subheader] == [TITLES[pid] for pid in LINKED]
    text = page_text(page)
    for found in demo_links:
        assert any(t.startswith(f"Shares {found.score.shared} fact patterns with this case") for t in text)
    assert not any(TITLES[pid] in t for pid in ("tw-synthetic-d", TAMPERED, "ccpr-synthetic-f") for t in text)


@needs_minilm
def test_both_sides_of_every_shared_pattern_have_a_source_button(page, demo_links):
    expected = sum(len(s.case_evidence) + 1 + (s.precedent_finding is not None) for found in demo_links for s in found.shared)
    assert len(source_buttons(page)) == expected
    for found in demo_links:
        for shared in found.shared:
            stem = f"similar-{found.precedent.id}-{shared.facet_id}"
            assert page.button(key=f"{stem}-case-0").label == "View source"
            assert page.button(key=f"{stem}-precedent-0").label == "View source"


@needs_minilm
def test_a_precedent_passage_opens_highlighted_in_its_document(page):
    fact = "Her lawyer was not allowed to attend the remand hearing and received the case file only the day before trial."
    page.button(key="similar-tw-synthetic-a-iccpr_14_3_b-precedent-0").click().run()
    assert not page.exception and not page.error
    assert any("checked against the source: exact match" in caption.value for caption in page.caption)
    assert any(f'<mark class="ratio-span">{fact}</mark>' in body for body in html_bodies(page))


@needs_minilm
def test_the_precedent_finding_is_quoted_with_its_label_and_opens_its_source(page):
    finding = "The Working Group considers that the detention of Ms. Kostova is arbitrary."
    quotes = [body for body in html_bodies(page) if 'class="ratio-quote"' in body]
    assert any("Working Group found the detention arbitrary" in body and finding in body for body in quotes)
    page.button(key="similar-wgad-synthetic-c-gc35_48h-precedent-1").click().run()
    assert any(f'<mark class="ratio-span">{finding}</mark>' in body for body in html_bodies(page))


@needs_minilm
def test_this_cases_passage_opens_in_its_own_document(page, demo_links):
    shared = demo_links[0].shared[0]
    page.button(key=f"similar-{demo_links[0].precedent.id}-{shared.facet_id}-case-0").click().run()
    assert not page.exception and not page.error
    assert any(f'<mark class="ratio-span">{shared.case_evidence[0].span.text}</mark>' in body for body in html_bodies(page))


@needs_minilm
def test_the_corpus_and_its_attribution_are_listed(page):
    assert any(e.label == CONFIG.messages.notes["similar_corpus_heading"] for e in page.expander)
    [sources] = [frame.value for frame in page.dataframe if "Terms of use" in frame.value.columns]
    assert len(sources) == 3 and set(sources["Terms of use"]) == {"https://example.invalid/terms"}
    assert any(t.startswith(f"Built on {BUILT_AT[:10]} from 6 documents") and "dropped" in t for t in page_text(page))


@needs_minilm
def test_no_wording_on_the_page_characterises_a_person(page):
    block_list = CONFIG.messages.block_list
    shown = page_text(page) + html_bodies(page)
    assert not [t for t in shown if contains_blocked_term(t, block_list)]


@needs_minilm
def test_the_case_page_links_to_similar_cases(app):
    app.switch_page("views/case.py").run()
    assert not app.exception
    assert SHOWN_LINKS in page_links(app)


@needs_minilm
def test_the_report_draft_on_the_case_page_lists_the_similar_cases(app, monkeypatch):
    passed = []
    real = report_module.draft_report

    def recording(*args, **kwargs):
        passed.append(kwargs.get("similar", ()))
        return real(*args, **kwargs)

    monkeypatch.setattr(report_module, "draft_report", recording)  # the page imports it on each run
    app.switch_page(CASE_PAGE).run()
    assert not app.exception and passed
    assert [found.precedent.id for found in passed[-1]] == list(LINKED)


@needs_minilm
def test_the_links_are_computed_once_and_reused_across_clicks_and_pages(app, monkeypatch):
    from ratio_ui import session

    session.clear_precedent_index()
    calls = []
    real = precedents_module.link
    monkeypatch.setattr(precedents_module, "link", lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs))
    app.switch_page(PAGE).run()
    app.button(key="similar-tw-synthetic-a-iccpr_14_3_b-precedent-0").click().run()
    app.switch_page(CASE_PAGE).run()
    app.switch_page(PAGE).run()
    assert not app.exception and len(calls) == 1


@needs_minilm
def test_the_case_page_never_waits_for_a_stalled_index(app, monkeypatch):
    from ratio_ui import session

    release = threading.Event()
    real = precedents_module.link

    def stalled(*args, **kwargs):
        release.wait(SLOW_LINK_SECONDS)
        return real(*args, **kwargs)

    session.clear_precedent_index()
    monkeypatch.setattr(precedents_module, "link", stalled)
    monkeypatch.setattr(session, "SIMILAR_WAIT_SECONDS", 0.2)
    try:
        app.switch_page(CASE_PAGE).run()
        assert not release.is_set() and not app.exception and not app.error
        assert not similar_lines(app)
        assert app.download_button(key="export_report").label == "Download report draft (.md)"
        release.set()  # the links keep computing and show on a later run
        assert wait_for_similar_line(app) == [SHOWN_LINKS]
    finally:
        release.set()
        session.clear_precedent_index()


@needs_minilm
@pytest.mark.parametrize("slow", ["open_index", "profile"])
def test_the_case_page_waits_at_most_the_bound_for_all_of_the_similar_cases_work(app, monkeypatch, slow):
    """Opening the index and reading the fact patterns run in the background with the links: the Case
    page waits for none of them longer than SIMILAR_WAIT_SECONDS."""
    from ratio_ui import session

    release = threading.Event()
    owner = session if slow == "open_index" else precedents_module
    real = getattr(owner, slow)

    def stalled(*args, **kwargs):
        release.wait(STALL_SECONDS)
        return real(*args, **kwargs)

    session.clear_precedent_index()
    monkeypatch.setattr(owner, slow, stalled)
    try:
        elapsed = timed_run(lambda: app.switch_page(CASE_PAGE).run())
        assert not app.exception and not app.error and not similar_lines(app)
        assert session.SIMILAR_WAIT_SECONDS - 0.1 <= elapsed < session.SIMILAR_WAIT_SECONDS + WAIT_SLACK_SECONDS
        assert app.download_button(key="export_report").label == "Download report draft (.md)"
        release.set()
        assert wait_for_similar_line(app) == [SHOWN_LINKS]
    finally:
        release.set()
        session.clear_precedent_index()


@needs_minilm
def test_once_a_wait_ran_out_the_case_page_returns_at_once_until_the_links_are_ready(app, monkeypatch):
    from ratio_ui import session

    release = threading.Event()
    real = precedents_module.link

    def stalled(*args, **kwargs):
        release.wait(SLOW_LINK_SECONDS)
        return real(*args, **kwargs)

    session.clear_precedent_index()
    monkeypatch.setattr(precedents_module, "link", stalled)
    try:
        first = timed_run(lambda: app.switch_page(CASE_PAGE).run())
        second = timed_run(app.run)
        assert not app.exception and not app.error and not similar_lines(app)
        assert first >= session.SIMILAR_WAIT_SECONDS - 0.1 and second < session.SIMILAR_WAIT_SECONDS / 2
        release.set()
        assert wait_for_similar_line(app) == [SHOWN_LINKS]
    finally:
        release.set()
        session.clear_precedent_index()


@needs_minilm
@pytest.mark.parametrize("broken", ["link", "open_index"])
def test_the_case_page_says_nothing_when_similar_cases_fail(app, monkeypatch, broken):
    from ratio_ui import session

    def failing(*args, **kwargs):
        raise RuntimeError(BROKEN)

    session.clear_precedent_index()
    monkeypatch.setattr(precedents_module if broken == "link" else session, broken, failing)
    try:
        app.switch_page(CASE_PAGE).run()
        assert not app.exception and not app.error
        assert not similar_lines(app)
        assert app.download_button(key="export_report").label == "Download report draft (.md)"
    finally:
        session.clear_precedent_index()


@needs_minilm
def test_a_failed_computation_is_remembered_logged_once_and_retried_after_a_while(app, monkeypatch, caplog):
    from ratio_ui import session

    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        raise RuntimeError(BROKEN)

    session.clear_precedent_index()
    monkeypatch.setattr(precedents_module, "link", failing)
    caplog.set_level(logging.WARNING)
    try:
        app.switch_page(CASE_PAGE).run()
        app.run()
        app.run()
        assert not app.exception and not app.error and not similar_lines(app)
        assert len(calls) == 1 and len(failure_logs(caplog.records)) == 1  # no retry storm, one log line
        monkeypatch.setattr(session, "SIMILAR_RETRY_SECONDS", 0.0)  # once that time is over, a run tries again
        app.run()
        assert len(calls) == 2 and not app.exception
    finally:
        session.clear_precedent_index()


@needs_minilm
@pytest.mark.parametrize("broken", ["link", "open_index"])
def test_the_similar_cases_page_shows_a_message_not_a_traceback_when_it_fails(app, monkeypatch, caplog, broken):
    from ratio_ui import session

    def failing(*args, **kwargs):
        raise RuntimeError(BROKEN)

    session.clear_precedent_index()
    monkeypatch.setattr(precedents_module if broken == "link" else session, broken, failing)
    caplog.set_level(logging.WARNING)
    try:
        app.switch_page(PAGE).run()
        assert not app.exception and not app.subheader
        assert [plain(error.value) for error in app.error] == [CONFIG.messages.notes["similar_failed"]]
        assert failure_logs(caplog.records)
        assert not any(BROKEN in text for text in page_text(app))
    finally:
        session.clear_precedent_index()


@needs_minilm
def test_the_similar_cases_page_waits_a_bounded_time_then_offers_to_check_again(app, monkeypatch):
    from ratio_ui import session

    release = threading.Event()
    real = precedents_module.link

    def stalled(*args, **kwargs):
        release.wait(SLOW_LINK_SECONDS)
        return real(*args, **kwargs)

    session.clear_precedent_index()
    monkeypatch.setattr(precedents_module, "link", stalled)
    monkeypatch.setattr(session, "SIMILAR_PAGE_WAIT_SECONDS", 0.5)
    try:
        app.switch_page(PAGE).run()
        assert not app.exception and not app.error and not app.subheader
        assert CONFIG.messages.notes["similar_computing"] in page_text(app)
        assert app.button(key="similar_check_again").label == CONFIG.messages.label("similar_check_again")
        assert any(e.label == CONFIG.messages.notes["similar_corpus_heading"] for e in app.expander)
        release.set()
        deadline = time.monotonic() + JOB_DEADLINE_SECONDS
        while not app.subheader and time.monotonic() < deadline:
            app.button(key="similar_check_again").click().run()
        assert not app.exception and [plain(s.value) for s in app.subheader] == [TITLES[pid] for pid in LINKED]
    finally:
        release.set()
        session.clear_precedent_index()


@needs_minilm
def test_a_finding_the_reviewer_rejected_makes_no_fact_pattern(app, monkeypatch, taxonomy):
    store = CaseStore()
    case_id = store.last_case()
    record, analysis = store.load_case(case_id), store.load_analysis(case_id)
    facet = next(f for f in profile(record, analysis, taxonomy).facets if f.origin == "finding")
    flag = next(f for f in analysis.all_flags() if f.id == facet.flag_id)
    expected = profile(record, analysis, taxonomy, rejected_flag_ids={flag.id})
    calls = []
    real = precedents_module.profile
    monkeypatch.setattr(precedents_module, "profile", lambda *args, **kwargs: calls.append(kwargs) or real(*args, **kwargs))
    store.add_review(Review(case_id=case_id, flag_id=flag.id, decision="rejected", note="The record does not support it.", flag=flag))
    try:
        app.switch_page(PAGE).run()
        assert not app.exception and not app.error
        assert calls and all(flag.id in kwargs.get("rejected_flag_ids", ()) for kwargs in calls)
        [patterns] = [plain(m.value) for m in app.markdown if m.value.startswith("- ")]
        assert patterns.splitlines() == [f"- {taxonomy.facet(f.facet_id).label} ({CONFIG.messages.label('facet_from_' + f.origin)})" for f in expected.facets]
    finally:
        store.add_review(Review(case_id=case_id, flag_id=flag.id, decision="reopened", flag=flag))


@needs_minilm
def test_with_no_corpus_the_page_says_what_to_run_and_the_case_page_says_nothing(app, tmp_path, monkeypatch):
    monkeypatch.setenv("RATIO_CORPUS_DB", str(tmp_path / "missing.db"))
    app.switch_page(PAGE).run()
    assert not app.exception and not app.error
    assert any("python -m corpus_builder fetch" in plain(info.value) and "build-db" in plain(info.value) for info in app.info)
    assert any("missing.db" in plain(caption.value) for caption in app.caption)
    assert not source_buttons(app) and not app.subheader
    assert any('class="ratio-banner"' in body for body in html_bodies(app))
    app.switch_page("views/case.py").run()
    assert not any(label.startswith("Similar cases") for label in page_links(app))


@needs_minilm
def test_with_opensearch_down_the_page_reads_the_corpus_file_and_says_so(app, monkeypatch):
    from ratio_ui import session

    monkeypatch.setattr(session, "precedent_settings", lambda: OPENSEARCH)
    opensearch_down(monkeypatch)
    app.switch_page(PAGE).run()
    assert not app.exception and not app.error
    assert "Precedent index: Corpus file (OpenSearch not running)." in page_text(app)
    assert [plain(s.value) for s in app.subheader] == [TITLES[pid] for pid in LINKED]


@needs_minilm
def test_when_nothing_shares_enough_fact_patterns_the_page_says_so(app, monkeypatch):
    from ratio_ui import session

    monkeypatch.setattr(session, "precedent_settings", lambda: PrecedentSettings(backend="sqlite", min_shared=9))
    app.switch_page(PAGE).run()
    assert not app.exception and not app.error and not app.subheader
    assert any(t.startswith("No public case in the corpus shares at least 9 of this case's fact patterns") for t in page_text(app))
