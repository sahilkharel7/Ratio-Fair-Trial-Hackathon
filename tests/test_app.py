"""The Streamlit app, driven headless with AppTest on the replayed demo case: every page renders
with the SYNTHETIC banner, every piece of evidence on screen has a button that opens its exact
source, the viewer refuses a span that is not in its document, and the privacy check holds."""

import re

import pytest
from streamlit import config as st_config
from streamlit.testing.v1 import AppTest

from ratio.config import default_config
from ratio.messages import contains_blocked_term
from ratio.paths import MINILM_DIR, REPO_ROOT
from ratio.store import CaseStore

APP = REPO_ROOT / "app" / "main.py"
PAGES = ("views/coverage.py", "views/timeline.py", "views/reuse.py", "views/judges.py")
JUDGE_PAGE = "views/judges.py"
pytestmark = [
    pytest.mark.embed,
    pytest.mark.skipif(not (MINILM_DIR / "modules.json").exists(), reason="run scripts/fetch_models.py once"),
]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RATIO_DB", str(tmp_path_factory.mktemp("app") / "ratio.db"))
        at = AppTest.from_file(str(APP), default_timeout=180)
        at.run()
        at.button(key="load_demo").click().run()
        assert not at.exception
        yield at


@pytest.fixture(scope="module")
def analysis(app):
    store = CaseStore()
    return store.load_analysis(store.last_case())


def html_bodies(at: AppTest) -> list[str]:
    return [element.proto.body for element in at.get("html")]


def source_buttons(at: AppTest):
    return [button for button in at.button if button.label == "Source"]


def test_case_page_summarises_the_demo_without_errors(app):
    assert not app.exception and not app.error
    assert any("11 findings" in m.value for m in app.markdown)
    assert any("SYNTHETIC DATA" in body for body in html_bodies(app))
    links = [link.proto.label for link in app.get("page_link")]
    assert any(label.startswith("Judge profile: Ilena Varda, 1 pattern that warrants review (9 cases, 3 indicators") for label in links)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_the_synthetic_banner(app, page):
    app.switch_page(page).run()
    assert not app.exception and not app.error
    assert any('class="ratio-banner"' in body and "SYNTHETIC DATA" in body for body in html_bodies(app))


def test_every_guarantee_opens_its_evidence_or_follow_up(app, analysis):
    app.switch_page("views/coverage.py").run()
    for assessment in analysis.absence.assessments:
        app.button(key=f"open-{assessment.rubric_id}").click().run()
        flag = next((f for f in analysis.absence.flags if f.id == assessment.flag_id), None)
        expected = len(flag.evidence) if flag else len(assessment.follow_up.context)
        assert len(source_buttons(app)) == expected, assessment.rubric_id
        if flag is None:
            assert any(assessment.follow_up.question.split("?")[0] in info.value for info in app.info)


def test_timeline_has_a_source_button_for_every_span(app, analysis):
    app.switch_page("views/timeline.py").run()
    clock = analysis.clock
    spans = sum(len(i.evidence) for i in clock.intervals) + sum(len(e.mentions) for e in clock.timeline)
    assert len(source_buttons(app)) == spans
    red = [i for i in clock.intervals if i.status == "exceeds_benchmark"]
    assert [i.benchmark_id for i in red] == ["gc35_48h"]


def test_reuse_page_links_both_sides_of_every_pair_and_every_argument(app, analysis):
    app.switch_page("views/reuse.py").run()
    reuse = analysis.reuse
    spans = 2 * len(reuse.pairs) + len(reuse.arguments) + sum(len(c.responding) for c in reuse.arguments)
    assert len(source_buttons(app)) == spans
    def panels() -> str:
        return "".join(body for body in html_bodies(app) if body.startswith('<div class="ratio-doc">'))

    reasoning_view = panels()  # opens at the court's assessment, where the matches are
    assert reasoning_view.startswith('<div class="ratio-doc">V. ASSESSMENT OF THE COURT')
    assert reasoning_view.count('<mark class="ratio-verbatim">') >= 8  # 4 pairs, both sides
    assert reasoning_view.count('<mark class="ratio-paraphrase">') == 2
    app.segmented_control(key="reuse_view").set_value("Whole documents").run()
    whole = panels()
    assert whole.startswith('<div class="ratio-doc"><span class="ratio-excluded">MIREVO DISTRICT COURT')
    assert 'class="ratio-tag">charge recital<' in whole and 'class="ratio-tag">statute quote<' in whole
    assert len(source_buttons(app)) == spans


def test_source_button_opens_the_exact_span_highlighted_in_its_document(app, analysis):
    app.switch_page("views/reuse.py").run()
    source_buttons(app)[0].click().run()
    pair = analysis.reuse.pairs[0]
    assert not app.exception
    assert any("exact match" in caption.value for caption in app.caption)
    assert any(f'<mark class="ratio-span">{pair.judgment.text}</mark>' in body for body in html_bodies(app))


def test_viewer_refuses_a_span_that_is_not_in_its_document(app):
    def script():
        from ratio.schema import SourceSpan
        from ratio_ui import session, viewer

        record = session.current_case().record
        document = record.documents_of_type("judgment")[0]
        viewer.show_source(SourceSpan(doc_id=document.id, start=0, end=5, text="FORGE"), record)

    at = AppTest.from_function(script, default_timeout=60).run()
    assert any("not found in its source document" in error.value for error in at.error)
    assert not any('class="ratio-span"' in body for body in html_bodies(at))


def test_privacy_check_rejects_unsafe_settings():
    from ratio_ui import privacy

    assert privacy.problems() == []
    original = st_config.get_option("server.address")
    try:
        st_config.set_option("server.address", "0.0.0.0")
        assert any("server.address" in problem for problem in privacy.problems())
    finally:
        st_config.set_option("server.address", original)


def test_the_app_uses_no_icon_fetched_from_the_internet_and_no_raw_html():
    sources = [path.read_text(encoding="utf-8") for path in (REPO_ROOT / "app").rglob("*.py")]
    assert sources and not any(":material/" in text for text in sources)
    assert not any("unsafe_allow_html=True" in text or "unsafe_allow_javascript=True" in text for text in sources)


def test_every_page_in_the_router_exists():
    main = APP.read_text(encoding="utf-8")
    for page in ("views/case.py", *PAGES):
        assert f'"{page}"' in main and (APP.parent / page).is_file()


# --- judge profile -----------------------------------------------------------------------------


def plain(markdown: str) -> str:
    """Markdown source as the reader sees it: no bold markers, and no backslashes before punctuation (md_escape)."""
    return re.sub(r"\\(.)", r"\1", markdown.replace("**", ""))


def page_text(at: AppTest) -> list[str]:
    elements = [*at.markdown, *at.caption, *at.warning, *at.info, *at.title, *at.subheader]
    return [plain(element.value) for element in elements]


@pytest.fixture(scope="module")
def judges_page(app):
    app.switch_page(JUDGE_PAGE).run()
    return app


@pytest.fixture(scope="module")
def judge_report(app):
    from ratio_ui import session

    return session.judge_report()


def test_judge_page_opens_on_the_demo_judge_with_the_fixed_note_first(judges_page):
    at = judges_page
    assert not at.exception and not at.error
    assert at.selectbox(key="ratio_judge_id").format_func(at.selectbox(key="ratio_judge_id").value).startswith("Ilena Varda · Mirevo")
    assert "not representative" in at.warning[0].value  # the selection-bias note, above everything else
    assert any("SYNTHETIC DATA" in body for body in html_bodies(at))


def test_judge_page_shows_one_pattern_with_its_sample_size_and_hides_small_counts(judges_page):
    text = page_text(judges_page)
    patterns = [t for t in text if t.startswith("Pattern that warrants review: ")]
    assert len(patterns) == 1 and "n=9 cases" in patterns[0] and "n=10 cases" in patterns[0]
    assert "indicators compared on this page: 3" in patterns[0]
    assert sum(t.startswith("Not distinguishable from the baseline at this sample size") for t in text) == 2
    assert {t for t in text if t.startswith("Not shown")} == {"Not shown: fewer than 5 cases (3).", "Not shown: fewer than 5 cases (4)."}
    assert any(t.startswith("This judge: 9 of 9 cases, 100% (95% interval") for t in text)


def test_judge_page_wording_never_characterises_the_judge(judges_page):
    block_list = default_config().messages.block_list
    assert not [t for t in page_text(judges_page) if contains_blocked_term(t, block_list)]


def test_every_counted_ruling_on_the_judge_page_has_a_source_button(judges_page, judge_report):
    profile = next(p for p in judge_report.profiles if p.display_name == "Ilena Varda" and p.court.startswith("Mirevo"))
    rulings = sum(len(i.outcomes) for i in profile.indicators) + sum(len(d.evidence) for d in profile.descriptive)
    assert rulings == 35 and len(source_buttons(judges_page)) == rulings
    assert {e.span.case_id for i in profile.indicators for o in i.outcomes for e in [o.ruling]} == set(profile.case_ids)


def test_a_ruling_from_another_case_opens_highlighted_in_its_own_document(judges_page, judge_report):
    at = judges_page
    source_buttons(at)[0].click().run()
    assert not at.exception
    assert any("exact match" in caption.value for caption in at.caption)
    quote = "the court ordered that Aren Holt be detained pending trial"
    assert any(f'<mark class="ratio-span">{quote}</mark>' in body for body in html_bodies(at))
    assert any("SYNTHETIC DATA" in body for body in html_bodies(at))


def test_the_same_name_at_another_court_is_a_separate_profile_with_nothing_shown(judges_page, judge_report):
    at = judges_page
    other = next(p for p in judge_report.profiles if p.court == "Port Elsin Regional Court")
    at.selectbox(key="ratio_judge_id").set_value(other.judge_id).run()
    assert not at.exception and not at.error
    text = page_text(at)
    assert not any(t.startswith("Pattern that warrants review") for t in text)
    assert not source_buttons(at)
    assert "not representative" in at.warning[0].value


def test_progress_labels_from_an_uploaded_case_are_escaped():
    def script():
        from ratio_ui import widgets

        widgets.progress_bar()("![beacon](http://attacker.example/b.png)", 1, 2)

    at = AppTest.from_function(script, default_timeout=60).run()
    [bar] = at.get("progress")
    assert "](http" not in bar.proto.text and "\\!\\[beacon\\]" in bar.proto.text


@pytest.fixture
def mixed_store(tmp_path, monkeypatch):
    """The synthetic demo history and two public cases in one store, with no case open."""
    from ratio.history import load_history
    from ratio.testing import CodedRuling, make_history_case

    monkeypatch.setenv("RATIO_DB", str(tmp_path / "ratio.db"))
    store = CaseStore()
    for record in load_history():
        store.save_case(record)
    court = "Northgate Public Court"
    store.save_case(make_history_case("pub-01", [CodedRuling("detention_ordered", "Maren Tull", "2024-01-10")], court=court, public=True))
    store.save_case(make_history_case("pub-02", [CodedRuling("detention_refused", "M. Tull", "2024-02-10")], court=court, public=True))
    return store


def visible_text(at: AppTest) -> str:
    options = list(at.selectbox(key="ratio_judge_id").options)  # the labels as shown
    tables = [frame.value.to_csv() for frame in at.dataframe]
    return "\n".join(page_text(at) + options + tables + html_bodies(at))


def test_synthetic_and_public_judges_are_never_on_one_view(mixed_store):
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.switch_page(JUDGE_PAGE).run()
    assert not at.exception and not at.error
    assert at.segmented_control(key="ratio_judge_data").value is True  # the first judge by name is synthetic
    synthetic_view = visible_text(at)
    assert "SYNTHETIC DATA" in synthetic_view and "hist-initials-01" in synthetic_view
    assert "Tull" not in synthetic_view and "Northgate" not in synthetic_view and "pub-0" not in synthetic_view
    at.segmented_control(key="ratio_judge_data").set_value(False).run()
    assert not at.exception and not at.error
    public_view = visible_text(at)
    assert "SYNTHETIC DATA" not in public_view
    assert "Maren Tull" in public_view and "pub-02" in public_view  # its pending name, on its own view
    for synthetic_word in ("Varda", "Mirevo", "Hollin", "Selik", "hist-"):
        assert synthetic_word not in public_view, synthetic_word


def test_the_judge_page_opens_on_the_judge_of_the_case_being_viewed_after_a_new_kind_of_data_arrives(tmp_path, monkeypatch):
    from ratio.history import load_history
    from ratio.testing import CodedRuling, make_history_case

    monkeypatch.setenv("RATIO_DB", str(tmp_path / "ratio.db"))
    store = CaseStore()
    for record in load_history():
        store.save_case(record)
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.switch_page(JUDGE_PAGE).run()  # only synthetic data so far: no choice of data shown
    assert not at.get("segmented_control") and "SYNTHETIC DATA" in "".join(html_bodies(at))
    public = make_history_case("pub-01", [CodedRuling("detention_ordered", "Maren Tull", "2024-01-10")], court="Northgate Public Court", public=True)
    store.save_case(public)
    store.set_last_case("pub-01")  # a public case is now the one being viewed
    at.switch_page("views/case.py").run()
    at.switch_page(JUDGE_PAGE).run()
    assert not at.exception and not at.error
    assert at.segmented_control(key="ratio_judge_data").value is False
    assert at.selectbox(key="ratio_judge_id").options == ["Maren Tull · Northgate Public Court"]
    assert "SYNTHETIC DATA" not in "".join(html_bodies(at))


def test_names_waiting_for_confirmation_are_shown_even_when_no_judge_of_their_kind_is_registered(tmp_path, monkeypatch):
    from ratio.testing import CodedRuling, make_history_case

    monkeypatch.setenv("RATIO_DB", str(tmp_path / "ratio.db"))
    store = CaseStore()
    title_only = [CodedRuling("detention_ordered", "The Presiding Judge", "2024-01-10")]
    store.save_case(make_history_case("pub-t1", title_only, court="Northgate Public Court", public=True))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.switch_page(JUDGE_PAGE).run()
    assert not at.exception and not at.error
    assert not any("No coded rulings" in info.value for info in at.info)
    shown = visible_text(at) if at.get("selectbox") else "\n".join(page_text(at) + [f.value.to_csv() for f in at.dataframe])
    assert "pub-t1" in shown and "The Presiding Judge" in shown and "data/alias_decisions.yaml" in shown
    assert "SYNTHETIC DATA" not in "".join(html_bodies(at))
