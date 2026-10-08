"""The Streamlit app, driven headless with AppTest on the replayed demo case: every page renders
with the SYNTHETIC banner, every piece of evidence on screen has a button that opens its exact
source, the viewer refuses a span that is not in its document, and the privacy check holds."""

import pytest
from streamlit import config as st_config
from streamlit.testing.v1 import AppTest

from ratio.paths import MINILM_DIR, REPO_ROOT
from ratio.store import CaseStore

APP = REPO_ROOT / "app" / "main.py"
PAGES = ("views/coverage.py", "views/timeline.py", "views/reuse.py")
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
