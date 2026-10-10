import pytest
from streamlit.testing.v1 import AppTest
from ratio.library import LibraryStore
from ratio.paths import REPO_ROOT, MINILM_DIR
from ratio.store import CaseStore

pytestmark = pytest.mark.skipif(
    not (MINILM_DIR / "modules.json").exists(),
    reason="local MiniLM required for recorded demo loading",
)


def test_homepage_is_a_stored_case_collection_and_case_opens_the_focused_view(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RATIO_DB", str(tmp_path / "cases.db"))
    app = AppTest.from_file(
        str(REPO_ROOT / "app" / "main.py"), default_timeout=180
    ).run()
    assert app.title[0].value == "Case collection"
    app.button(key="load_demo").click().run()
    assert not app.exception and len(LibraryStore(CaseStore()).list()) == 3
    app.button(key="open-sorin-2025").click().run()
    assert not app.exception and app.title[0].value == "Nadia Sorin"
    text = "\n".join(x.value for x in app.markdown)
    assert "3 years imprisonment" in text and "2 years imprisonment" in text
    assert "Up to 5 years imprisonment" in text
    assert any("Presumption of innocence" in h.value for h in app.header)
    assert CaseStore().last_case() == "sorin-2025"
    assert any("Source" in b.label or "View source" == b.label for b in app.button)
