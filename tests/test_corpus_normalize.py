"""Corpus builder, normalize step: stable text from the SYNTHETIC fixture PDF and HTML page, the cleaning
rules, the id rules, the English-only check, and normalize_all's provenance fields and skips."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from corpus_builder.fetch import HTML, PDF
from corpus_builder.normalize import (
    MAX_ID_CHARS,
    clean_text,
    find_symbol,
    html_page,
    language_problem,
    make_id,
    normalize_all,
    url_slug,
)
from corpus_builder.sources import Source
from corpus_builder.store import BuildStore, RawItem

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"
FETCHED_AT = "2026-10-09T12:00:00+00:00"
ENGLISH = (
    "SYNTHETIC: an invented decision for tests. The author of the communication was a journalist who was "
    "arrested and held for six days before a judge saw him. The Committee notes that his lawyer was not "
    "present at the hearing, and that the court did not address the arguments of the defence. "
) * 3


def make_source(kind: str = "trialwatch_report", source_id: str = "example") -> Source:
    return Source.model_validate({
        "id": source_id, "kind": kind, "name": "Example (SYNTHETIC)", "body": "Example monitor",
        "attribution": "SYNTHETIC attribution", "terms_url": "https://example.org/terms/", "terms_checked": "2026-10-09",
        "terms_summary": "Invented for tests.", "allowed_prefixes": ["https://example.org/"], "min_interval_s": 3,
    })  # fmt: skip


@pytest.fixture
def store(tmp_path: Path) -> BuildStore:
    return BuildStore(tmp_path / "build.db")


def record(store: BuildStore, tmp_path: Path, url: str, data: bytes, content_type: str, **meta: object) -> RawItem:
    sha = hashlib.sha256(data).hexdigest()
    path = tmp_path / "raw" / f"{sha}.{'pdf' if content_type == PDF else 'html'}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    item = RawItem(url, str(meta.pop("source_id", "example")), sha, content_type, path, FETCHED_AT, **meta)
    store.record_raw(item)
    return item


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def expected(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8").removesuffix("\n")


# --- the fixtures ---------------------------------------------------------------------------------


def test_the_fixtures_are_synthetic():
    assert fixture_bytes("report.html").startswith(b"<!-- SYNTHETIC:")
    for name in ("report_pdf.expected.txt", "report_html.expected.txt"):
        assert expected(name).startswith("SYNTHETIC: ")


def test_the_fixture_pdf_normalises_to_the_expected_text(store, tmp_path):
    url = "https://example.org/wp-content/uploads/2024/03/Orlo-Fairness-Report.pdf"
    record(store, tmp_path, url, fixture_bytes("report.pdf"), PDF, title="Fairness Report (SYNTHETIC)", state="Quillmark", year=2024)
    report = normalize_all(store, [make_source()])
    (doc,) = store.documents()
    assert report.documents == (doc.id,) == ("tw-orlo-fairness-report",)
    assert doc.text == expected("report_pdf.expected.txt")
    assert "­" not in doc.text and "international" in doc.text  # soft hyphen removed
    assert "witnes-\nses" in doc.text  # a line-end hyphen is kept: no dehyphenation
    assert "standards.\n\nThe court extended" in doc.text  # pages joined by one blank line


def test_the_fixture_html_page_normalises_to_the_article_text(store, tmp_path):
    record(store, tmp_path, "https://example.org/reports/rafe-corran/", fixture_bytes("report.html"), HTML)
    normalize_all(store, [make_source()])
    (doc,) = store.documents()
    assert doc.id == "tw-rafe-corran"
    assert doc.text == expected("report_html.expected.txt")
    assert doc.title == "Fairness Report: the Trial of Rafe Corran in the Republic of Quillmark"
    for dropped in ("SCRIPT TEXT", "NAVIGATION", "FORM TEXT", "FOOTER", "sidebar", "hidden-style-text", "Example Monitor"):
        assert dropped not in doc.text


def test_normalising_again_gives_the_same_document(store, tmp_path):
    record(store, tmp_path, "https://example.org/reports/a.pdf", fixture_bytes("report.pdf"), PDF)
    record(store, tmp_path, "https://example.org/reports/rafe-corran/", fixture_bytes("report.html"), HTML)
    normalize_all(store, [make_source()])
    first = store.documents()
    normalize_all(store, [make_source()])
    assert store.documents() == first
    assert all(clean_text(doc.text) == doc.text for doc in first)


# --- provenance fields ----------------------------------------------------------------------------


def test_a_document_carries_its_hashes_source_and_retrieval_time(store, tmp_path):
    data = fixture_bytes("report.pdf")
    record(store, tmp_path, "https://example.org/reports/a.pdf", data, PDF, state="Quillmark", year=2024)
    normalize_all(store, [make_source()])
    (doc,) = store.documents()
    assert doc.raw_sha256 == hashlib.sha256(data).hexdigest()
    assert doc.text_sha256 == hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
    assert (doc.body, doc.attribution, doc.retrieved_at) == ("Example monitor", "SYNTHETIC attribution", FETCHED_AT)
    assert (doc.kind, doc.state, doc.year, doc.language, doc.url) == ("trialwatch_report", "Quillmark", 2024, "en", "https://example.org/reports/a.pdf")


def test_without_a_seed_title_the_first_line_names_a_pdf(store, tmp_path):
    record(store, tmp_path, "https://example.org/reports/a.pdf", fixture_bytes("report.pdf"), PDF)
    normalize_all(store, [make_source()])
    assert store.documents()[0].title.startswith("SYNTHETIC: an invented fairness report")


def test_a_trialwatch_page_reports_its_linked_report_pdfs(store, tmp_path):
    record(store, tmp_path, "https://example.org/reports/rafe-corran/", fixture_bytes("report.html"), HTML)
    report = normalize_all(store, [make_source()])
    assert report.discovered_pdfs == (
        "https://example.org/wp-content/uploads/2024/03/Corran-Annex.PDF",
        "https://example.org/wp-content/uploads/2024/03/Corran-Fairness-Report.pdf",
    )


def test_linked_pdfs_already_fetched_or_from_other_kinds_are_not_reported(store, tmp_path):
    record(store, tmp_path, "https://example.org/reports/rafe-corran/", fixture_bytes("report.html"), HTML)
    record(store, tmp_path, "https://example.org/wp-content/uploads/2024/03/Corran-Annex.PDF", fixture_bytes("report.pdf"), PDF)
    report = normalize_all(store, [make_source()])
    assert report.discovered_pdfs == ("https://example.org/wp-content/uploads/2024/03/Corran-Fairness-Report.pdf",)
    page = html_page(fixture_bytes("report.html").decode("utf-8"))
    assert page.links  # the page has links, but a Views source never reports them
    other = BuildStore(tmp_path / "other.db")
    html = (ENGLISH + " CCPR/C/1/D/2/2020").encode()
    record(other, tmp_path, "https://example.org/views/", b"<html><body><p>" + html + b"</p></body></html>", HTML)
    assert normalize_all(other, [make_source("ccpr_views")]).discovered_pdfs == ()


# --- cleaning --------------------------------------------------------------------------------------


def test_clean_text_removes_soft_hyphens_nul_and_control_characters():
    assert clean_text("inter­national\x00 law\x0c.") == "international law."


def test_clean_text_normalises_line_endings_trailing_spaces_and_blank_lines():
    assert clean_text("  a  \r\nb\t \rc\n\n\n\n \n d \n") == "a\nb\nc\n\n d"


def test_clean_text_keeps_hyphens_and_line_breaks_inside_paragraphs():
    assert clean_text("the accu-\nsed was held") == "the accu-\nsed was held"


def test_clean_text_composes_accents_and_is_idempotent():
    text = clean_text("Altinél \ud800 ok")
    assert text == "Altinél ? ok" and clean_text(text) == text


# --- HTML regions -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("markup", "text"),
    [
        ("<main><p>main</p><article><p>article</p></article></main>", "article"),
        ("<main><p>main</p><div class='post entry-content'><p>entry</p></div></main>", "entry"),
        ("<body><nav>menu</nav><main><p>main</p></main><p>outside</p></body>", "main"),
        ("<body><nav>menu</nav><p>only</p><footer>foot</footer></body>", "only"),
        ("<article><p>first</p></article><article><p>related</p></article>", "first"),
    ],
)
def test_html_prefers_article_then_entry_content_then_main(markup: str, text: str):
    assert html_page(markup).text == text


def test_html_title_falls_back_to_the_title_element():
    page = html_page("<html><head><title> A  page </title></head><body><p>text</p></body></html>")
    assert page.title == "A page" and page.text == "text"
    assert html_page("<p>no title</p>").title is None


def test_unclosed_and_stray_tags_do_not_break_extraction():
    page = html_page("<article><p>one<p>two</span></div><br/>three</article><p>after")
    assert page.text == "one\n\ntwo\nthree"


# --- ids --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "url", "symbol", "expected_id"),
    [
        ("ccpr_views", "https://example.org/v.pdf", "CCPR/C/107/D/1787/2008", "ccpr-1787-2008"),
        ("wgad_opinion", "https://example.org/o.pdf", "A/HRC/WGAD/2023/01", "wgad-2023-1"),
        ("trialwatch_report", "https://example.org/reports/some-trial/", None, "tw-some-trial"),
        (
            "trialwatch_report",
            "https://example.org/files/Fairness%20Report_Thailand-Pan%20Thalufah%20June%202025-2.pdf",
            None,
            "tw-fairness-report-thailand-pan-thalufah-june-2025-2",
        ),
        ("trialwatch_report", "https://example.org/files/Altınel_Report.PDF", None, "tw-altinel-report"),
        ("ccpr_views", "https://example.org/v.pdf", None, None),
        ("wgad_opinion", "https://example.org/o.pdf", "A/HRC/39/45", None),
        ("trialwatch_report", "https://example.org/", None, None),
    ],
)
def test_id_rules(kind: str, url: str, symbol: str | None, expected_id: str | None):
    assert make_id(kind, url, symbol) == expected_id


@pytest.mark.parametrize(
    ("symbol", "expected_id"),
    [
        ("bangladesh-v-shahidul-alam", "tw-bangladesh-v-shahidul-alam"),
        ("Turkey v. Ahmet Tuna Altınel", "tw-turkey-v-ahmet-tuna-altinel"),  # made a slug like a file name
        ("", "tw-shahidul-alam-fairness-report-january-2023"),  # empty: the file name, as before
        ("  ", "tw-shahidul-alam-fairness-report-january-2023"),
    ],
)
def test_a_trialwatch_id_prefers_the_seeds_symbol_over_the_file_name(symbol: str, expected_id: str):
    url = "https://example.org/wp-content/uploads/2023/07/Shahidul-Alam-Fairness-Report_January-2023.pdf"
    assert make_id("trialwatch_report", url, symbol) == expected_id


def test_a_trialwatch_seed_with_a_symbol_is_stored_under_tw_symbol(store, tmp_path):
    url = "https://example.org/wp-content/uploads/2024/03/Orlo-Fairness-Report.pdf"
    record(store, tmp_path, url, fixture_bytes("report.pdf"), PDF, symbol="quillmark-v-orlo-venn")
    report = normalize_all(store, [make_source()])
    (doc,) = store.documents()
    assert report.documents == (doc.id,) == ("tw-quillmark-v-orlo-venn",)
    assert doc.symbol == "quillmark-v-orlo-venn" and doc.url == url


def test_ids_are_at_most_80_characters_and_never_end_with_a_hyphen():
    from_url = make_id("trialwatch_report", "https://example.org/" + "word-" * 30 + ".pdf", None)
    from_symbol = make_id("trialwatch_report", "https://example.org/a.pdf", "word-" * 30)
    for precedent_id in (from_url, from_symbol):
        assert precedent_id is not None and len(precedent_id) <= MAX_ID_CHARS and not precedent_id.endswith("-")
        assert set(precedent_id) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")


def test_url_slug_ignores_the_query_and_trailing_slash():
    assert url_slug("https://example.org/reports/A_Trial/?utm=x") == "a-trial"


def test_a_symbol_is_found_near_the_start_of_the_text():
    text = "Views adopted by the Committee\nCCPR/C/135/D/2850/2016\n" + ENGLISH + " cites CCPR/C/99/D/1/2001"
    assert find_symbol("ccpr_views", text) == "CCPR/C/135/D/2850/2016"
    assert find_symbol("wgad_opinion", "Opinion\nA/HRC/WGAD/2022/45\n") == "A/HRC/WGAD/2022/45"
    assert find_symbol("trialwatch_report", text) is None
    assert find_symbol("ccpr_views", "x" * 4000 + "CCPR/C/1/D/2/2020") is None


def test_a_views_document_gets_its_id_and_symbol_from_the_text(store, tmp_path):
    body = f"<html><body><article><p>CCPR/C/135/D/2850/2016</p><p>{ENGLISH}</p></article></body></html>"
    record(store, tmp_path, "https://example.org/views/", body.encode(), HTML)
    normalize_all(store, [make_source("ccpr_views")])
    (doc,) = store.documents()
    assert (doc.id, doc.symbol, doc.year) == ("ccpr-2850-2016", "CCPR/C/135/D/2850/2016", None)


def test_an_opinion_takes_its_year_from_its_symbol(store, tmp_path):
    record(store, tmp_path, "https://example.org/o/", f"<p>{ENGLISH}</p>".encode(), HTML, symbol="A/HRC/WGAD/2023/7")
    normalize_all(store, [make_source("wgad_opinion")])
    (doc,) = store.documents()
    assert (doc.id, doc.year, doc.title) == ("wgad-2023-7", 2023, "A/HRC/WGAD/2023/7")


# --- English only, and the skips ----------------------------------------------------------------------


def test_language_check():
    assert language_problem(ENGLISH) is None
    assert "too short" in language_problem("SYNTHETIC: short.")
    assert "ASCII" in language_problem("SYNTHETIC " + "Суд рассмотрел дело и вынес решение. " * 30)
    french = "SYNTHETIC: Le Comité note que l'auteur a été arrêté et détenu pendant six jours avant de voir un juge. " * 8
    assert "English" in language_problem(french)


def test_skips_are_reported_and_nothing_is_stored_for_them(store, tmp_path):
    record(store, tmp_path, "https://example.org/short/", b"<p>SYNTHETIC: too short</p>", HTML)
    record(store, tmp_path, "https://example.org/views/", f"<p>{ENGLISH}</p>".encode(), HTML, source_id="views")
    record(store, tmp_path, "https://example.org/unknown/", f"<p>{ENGLISH}</p>".encode(), HTML, source_id="gone")
    report = normalize_all(store, [make_source(), make_source("ccpr_views", "views")])
    reasons = {issue.url: issue.reason for issue in report.skipped}
    assert "too short" in reasons["https://example.org/short/"]
    assert "symbol" in reasons["https://example.org/views/"]
    assert "not in sources.yaml" in reasons["https://example.org/unknown/"]
    assert report.documents == () and store.documents() == []


def test_a_raw_file_changed_after_fetching_is_skipped(store, tmp_path):
    item = record(store, tmp_path, "https://example.org/reports/a.pdf", fixture_bytes("report.pdf"), PDF)
    item.path.write_bytes(b"%PDF-1.4 tampered")
    report = normalize_all(store, [make_source()])
    assert "sha256 differs" in report.skipped[0].reason


def test_a_missing_or_unreadable_raw_file_is_skipped(store, tmp_path):
    item = record(store, tmp_path, "https://example.org/reports/a.pdf", b"%PDF-1.4 not really a pdf", PDF)
    report = normalize_all(store, [make_source()])
    assert "unreadable PDF" in report.skipped[0].reason
    item.path.unlink()
    assert "cannot read raw file" in normalize_all(store, [make_source()]).skipped[0].reason


def test_two_files_with_the_same_id_keep_the_first(store, tmp_path):
    record(store, tmp_path, "https://example.org/a/same-report.pdf", fixture_bytes("report.pdf"), PDF)
    record(store, tmp_path, "https://example.org/b/same-report/", fixture_bytes("report.html"), HTML)
    report = normalize_all(store, [make_source()])
    assert report.documents == ("tw-same-report",)
    assert "same id tw-same-report as https://example.org/a/same-report.pdf" in report.skipped[0].reason
