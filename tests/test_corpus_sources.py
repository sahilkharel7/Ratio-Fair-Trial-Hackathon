"""Corpus builder, the repository's sources.yaml: the verified seeds of each public source, each on its
source's allowlist. Reads the file only: these tests never touch the network."""

from __future__ import annotations

import re

import pytest

from corpus_builder.sources import Source, load_sources

SEED_COUNTS = {"trialwatch_reports": 20, "hri_fairness_reports": 3, "wgad_opinions": 11, "ccpr_views": 7}
CCPR_FULLTEXT = "https://ccprcentre.org/files/decisions/"


@pytest.fixture(scope="module")
def sources() -> dict[str, Source]:
    return {source.id: source for source in load_sources()}


def test_each_source_has_its_verified_seeds(sources):
    assert {source_id: len(source.seeds) for source_id, source in sources.items()} == SEED_COUNTS
    assert all(source.terms_checked == "2026-10-09" for source in sources.values())


def test_every_url_is_on_its_sources_allowlist(sources):
    for source in sources.values():
        urls = [seed.url for seed in source.seeds] + ([source.sitemap] if source.sitemap else [])
        assert [url for url in urls if not source.allows(url)] == [], source.id


def test_no_document_is_listed_twice_across_sources(sources):
    urls = [seed.url for source in sources.values() for seed in source.seeds]
    assert len(urls) == len(set(urls))


def test_trialwatch_reports_are_curated_pdf_seeds_without_a_sitemap(sources):
    trialwatch = sources["trialwatch_reports"]
    assert trialwatch.sitemap is None and trialwatch.url_pattern is None
    assert not trialwatch.wants("https://cfj.org/reports/bangladesh-v-shahidul-alam/")  # no HTML duplicate of a PDF
    assert trialwatch.min_interval_s == 10
    for seed in trialwatch.seeds:
        assert seed.url.startswith("https://cfj.org/wp-content/uploads/") and seed.url.endswith(".pdf")
        assert re.fullmatch(r"[a-z0-9-]+", seed.symbol) and seed.title and seed.state and seed.year


def test_every_seed_carries_its_title_state_and_year(sources):
    for source in sources.values():
        assert all(seed.title and seed.state and seed.year for seed in source.seeds), source.id


def test_wgad_opinions_carry_their_un_symbol(sources):
    wgad = sources["wgad_opinions"]
    assert wgad.terms_url == "https://www.ohchr.org/en/copyright"
    for seed in wgad.seeds:
        match = re.fullmatch(r"A/HRC/WGAD/(\d{4})/\d+", seed.symbol)
        assert match and int(match.group(1)) == seed.year
        assert seed.url.startswith("https://www.ohchr.org/sites/default/files/") and seed.url.endswith(".pdf")


def test_ccpr_views_are_only_the_full_text_un_pdfs_never_the_centres_own_digest_pages(sources):
    ccpr = sources["ccpr_views"]
    assert ccpr.allows("https://cdn.ccprcentre.org/files/decisions/G1813926.pdf")  # where the PDFs redirect
    by_title = {seed.title: seed.url for seed in ccpr.seeds}
    assert by_title["Sannikov v. Belarus"] == "https://ccprcentre.org/files/decisions/G1813304.pdf"
    for title in ("Lydia Cacho Ribeiro v. Mexico", "Esergepov v. Kazakhstan", "Kovsh (Abramova) v. Belarus"):
        assert title not in by_title  # only a CCPR Centre decision page (its own digest, all rights reserved)
    for seed in ccpr.seeds:
        assert seed.url.startswith(CCPR_FULLTEXT) and seed.url.endswith(".pdf"), seed.url
        assert "/decision/" not in seed.url, seed.url
        assert re.fullmatch(r"CCPR/C/\d+/D/\d+/\d{4}", seed.symbol)
