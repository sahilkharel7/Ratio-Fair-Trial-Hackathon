"""A tiny SYNTHETIC precedent corpus, written straight to a precedents.db through the real DDL and meta,
for the Similar cases tests. Every title, name, state and body below is invented.

The precedents are chosen against the demo case (Calderra: a journalist charged with false information,
five days before a judge, no lawyer at remand, witnesses refused, reused reasoning, repeated renewals):
- tw-synthetic-a, wgad-synthetic-c and ccpr-synthetic-b share at least two fact patterns, one about
  procedure, so they link (in that order: rarer shared facets weigh more);
- tw-synthetic-d shares only who was prosecuted and the charge, so it must NOT link;
- tw-synthetic-e would rank first, but its text was edited after its quotes were verified, so every
  quote of it fails the provenance check and it is dropped;
- ccpr-synthetic-f shares a single fact pattern, below min_shared.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ratio.config import FactPatternTaxonomy
from ratio.embeddings import EMBEDDING_DIM
from ratio.precedent_schema import EMBED_MODEL_TAG, SCHEMA_VERSION, SQLITE_DDL, CaseFacet, CaseProfile
from ratio.schema import Evidence, SourceSpan
from ratio.testing import FakeEmbedder as _HashedEmbedder

SYNTHETIC_LINE = "SYNTHETIC: test precedent; every name, place and body in it is invented."
BUILT_AT = "2026-10-09T00:00:00+00:00"
LINKED = ("tw-synthetic-a", "wgad-synthetic-c", "ccpr-synthetic-b")  # the demo case's links, best first
NOT_LINKED = ("tw-synthetic-d", "tw-synthetic-e", "ccpr-synthetic-f")
TAMPERED = "tw-synthetic-e"


class FakeEmbedder(_HashedEmbedder):
    """Deterministic, L2-normalised bag-of-words vectors with the corpus dimension (384)."""

    def __init__(self) -> None:
        super().__init__(dim=EMBEDDING_DIM)


@dataclass(frozen=True)
class FixtureFacet:
    facet_id: str
    facts: tuple[str, ...]
    finding_kind: str
    finding: str | None = None


@dataclass(frozen=True)
class FixturePrecedent:
    id: str
    kind: str
    title: str
    body: str
    state: str
    year: int
    paragraphs: tuple[str, ...]
    facets: tuple[FixtureFacet, ...]
    tamper: tuple[str, str] | None = None  # (old, new) of the same length, applied after verification

    @property
    def text(self) -> str:
        return "\n\n".join((SYNTHETIC_LINE, *self.paragraphs))


_TW_FINDING = "In the monitor's assessment, the proceedings fell short of the guarantees of a fair trial."

PRECEDENTS: tuple[FixturePrecedent, ...] = (
    FixturePrecedent(
        id="tw-synthetic-a",
        kind="trialwatch_report",
        title="Fairness report: the trial of Ilsa Moravec (synthetic)",
        body="TrialWatch (synthetic fixture)",
        state="Varenia",
        year=2023,
        paragraphs=(
            "Ilsa Moravec is a journalist who edits an independent news website in Varenia.",
            "She was charged with spreading false information after publishing reports on a flood relief fund.",
            "Police held her at a station overnight and questioned her about her sources. "
            "She was arrested at her home and was brought before a judge six days later.",
            "Her lawyer was not allowed to attend the remand hearing and received the case file only the day before trial.",
            _TW_FINDING,
        ),
        facets=(
            FixtureFacet(
                "gc35_48h",
                (
                    "Police held her at a station overnight and questioned her about her sources.",
                    "She was arrested at her home and was brought before a judge six days later.",
                ),
                "monitor_assessment",
                _TW_FINDING,
            ),
            FixtureFacet(
                "iccpr_14_3_b",
                ("Her lawyer was not allowed to attend the remand hearing and received the case file only the day before trial.",),
                "monitor_assessment",
                _TW_FINDING,
            ),
            FixtureFacet("media_defendant", ("Ilsa Moravec is a journalist who edits an independent news website in Varenia.",), "not_examined"),
            FixtureFacet(
                "charge_false_information",
                ("She was charged with spreading false information after publishing reports on a flood relief fund.",),
                "not_examined",
            ),
        ),
    ),
    FixturePrecedent(
        id="ccpr-synthetic-b",
        kind="ccpr_views",
        title="Tomasz Ardel v. Republic of Ostrava (synthetic)",
        body="UN Human Rights Committee (synthetic fixture)",
        state="Ostrava",
        year=2019,
        paragraphs=(
            "The author, a writer, was convicted of criminal defamation of a regional governor.",
            "The court refused to hear the two defence witnesses, without giving reasons.",
            "The judgment reproduced the statement of facts of the indictment word for word.",
            "The Committee concludes that the refusal to hear the defence witnesses violated article 14(3)(e) of the Covenant.",
            "The Committee finds that the author's rights under article 14(1) were violated.",
        ),
        facets=(
            FixtureFacet(
                "iccpr_14_3_e",
                ("The court refused to hear the two defence witnesses, without giving reasons.",),
                "violation_found",
                "The Committee concludes that the refusal to hear the defence witnesses violated article 14(3)(e) of the Covenant.",
            ),
            FixtureFacet(
                "reasoning_reuse",
                ("The judgment reproduced the statement of facts of the indictment word for word.",),
                "violation_found",
                "The Committee finds that the author's rights under article 14(1) were violated.",
            ),
            FixtureFacet("charge_defamation", ("The author, a writer, was convicted of criminal defamation of a regional governor.",), "not_examined"),
        ),
    ),
    FixturePrecedent(
        id="wgad-synthetic-c",
        kind="wgad_opinion",
        title="Opinion concerning Mirela Kostova (synthetic)",
        body="UN Working Group on Arbitrary Detention (synthetic fixture)",
        state="Dravia",
        year=2021,
        paragraphs=(
            "Ms. Kostova is a reporter for a regional newspaper.",
            "Ms. Kostova was arrested at her home and was brought before a judge only five days later.",
            "Her detention was extended three times on the same grounds, repeated in each order.",
            "The Working Group considers that the detention of Ms. Kostova is arbitrary.",
        ),
        facets=(
            FixtureFacet(
                "gc35_48h",
                ("Ms. Kostova was arrested at her home and was brought before a judge only five days later.",),
                "violation_found",
                "The Working Group considers that the detention of Ms. Kostova is arbitrary.",
            ),
            FixtureFacet(
                "renewal_review",
                ("Her detention was extended three times on the same grounds, repeated in each order.",),
                "violation_found",
                "The Working Group considers that the detention of Ms. Kostova is arbitrary.",
            ),
            FixtureFacet("media_defendant", ("Ms. Kostova is a reporter for a regional newspaper.",), "not_examined"),
        ),
    ),
    FixturePrecedent(
        id="tw-synthetic-d",
        kind="trialwatch_report",
        title="Fairness report: the case of Petar Lund (synthetic)",
        body="TrialWatch (synthetic fixture)",
        state="Varenia",
        year=2022,
        paragraphs=(
            "Petar Lund is a blogger who writes about local politics.",
            "He was charged with spreading false news about an election.",
        ),
        facets=(
            FixtureFacet("media_defendant", ("Petar Lund is a blogger who writes about local politics.",), "not_examined"),
            FixtureFacet("charge_false_information", ("He was charged with spreading false news about an election.",), "not_examined"),
        ),
    ),
    FixturePrecedent(
        id="tw-synthetic-e",
        kind="trialwatch_report",
        title="Fairness report: the trial of Anja Brel (synthetic)",
        body="TrialWatch (synthetic fixture)",
        state="Varenia",
        year=2024,
        paragraphs=(
            "Anja Brel is a journalist at a weekly newspaper.",
            "She was charged with spreading false information about a ministry.",
            "She was arrested and was brought before a judge seven days later.",
            "Her lawyer was not present at the remand hearing.",
            "The judgment repeated the indictment word for word.",
        ),
        facets=(
            FixtureFacet("gc35_48h", ("She was arrested and was brought before a judge seven days later.",), "monitor_assessment"),
            FixtureFacet("iccpr_14_3_b", ("Her lawyer was not present at the remand hearing.",), "monitor_assessment"),
            FixtureFacet("reasoning_reuse", ("The judgment repeated the indictment word for word.",), "monitor_assessment"),
            FixtureFacet("media_defendant", ("Anja Brel is a journalist at a weekly newspaper.",), "not_examined"),
            FixtureFacet("charge_false_information", ("She was charged with spreading false information about a ministry.",), "not_examined"),
        ),
        tamper=("seven", "eight"),
    ),
    FixturePrecedent(
        id="ccpr-synthetic-f",
        kind="ccpr_views",
        title="Oskar Venn v. Republic of Dravia (synthetic)",
        body="UN Human Rights Committee (synthetic fixture)",
        state="Dravia",
        year=2018,
        paragraphs=(
            "The author did not understand the language of the court and was given no interpreter.",
            "The court refused to call the expert the defence had named.",
        ),
        facets=(
            FixtureFacet("iccpr_14_3_f", ("The author did not understand the language of the court and was given no interpreter.",), "violation_found"),
            FixtureFacet("iccpr_14_3_e", ("The court refused to call the expert the defence had named.",), "no_violation"),
        ),
    ),
)

SOURCES = (
    ("ccpr_views", "UN Human Rights Committee Views (synthetic fixture)"),
    ("wgad_opinion", "UN Working Group on Arbitrary Detention opinions (synthetic fixture)"),
    ("trialwatch_report", "TrialWatch fairness reports (synthetic fixture)"),
)


def case_facet(facet_id: str, sentence: str, origin: str = "finding") -> CaseFacet:
    """A case fact pattern resting on one invented sentence."""
    span = SourceSpan(doc_id="test-case/notes.txt", start=0, end=len(sentence), text=sentence)
    return CaseFacet(facet_id=facet_id, origin=origin, evidence=(Evidence(role="supporting", span=span),), flag_id="f" if origin == "finding" else None)


# The demo case's fact patterns (see test_precedents.test_the_demo_profile), on invented sentences.
DEMO_LIKE = CaseProfile(
    case_id="test-case",
    facets=(
        case_facet("iccpr_14_3_b", "Defence counsel was not present at the remand hearing."),
        case_facet("iccpr_14_3_e", "The court refused to hear two defence witnesses."),
        case_facet("gc35_48h", "The accused was arrested at his home and was brought before a judge five days later."),
        case_facet("reasoning_reuse", "The judgment repeats the statement of facts of the indictment."),
        case_facet("unaddressed_defense_argument", "The judgment does not answer the defence argument."),
        case_facet("renewal_review", "Detention was extended on the same grounds as before."),
        case_facet("media_defendant", "The accused is a journalist.", "keyword"),
        case_facet("charge_false_information", "He is charged with disseminating false information.", "keyword"),
    ),
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _offset(text: str, quote: str) -> int:
    start = text.find(quote)
    assert start >= 0 and text.find(quote, start + 1) < 0, f"fixture quote not found exactly once: {quote!r}"
    return start


def _write_meta(db: sqlite3.Connection, taxonomy: FactPatternTaxonomy) -> None:
    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "taxonomy_sha": taxonomy.sha,
        "embed_model": EMBED_MODEL_TAG,
        "extraction_model": "synthetic fixture (no model)",
        "prompt_sha": "0" * 64,
        "built_at": BUILT_AT,
        "dropped_quotes": "2",
    }
    db.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", sorted(meta.items()))


def _write_sources(db: sqlite3.Connection) -> None:
    rows = [(kind, name, "https://example.invalid/terms", "2026-10-09", f"{name}: SYNTHETIC test data") for kind, name in SOURCES]
    db.executemany("INSERT INTO sources (kind, name, terms_url, terms_checked, attribution) VALUES (?, ?, ?, ?, ?)", rows)


def _write_document(db: sqlite3.Connection, precedent: FixturePrecedent) -> None:
    verified = precedent.text
    stored = verified.replace(*precedent.tamper) if precedent.tamper else verified
    row = (
        precedent.id, precedent.kind, None, precedent.title, precedent.body, precedent.state, precedent.year,
        f"https://example.invalid/precedents/{precedent.id}", BUILT_AT, _sha(verified + "raw"), _sha(verified), "en",
        f"{precedent.body}: SYNTHETIC test data", stored, 0,
    )  # fmt: skip
    db.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)


def _write_facet(db: sqlite3.Connection, precedent: FixturePrecedent, facet: FixtureFacet, embedder: FakeEmbedder) -> None:
    text = precedent.text
    db.execute("INSERT INTO facets VALUES (?, ?, ?)", (precedent.id, facet.facet_id, facet.finding_kind))
    vectors = embedder.encode(list(facet.facts)).astype(np.float32)
    for quote, vec in zip(facet.facts, vectors, strict=True):
        start = _offset(text, quote)
        db.execute("INSERT INTO quotes VALUES (?, ?, 'fact', ?, ?, 'exact', ?)", (precedent.id, facet.facet_id, start, start + len(quote), vec.tobytes()))
    if facet.finding:
        start = _offset(text, facet.finding)
        db.execute("INSERT INTO quotes VALUES (?, ?, 'finding', ?, ?, 'exact', NULL)", (precedent.id, facet.facet_id, start, start + len(facet.finding)))


def _write_passages(db: sqlite3.Connection, precedent: FixturePrecedent, embedder: FakeEmbedder) -> None:
    """Each paragraph is one search passage, at its place in the verified text."""
    text = precedent.text
    vectors = embedder.encode(list(precedent.paragraphs)).astype(np.float32)
    rows = []
    for paragraph, vec in zip(precedent.paragraphs, vectors, strict=True):
        start = _offset(text, paragraph)
        rows.append((precedent.id, start, start + len(paragraph), vec.tobytes()))
    db.executemany("INSERT INTO passages VALUES (?, ?, ?, ?)", rows)


def build_fixture_corpus(
    path: Path, embedder: FakeEmbedder, taxonomy: FactPatternTaxonomy, precedents: Sequence[FixturePrecedent] = PRECEDENTS
) -> Path:
    """Write a fresh precedents.db at ``path`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        for statement in SQLITE_DDL:
            db.execute(statement)
        _write_meta(db, taxonomy)
        _write_sources(db)
        for precedent in precedents:
            _write_document(db, precedent)
            for facet in precedent.facets:
                _write_facet(db, precedent, facet, embedder)
            _write_passages(db, precedent, embedder)
    return path


def set_meta(path: Path, key: str, value: str | None) -> None:
    """Change (or, with None, delete) one meta value of a built corpus, as a stale build would have it."""
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DELETE FROM meta WHERE key = ?", (key,))
        if value is not None:
            db.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (key, value))
