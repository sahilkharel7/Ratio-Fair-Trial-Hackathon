import sqlite3
from pathlib import Path
import pytest
from ratio.extraction.loader import CaseManifest, LoaderError
from ratio.library import LibraryStore
from ratio.paths import DEMO_DIR
from ratio.store import CaseStore
from ratio.workspace import seed_collection, case_payload, focus_report


@pytest.fixture
def library(tmp_path):
    return LibraryStore(CaseStore(tmp_path / "cases.db"))


def stage_sorin(library, id="uploaded-sorin"):
    folder = DEMO_DIR / "focused" / "sorin-2025"
    raw = {
        name: (folder / name).read_bytes()
        for name in ("indictment.txt", "judgment.txt")
    }
    library.stage(raw, provenance="synthetic")
    ids = {f["name"]: f["id"] for f in library.pending_files()}
    manifest = CaseManifest(
        case_id=id,
        title="Case of Nadia Sorin",
        court="Selected court",
        charge_type="Not recorded",
        data_provenance="synthetic",
        synthetic=True,
        documents=[
            {
                "path": name,
                "type": "indictment" if name == "indictment.txt" else "judgment",
                "title": name,
            }
            for name in raw
        ],
    )
    return manifest, ids


def test_collection_survives_restart_and_does_not_promote_reference_history(library):
    seed_collection(library)
    assert len(library.list()) == 3
    reopened = LibraryStore(CaseStore(library.cases.path))
    assert {c["case_id"] for c in reopened.list()} == {
        "venn-2025",
        "sorin-2025",
        "maren-2025",
    }
    assert len(reopened.list("Mirevo District Court, Criminal Chamber")) == 3
    assert reopened.list("Another court") == []


NEW_NOTE = (
    "SYNTHETIC: fictional case; all people, courts, country, language and laws are invented.\n"
    "MONITORING NOTE\nHearing date: 5 August 2025\nThe defence asked the court for more time to prepare.\n"
)


def test_a_case_is_deleted_with_everything_stored_for_it(library):
    manifest, ids = stage_sorin(library)
    library.import_manifest(manifest, ids)
    library.add_note("uploaded-sorin", "Check the monitor's second note.")

    library.delete_case("uploaded-sorin")

    assert library.cases.load_case("uploaded-sorin") is None
    assert all(c["case_id"] != "uploaded-sorin" for c in library.list())
    with library.connect() as conn:
        for table in ("matter_files", "case_notes", "matter_catalogue", "cases"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE case_id=?", ("uploaded-sorin",)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM intake_files WHERE assigned_case_id=?", ("uploaded-sorin",)).fetchone()[0] == 0
    with pytest.raises(KeyError):
        library.delete_case("uploaded-sorin")


def test_files_are_added_to_an_uploaded_case_without_moving_its_sources(library):
    manifest, ids = stage_sorin(library)
    library.import_manifest(manifest, ids)
    before = {d.path: d.text for d in library.cases.load_case("uploaded-sorin").documents}
    library.stage({"note-2.txt": NEW_NOTE.encode()}, provenance="synthetic")
    new_id = next(f["id"] for f in library.pending_files() if f["name"] == "note-2.txt")

    library.add_files("uploaded-sorin", [new_id], "monitoring_note")

    stored = library.cases.load_case("uploaded-sorin")
    assert {d.path for d in stored.documents} == {"indictment.txt", "judgment.txt", "note-2.txt"}
    assert all(d.text == before[d.path] for d in stored.documents if d.path in before)
    assert not any(f["id"] == new_id for f in library.pending_files())
    with pytest.raises(LoaderError):  # each file is assigned once
        library.add_files("uploaded-sorin", [new_id], "monitoring_note")


def test_the_demo_cases_loaded_from_the_examples_are_read_only(library):
    seed_collection(library)
    library.stage({"note-2.txt": NEW_NOTE.encode()}, provenance="synthetic")
    new_id = library.pending_files()[0]["id"]
    with pytest.raises(LoaderError):
        library.add_files("venn-2025", [new_id], "monitoring_note")
    assert library.pending_files()[0]["id"] == new_id  # nothing was assigned


def test_a_saved_public_judgment_opens_as_an_active_case(library):
    text = b"Synthetic Court of Testland. The applicant was charged with an invented offence. The court sat in public."
    kwargs = dict(doc_id="court-testland-1/99", title="Applicant v. Testland (synthetic)",
                  court="Synthetic Court of Testland", source_url="https://example.org/judgment",
                  filename="court-testland-1-99.txt", content=text)

    case_id = library.open_reference_case(**kwargs)

    record = library.cases.load_case(case_id)
    assert case_id == "court-testland-1-99" and record.meta.data_provenance == "public"
    assert [d.type for d in record.documents] == ["judgment"]
    assert library.focus(case_id).charge == "an invented offence"
    assert library.focus(case_id).defendant == "Applicant"  # "Applicant v. State": the person is named first
    assert library.open_reference_case(**kwargs) == case_id  # opening it again reuses the case
    assert sum(c["case_id"] == case_id for c in library.list()) == 1


def test_stored_charge_panels_are_recomputed_with_the_current_rules(library):
    manifest, ids = stage_sorin(library)
    library.import_manifest(manifest, ids)
    fresh = library.focus("uploaded-sorin")
    with library.connect() as conn:  # a panel saved by an earlier version of the rules
        stale = fresh.model_copy(update={"charge": "Not recorded", "charge_span": None})
        conn.execute("UPDATE matter_catalogue SET charge=?, focus_json=? WHERE case_id=?",
                     ("Not recorded", stale.model_dump_json(), "uploaded-sorin"))

    assert library.refresh_focus() >= 1

    assert library.focus("uploaded-sorin").charge == fresh.charge != "Not recorded"


def test_demo_seeding_skips_entries_that_are_not_case_folders(library, tmp_path, monkeypatch):
    import shutil

    import ratio.workspace

    demo = tmp_path / "demo"
    shutil.copytree(DEMO_DIR / "focused", demo / "focused")
    (demo / "focused" / ".DS_Store").write_bytes(b"\0\0\0\1Bud1")
    (demo / "focused" / "notes").mkdir()
    monkeypatch.setattr(ratio.workspace, "DEMO_DIR", demo)

    seed_collection(library)

    assert {c["case_id"] for c in library.list()} == {"venn-2025", "sorin-2025", "maren-2025"}


def test_staged_files_are_assigned_once_and_open_from_sqlite(library):
    manifest, ids = stage_sorin(library)
    assert len(library.pending_files()) == 2
    record = library.import_manifest(manifest, ids, defendant="Nadia Sorin")
    assert library.pending_files() == []
    reopened = LibraryStore(CaseStore(library.cases.path))
    payload = case_payload(reopened, record.case_id)
    assert payload["focus"]["defendant"] == "Nadia Sorin"
    assert payload["focus"]["prompts"][0]["pattern"] == "burden_shift"
    assert "download_url" in payload["record"]["documents"][0]
    assert reopened.list()[0]["court"] == "Selected court"
    with pytest.raises(LoaderError, match="already stored"):
        reopened.import_manifest(manifest, ids)


def test_import_rejects_reused_file_or_wrong_declaration_without_partial_case(library):
    manifest, ids = stage_sorin(library)
    with pytest.raises(LoaderError, match="once"):
        library.import_manifest(
            manifest, {path: next(iter(ids.values())) for path in ids}
        )
    assert library.cases.load_case(manifest.case_id) is None
    assert len(library.pending_files()) == 2
    public = manifest.model_copy(
        update={
            "data_provenance": "public",
            "synthetic": False,
            "source_note": "Published reference",
        }
    )
    with pytest.raises(LoaderError, match="combine"):
        library.import_manifest(public, ids)
    assert library.cases.load_case(manifest.case_id) is None


def test_malformed_pdf_stays_available_for_ocr_and_cannot_be_silently_analysed(library):
    library.stage(
        {"scan.pdf": b"not a PDF"}, provenance="public", source_note="Test publication"
    )
    assert library.pending_files()[0]["problem"]
    manifest = CaseManifest(
        case_id="scan",
        title="Scan",
        court="Court",
        charge_type="Unknown",
        data_provenance="public",
        synthetic=False,
        source_note="Test publication",
        documents=[{"path": "scan.pdf", "type": "judgment", "title": "Scan"}],
    )
    with pytest.raises(LoaderError):
        library.import_manifest(
            manifest, {"scan.pdf": library.pending_files()[0]["id"]}
        )
    assert library.cases.load_case("scan") is None


def test_assessments_keep_their_source_snapshot_and_history(library):
    seed_collection(library)
    prompt = library.focus("sorin-2025").prompts[0].model_dump(mode="json")
    decision = {
        "case_id": "sorin-2025",
        "prompt_id": prompt["id"],
        "decision": "supported",
        "reason": "Source-backed demonstration assessment",
        "source_snapshot": prompt,
    }
    library.save_decision(decision)
    with pytest.raises(ValueError, match="unchanged"):
        library.save_decision(
            {**decision, "source_snapshot": {**prompt, "title": "Changed"}}
        )
    library.save_decision({**decision, "decision": "reopened", "reason": ""})
    assert len(library.decisions("sorin-2025")) == 2
    assert (
        next(c for c in library.list() if c["case_id"] == "sorin-2025")[
            "pending_prompts"
        ]
        == 1
    )
    library.add_note(
        "sorin-2025", "Ask the monitor how the burden of proof was handled."
    )
    report = focus_report(library, "sorin-2025")
    assert prompt["span"]["text"].splitlines()[-1] in report
    assert "Source-backed demonstration assessment" in report and "reopened" in report
    assert "Ask the monitor" in report


def test_public_sources_and_input_limits_are_enforced(library):
    with pytest.raises(LoaderError, match="publication"):
        library.stage({"doc.txt": b"Text"}, provenance="public")
    with pytest.raises(LoaderError):
        library.stage({"../doc.txt": b"Text"}, provenance="synthetic")
    with pytest.raises(LoaderError):
        library.stage({"doc.txt": b"x" * 5_000_001}, provenance="synthetic")
