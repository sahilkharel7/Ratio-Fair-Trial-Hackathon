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
