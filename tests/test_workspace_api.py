import base64
import hashlib
import importlib.util
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from ratio.library import LibraryStore
from ratio.paths import REPO_ROOT
from ratio.store import CaseStore

spec = importlib.util.spec_from_file_location(
    "serve_workspace", REPO_ROOT / "scripts" / "serve_workspace.py"
)
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)


@pytest.fixture
def api(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>Case workspace</html>")
    library = LibraryStore(CaseStore(tmp_path / "cases.db"))
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), server_module.handler_for(library, dist)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", library
    server.shutdown()
    server.server_close()
    thread.join()


def call(root, path, body=None, headers=None):
    request = Request(
        root + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urlopen(request) as response:
        return json.load(response)


def test_upload_pdf_creates_a_case_opens_persisted_record_and_downloads_original(api):
    root, library = api
    raw = (REPO_ROOT / "tests" / "fixtures" / "corpus" / "report.pdf").read_bytes()
    staged = call(
        root,
        "/api/intake",
        {
            "provenance": "synthetic",
            "files": [
                {"name": "report.pdf", "content": base64.b64encode(raw).decode()}
            ],
        },
    )
    file = staged["pending_files"][0]
    assert not file["problem"]
    created = call(
        root,
        "/api/cases",
        {
            "defendant": "Tamsin Orlo",
            "file_ids": {"report.pdf": file["id"]},
            "manifest": {
                "case_id": "pdf-example",
                "title": "Tamsin Orlo example",
                "court": "Selected court",
                "charge_type": "Not recorded",
                "data_provenance": "synthetic",
                "synthetic": True,
                "documents": [
                    {
                        "path": "report.pdf",
                        "type": "monitoring_note",
                        "title": "Report PDF",
                    }
                ],
            },
        },
    )
    assert created["record"]["case_id"] == "pdf-example"
    assert call(root, "/api/library")["cases"][0]["defendant"] == "Tamsin Orlo"
    opened = call(root, "/api/cases/pdf-example")
    assert (
        opened["record"]["documents"][0]["text"]
        == created["record"]["documents"][0]["text"]
    )
    url = opened["record"]["documents"][0]["download_url"]
    with urlopen(root + url) as response:
        downloaded = response.read()
    assert hashlib.sha256(downloaded).digest() == hashlib.sha256(raw).digest()
    call(
        root,
        "/api/case-notes",
        {
            "case_id": "pdf-example",
            "note": "Ask for the indictment and the prosecution’s sentencing request.",
        },
    )
    assert (
        "prosecution"
        in LibraryStore(CaseStore(library.cases.path)).notes("pdf-example")[0]["note"]
    )


def test_browser_cross_origin_and_dns_rebinding_hosts_are_rejected(api):
    root, _ = api
    for headers in (
        {"Origin": "https://external.example"},
        {"Host": "external.example"},
        {"Sec-Fetch-Site": "cross-site"},
    ):
        with pytest.raises(HTTPError) as raised:
            call(root, "/api/library", headers=headers)
        assert raised.value.code == 403


def test_unknown_case_is_not_replaced_with_the_demo(api):
    root, _ = api
    with pytest.raises(HTTPError) as raised:
        call(root, "/api/cases/missing")
    assert raised.value.code == 404
