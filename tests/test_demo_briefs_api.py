"""The optional demo API generates and downloads a source-checked draft locally."""

import base64
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_demo_briefs import QUOTE, draft, one_page_pdf
from test_workspace_api import call, server_module

from ratio.demo_briefs import DemoBriefService
from ratio.library import LibraryStore
from ratio.store import CaseStore
from ratio.testing import make_record


@pytest.fixture
def demo_api(tmp_path):
    library = LibraryStore(CaseStore(tmp_path / "cases.db"))
    record = make_record([("judgment.txt", "judgment", QUOTE)])
    library.cases.save_case(record)
    library.register(record)
    raw = one_page_pdf()

    def runner(_):
        return {
            "sections": draft(record.documents[0].id).model_dump()["sections"],
            "pdf": base64.b64encode(raw).decode(),
        }

    service = DemoBriefService(library, runner=runner)
    dist = tmp_path / "dist"
    dist.mkdir()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), server_module.handler_for(library, dist, briefs=service)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", service, record, raw
    server.shutdown()
    server.server_close()
    service.close()
    thread.join(timeout=2)


def test_start_poll_download_and_cross_origin_rejection(demo_api):
    root, service, record, raw = demo_api
    path = f"/api/cases/{record.case_id}/brief"
    assert call(root, "/api/library")["ai_briefs"]["enabled"]
    assert call(root, path)["status"] == "not_generated"
    req = Request(
        root + path,
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "Origin": "https://outside.invalid",
        },
    )
    with pytest.raises(HTTPError) as rejected:
        urlopen(req)
    assert rejected.value.code == 403
    assert service.get(record.case_id)["status"] == "not_generated"
    assert call(root, path, {})["status"] in {"queued", "running", "ready"}
    service._pool.submit(lambda: None).result(timeout=3)
    assert call(root, path)["status"] == "ready"
    with urlopen(root + path + ".pdf") as response:
        assert response.read() == raw
        assert response.headers["Cache-Control"] == "no-store"
    assert "API_KEY" not in str(call(root, path))
