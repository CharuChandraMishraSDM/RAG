import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_embedding_service
from app.main import app
from app.tests.conftest import FAKE_DIMENSION


@pytest.fixture
def client(fake_embedding_service):
    app.dependency_overrides[get_embedding_service] = lambda: fake_embedding_service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_upload_extracts_chunks_and_embeds_pdf(client, make_pdf):
    pdf = make_pdf(["Connection pool size is 20.", "", "Backups run nightly."])

    response = client.post(
        "/documents/upload", files={"file": ("runbook.pdf", pdf, "application/pdf")}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"].startswith("doc_")
    assert body["filename"] == "runbook.pdf"
    assert body["page_count"] == 3
    assert body["chunk_count"] == 2
    assert body["embedding_dimension"] == FAKE_DIMENSION


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/documents/upload", files={"file": ("notes.txt", b"plain text", "text/plain")}
    )

    assert response.status_code == 400
    assert "not a .pdf" in response.json()["detail"]


def test_upload_rejects_pdf_without_extractable_text(client, make_pdf):
    response = client.post(
        "/documents/upload", files={"file": ("scan.pdf", make_pdf(["", ""]), "application/pdf")}
    )

    assert response.status_code == 400
    assert "no extractable text" in response.json()["detail"]
