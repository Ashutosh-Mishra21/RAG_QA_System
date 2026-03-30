from fastapi.testclient import TestClient
from backend.app.main import app
import backend.app.api.routes.upload as upload_route
import backend.app.api.routes.chat as chat_route


class FakeIngestionService:
    def save_file(self, filename: str, file_bytes: bytes):
        return {
            "document_id": "doc-1",
            "filename": filename,
            "stored_as": filename,
            "file_path": "/tmp/file.md",
            "timestamp": "now",
        }

    def ingest_and_index(self, file_path: str):
        return {"document_id": "doc-1", "chunks_indexed": 3}


class FakeRagService:
    def answer(self, query: str, metadata_filters=None):
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.1,
            "sources": [],
        }


def test_upload_and_chat_endpoints():
    upload_route._ingestion_service = FakeIngestionService()
    chat_route._rag_service = FakeRagService()

    client = TestClient(app)

    upload = client.post(
        "/api/upload",
        files={"file": ("a.md", b"hello", "text/markdown")},
    )
    assert upload.status_code == 200
    assert upload.json()["pipeline"]["chunks_indexed"] == 3

    chat = client.post("/api/chat", json={"message": "hi"})
    assert chat.status_code == 200
    assert chat.json()["answer"] == "I don't know"
