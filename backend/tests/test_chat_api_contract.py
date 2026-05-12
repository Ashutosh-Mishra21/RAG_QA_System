from fastapi.testclient import TestClient

from backend.app.api.dependencies import get_rag_service
from backend.app.main import app


class FakeRagService:
    def answer(self, message: str) -> dict:
        return {
            "answer": f"Answer for: {message}",
            "citations": ["1"],
            "confidence": 1.0,
            "sources": [{"document_id": "doc-1", "section": "Intro"}],
            "provider": "test-provider",
        }


class FailingRagService:
    def answer(self, message: str) -> dict:
        raise RuntimeError("provider exploded")


def test_chat_endpoint_returns_standard_success_contract():
    app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": {
            "answer": "Answer for: hello",
            "citations": ["1"],
            "confidence": 1.0,
            "sources": [{"document_id": "doc-1", "section": "Intro"}],
        },
        "error": None,
    }

    app.dependency_overrides.clear()


def test_chat_endpoint_returns_standard_error_contract():
    app.dependency_overrides[get_rag_service] = lambda: FailingRagService()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 500
    assert response.json() == {
        "success": False,
        "data": None,
        "error": "Chat generation failed",
    }

    app.dependency_overrides.clear()


def test_chat_endpoint_validation_uses_standard_error_contract():
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 422
    assert response.json() == {
        "success": False,
        "data": None,
        "error": "Invalid request payload",
    }
