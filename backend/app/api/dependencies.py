from backend.app.core.config import Settings
from backend.app.services.rag_service import RagService

_rag_service = None


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service


def get_settings() -> Settings:
    return Settings()
