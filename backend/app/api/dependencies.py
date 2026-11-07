from backend.app.core.config import Settings
from pathlib import Path
from backend.app.services.rag_service import RagService

_rag_service = None


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService(storage_dir=Path("data"))
    return _rag_service


def get_settings() -> Settings:
    return Settings()


def reset_rag_service() -> None:
    """Release request-serving resources before the app lifespan ends."""
    global _rag_service
    service, _rag_service = _rag_service, None
    if service is not None:
        service.close()
