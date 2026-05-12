import logging

from fastapi import APIRouter

from backend.app.api.responses import success_response
from backend.app.core.config import settings
from backend.app.core.constants import APP_VERSION

router = APIRouter(tags=["health"])

logger = logging.getLogger(__name__)


@router.get("/health")
def health_check():

    # =========================================
    # 🔹 LLM STATUS
    # =========================================
    llm_available = bool(settings.OPENROUTER_API_KEY)

    # =========================================
    # 🔹 QDRANT STATUS
    # =========================================
    qdrant_available = False

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

        client.get_collections()

        qdrant_available = True

    except Exception as e:
        logger.warning("Qdrant health check failed: %s", e)

    # =========================================
    # 🔹 FINAL RESPONSE
    # =========================================
    overall_status = "healthy" if qdrant_available else "degraded"

    return success_response(
        {
            "status": overall_status,
            "version": APP_VERSION,
            "services": {
                "llm": llm_available,
                "qdrant": qdrant_available,
            },
        }
    )
