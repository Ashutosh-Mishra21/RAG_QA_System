import logging
from threading import Lock
from typing import Optional

from backend.app.indexing import Embedder, KeywordIndex
from backend.app.retrieval import CrossEncoderReranker
from backend.app.core.config import settings
from backend.app.core.embedding_provider import EmbeddingModelProvider

logger = logging.getLogger(__name__)


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None
    _lock = Lock()

    def __init__(self) -> None:
        self._embedder: Optional[Embedder] = None
        self._embedding_provider = EmbeddingModelProvider(settings.EMBEDDING_MODEL)
        self._reranker: Optional[CrossEncoderReranker] = None
        self._enricher = None
        self._keyword_index: Optional[KeywordIndex] = None
        self._llm_router = None
        self._qdrant_client = None
        self._resource_lock = Lock()

    @classmethod
    def instance(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_qdrant_client(self):
        """Return the process-shared synchronous Qdrant client and HTTP pool."""
        with self._resource_lock:
            if self._qdrant_client is None:
                from qdrant_client import QdrantClient

                self._qdrant_client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    pool_size=max(1, settings.QDRANT_POOL_SIZE),
                )
            return self._qdrant_client

    def close(self) -> None:
        """Close process-owned clients during application shutdown."""
        with self._resource_lock:
            client, self._qdrant_client = self._qdrant_client, None
        if client is not None:
            client.close()

    def get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(
                model=self._embedding_provider.get_model(),
                batch_size=settings.EMBEDDING_BATCH_SIZE,
            )
        return self._embedder

    def get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def get_enricher(self):
        if self._enricher is None:
            from backend.app.ingestion.enrichment import ChunkEnricher

            self._enricher = ChunkEnricher(
                embedding_model=self._embedding_provider.get_model(),
                top_k_keywords=settings.TOP_K_KEYWORDS,
                enabled=settings.ENABLE_KEYWORD_EXTRACTION,
            )
        return self._enricher

    def get_keyword_index(self) -> KeywordIndex:
        if self._keyword_index is None:
            self._keyword_index = KeywordIndex()
        return self._keyword_index

    def get_llm_router(self):

        from backend.app.models import LLMRouter, OllamaLLM, GoogleAIStudioLLM

        if self._llm_router is not None:
            return self._llm_router

        primary = None
        fallback = None
        gemini_api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        logger.info("[MODEL_REGISTRY] Gemini API key present: %s", bool(gemini_api_key))
        logger.info("[MODEL_REGISTRY] OLLAMA_MODEL: %s", settings.OLLAMA_MODEL)
        logger.info(
            "[MODEL_REGISTRY] OLLAMA_BASE_URL: %s", settings.OLLAMA_BASE_URL
        )

        if gemini_api_key:
            model_name = settings.GEMINI_MODEL
            logger.info(
                "[MODEL_REGISTRY] Initializing primary LLM provider: Google AI Studio (%s)",
                model_name,
            )
            try:
                primary = GoogleAIStudioLLM(model=model_name, api_key=gemini_api_key)
                logger.info(
                    "[MODEL_REGISTRY] Google AI Studio primary initialized successfully"
                )
            except Exception as exc:
                logger.exception(
                    "[MODEL_REGISTRY] Failed to initialize GoogleAIStudioLLM: %s", exc
                )
                primary = None

        if settings.OLLAMA_MODEL or settings.OLLAMA_BASE_URL:
            ollama_model = settings.OLLAMA_MODEL
            ollama_base_url = settings.OLLAMA_BASE_URL
            logger.info(
                "[MODEL_REGISTRY] Configured fallback LLM provider: Ollama (%s @ %s)",
                ollama_model,
                ollama_base_url,
            )
            fallback = OllamaLLM(
                model=ollama_model,
                base_url=ollama_base_url,
            )

            logger.info(
                "[MODEL_REGISTRY] Fallback LLM provider initialized successfully"
            )

        if primary is None and fallback is None:
            raise RuntimeError(
                "No LLM provider configured. Set GEMINI_API_KEY (or GOOGLE_API_KEY) or configure Ollama."
            )

        logger.info(
            "[MODEL_REGISTRY] LLMRouter initialization complete (primary=%s, fallback=%s)",
            primary is not None,
            fallback is not None,
        )

        self._llm_router = LLMRouter(primary=primary, fallback=fallback)
        return self._llm_router
