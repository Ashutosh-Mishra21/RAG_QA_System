import os
import logging
from threading import Lock
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

from backend.app.indexing import Embedder, KeywordIndex
from backend.app.retrieval import CrossEncoderReranker

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None
    _lock = Lock()

    def __init__(self) -> None:
        self._embedder: Optional[Embedder] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._enricher = None
        self._keyword_index: Optional[KeywordIndex] = None
        self._llm_router = None

    @classmethod
    def instance(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def get_enricher(self):
        if self._enricher is None:
            from backend.app.ingestion.enrichment import ChunkEnricher

            self._enricher = ChunkEnricher()
        return self._enricher

    def get_keyword_index(self) -> KeywordIndex:
        if self._keyword_index is None:
            self._keyword_index = KeywordIndex()
        return self._keyword_index

    def get_llm_router(self):

        from backend.app.models import LLMRouter, OllamaLLM, OpenRouterLLM

        if self._llm_router is not None:
            return self._llm_router

        primary = None
        fallback = None
        logger.info(
            "[MODEL_REGISTRY] OPENROUTER_API_KEY present: %s",
            bool(os.getenv("OPENROUTER_API_KEY")),
        )
        logger.info("[MODEL_REGISTRY] OLLAMA_MODEL: %s", os.getenv("OLLAMA_MODEL"))
        logger.info(
            "[MODEL_REGISTRY] OLLAMA_BASE_URL: %s", os.getenv("OLLAMA_BASE_URL")
        )

        if os.getenv("OPENROUTER_API_KEY"):
            model_name = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
            print(
                "\n",
                f"[MODEL_REGISTRY] OPENROUTER_API_KEY found. Initializing OpenRouter primary: {model_name}",
            )
            logger.info(
                "[MODEL_REGISTRY] Initializing primary LLM provider: OpenRouter (%s)",
                model_name,
            )
            try:
                primary = OpenRouterLLM(model=model_name)
                print("[MODEL_REGISTRY] Primary provider ready: OpenRouter")
                logger.info(
                    "[MODEL_REGISTRY] Primary LLM provider initialized successfully"
                )
            except Exception as exc:
                print(
                    f"[MODEL_REGISTRY] Primary provider init failed. API will be skipped: {exc}"
                )
                logger.exception(
                    "[MODEL_REGISTRY] Failed to initialize OpenRouterLLM: %s", exc
                )
                primary = None

        if os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_BASE_URL"):
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "No real LLM provider configured. Set OPENROUTER_API_KEY or OLLAMA_* env vars."
            )

        logger.info(
            "[MODEL_REGISTRY] LLMRouter initialization complete (primary=%s, fallback=%s)",
            primary is not None,
            fallback is not None,
        )

        self._llm_router = LLMRouter(primary=primary, fallback=fallback)
        return self._llm_router
