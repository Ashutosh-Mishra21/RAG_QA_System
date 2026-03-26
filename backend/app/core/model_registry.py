import os
from threading import Lock
from typing import Optional
from dotenv import load_dotenv, find_dotenv

from backend.app.indexing import Embedder, KeywordIndex
from backend.app.retrieval import CrossEncoderReranker

load_dotenv(find_dotenv())


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
        print("OPENROUTER_API_KEY:", os.getenv("OPENROUTER_API_KEY"))
        print("OLLAMA_MODEL:", os.getenv("OLLAMA_MODEL"))
        print("OLLAMA_BASE_URL:", os.getenv("OLLAMA_BASE_URL"))
        if self._llm_router is not None:
            return self._llm_router

        from backend.app.models import LLMRouter, OllamaLLM, OpenRouterLLM

        primary = None
        fallback = None

        if os.getenv("OPENROUTER_API_KEY"):
            model_name = os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.5-air:free")
            primary = OpenRouterLLM(model=model_name)

        if os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_BASE_URL"):
            fallback = OllamaLLM(
                model=os.getenv("OLLAMA_MODEL", "llama3"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

        if primary is None and fallback is None:
            raise RuntimeError(
                "No real LLM provider configured. Set OPENROUTER_API_KEY or OLLAMA_* env vars."
            )

        self._llm_router = LLMRouter(primary=primary, fallback=fallback)
        return self._llm_router
