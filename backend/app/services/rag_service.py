from typing import Dict, Optional, Any
from backend.app.core import ModelRegistry
from backend.app.generation import (
    AnswerValidator,
    ContextBuilder,
    Generator,
    GenerationPipeline,
    PromptBuilder,
)
from backend.app.models import LLMRouter, OllamaLLM, OpenRouterLLM
from backend.app.retrieval import HybridRetriever, SemanticRetriever


class FallbackLLM:
    def generate(self, prompt: str) -> str:
        return "I don't know"


def _build_llm_router() -> LLMRouter:
    primary = None
    fallback = None

    try:
        primary = OpenRouterLLM("qwen/qwen3-next-80b-a3b-instruct:free")
    except Exception:
        primary = None

    try:
        fallback = OllamaLLM("llama3")
    except Exception:
        fallback = None

    if primary is None and fallback is None:
        return LLMRouter(primary=FallbackLLM(), fallback=None)

    return LLMRouter(primary=primary, fallback=fallback)


class RagService:
    def __init__(self) -> None:
        registry = ModelRegistry.instance()
        self.retriever = HybridRetriever(
            dense_retriever=SemanticRetriever(embedder=registry.get_embedder()),
            keyword_index=registry.get_keyword_index(),
        )
        self.pipeline = GenerationPipeline(
            retriever=self.retriever,
            reranker=registry.get_reranker(),
            context_builder=ContextBuilder(max_chunks=5),
            prompt_builder=PromptBuilder(),
            generator=Generator(_build_llm_router()),
            validator=AnswerValidator(),
        )

    def answer(
        self, query: str, metadata_filters: Optional[Dict[str, Any]] = None
    ) -> dict:
        return self.pipeline.run(query=query, metadata_filters=metadata_filters or {})
