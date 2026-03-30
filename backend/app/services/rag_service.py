from typing import Any, Dict, Optional

from backend.app.core import ModelRegistry
from backend.app.generation import (
    AnswerValidator,
    ContextBuilder,
    GenerationPipeline,
    Generator,
    PromptBuilder,
)
from backend.app.retrieval import HybridRetriever, SemanticRetriever


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
            generator=Generator(registry.get_llm_router()),
            validator=AnswerValidator(),
        )

    def answer(
        self, query: str, metadata_filters: Optional[Dict[str, Any]] = None
    ) -> dict:
        return self.pipeline.run(query=query, metadata_filters=metadata_filters or {})
