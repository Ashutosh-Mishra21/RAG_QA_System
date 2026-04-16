from typing import Any, Dict, Optional

from backend.app.core import ModelRegistry
from backend.app.services.ingestion_service import IngestionService

from backend.app.generation import (
    AnswerValidator,
    ContextBuilder,
    GenerationPipeline,
    Generator,
    PromptBuilder,
)

from backend.app.retrieval import HybridRetriever, SemanticRetriever, QueryRewriter


class RagService:
    def __init__(self, storage_dir=None) -> None:
        registry = ModelRegistry.instance()

        # 🔹 Ingestion (optional usage)
        self.ingestion_service = IngestionService(storage_dir) if storage_dir else None

        # 🔹 Retrieval
        self.query_rewriter = QueryRewriter(registry.get_llm_router())
        self.retriever = HybridRetriever(
            dense_retriever=SemanticRetriever(embedder=registry.get_embedder()),
            keyword_index=registry.get_keyword_index(),
        )

        # 🔹 Generation Pipeline (core)
        self.pipeline = GenerationPipeline(
            retriever=self.retriever,
            reranker=registry.get_reranker(),
            context_builder=ContextBuilder(max_chunks=5),
            prompt_builder=PromptBuilder(),
            generator=Generator(registry.get_llm_router()),
            validator=AnswerValidator(),
        )

    # =========================
    # 🔹 INGESTION (Phase 1)
    # =========================
    def ingest(self, file_path: str):
        if not self.ingestion_service:
            raise ValueError("IngestionService not initialized")
        return self.ingestion_service.ingest_and_index(file_path)

    # =========================
    # 🔹 QUERY → ANSWER (Phase 2)
    # =========================
    def answer(self, query: str, metadata_filters=None) -> dict:

        # 🔥 STEP 1: Rewrite query
        rewritten_query = self.query_rewriter.rewrite(query)

        print(f"\n[QUERY REWRITE]")
        print(f"Original: {query}")
        print(f"Rewritten: {rewritten_query}\n")

        # 🔥 STEP 2: Use rewritten query
        return self.pipeline.run(
            query=rewritten_query,
            metadata_filters=metadata_filters or {},
        )

    # =========================
    # 🔥 EVALUATION ENTRYPOINT (Phase 3)
    # =========================
    def run_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Unified interface for evaluation layer
        """

        result = self.answer(query)

        # 🔥 IMPORTANT: use real context (modify pipeline to return it)
        context = result.get("context", "")

        return {
            "answer": result["answer"],
            "context": context,
        }
