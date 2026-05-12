import logging
from typing import Any, Dict, List

from backend.app.core import ModelRegistry, ResponseCache
from backend.app.services.ingestion_service import IngestionService

from backend.app.generation import (
    AnswerValidator,
    ContextBuilder,
    GenerationPipeline,
    Generator,
    PromptBuilder,
)

from backend.app.retrieval import (
    HybridRetriever,
    SemanticRetriever,
    QueryRewriter,
    QueryDecomposer,
)

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self, storage_dir=None) -> None:
        registry = ModelRegistry.instance()
        self.response_cache = ResponseCache()
        # 🔹 Ingestion (optional usage)
        self.ingestion_service = IngestionService(storage_dir) if storage_dir else None

        # 🔹 Retrieval
        self.query_rewriter = QueryRewriter(registry.get_llm_router())
        self.decomposer = QueryDecomposer(registry.get_llm_router())
        self.retriever = HybridRetriever(
            dense_retriever=SemanticRetriever(embedder=registry.get_embedder()),
            keyword_index=registry.get_keyword_index(),
            query_rewriter=self.query_rewriter,
        )
        # 🔥 inject rewriter
        self.retriever.query_rewriter = self.query_rewriter

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

    def _normalize_source(self, source: Any) -> dict:
        if hasattr(source, "model_dump"):
            source = source.model_dump()
        elif hasattr(source, "dict"):
            source = source.dict()

        return source if isinstance(source, dict) else {}

    def _normalize_chat_result(self, result: Any) -> dict:
        if isinstance(result, str):
            result = {"answer": result}
        elif not isinstance(result, dict):
            result = {}

        citations = result.get("citations") or []
        if not isinstance(citations, list):
            citations = [citations]

        sources = result.get("sources") or []
        if not isinstance(sources, list):
            sources = [sources]

        confidence = result.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "answer": str(result.get("answer") or ""),
            "citations": [str(citation) for citation in citations],
            "confidence": min(1.0, max(0.0, confidence)),
            "sources": [
                source
                for source in (self._normalize_source(item) for item in sources)
                if source
            ],
        }

    def aggregate_answers(self, query: str, answers: List[str]) -> str:

        prompt = f"""
            Combine the following answers into one coherent final answer.

            - Remove redundancy
            - Keep it concise
            - Ensure logical flow

            Question:
            {query}

            Answers:
            {chr(10).join(answers)}

            Final Answer:
        """

        try:
            final_answer, _, _ = self.pipeline.generator.llm_router.generate(prompt)
            return final_answer
        except Exception:
            logger.exception("Answer aggregation failed; returning concatenated answers")
            return " ".join(answers)

    # =========================
    # 🔹 QUERY → ANSWER (Phase 2)
    # =========================
    def answer(
        self,
        query: str,
        metadata_filters=None,
        document_id: str | None = None,
    ) -> dict:
        filters = metadata_filters or {}
        if document_id:
            filters["document_id"] = document_id

        cached = self.response_cache.get(query, document_id)
        if cached:
            logger.info("Response cache hit for query (document_id=%s)", document_id)
            return self._normalize_chat_result(cached)

        # 🔥 STEP 1: Decompose
        subqueries = self.decomposer.decompose(query)

        logger.info("Query decomposed into %s subquery/subqueries", len(subqueries))
        for i, sq in enumerate(subqueries, start=1):
            logger.info("Subquery %s: %s", i, sq)

        all_answers = []
        all_citations = []
        all_sources = []

        # 🔥 STEP 2: Solve each subquery
        max_hops = 1  # 🔥 control explosion

        for sq in subqueries:
            rewritten_sq = self.query_rewriter.rewrite(sq)

            result = self.pipeline.run(
                query=rewritten_sq,
                original_query=query,
                metadata_filters=filters,
            )

            result = self._normalize_chat_result(result)

            all_answers.append(result["answer"])
            all_citations.extend(result.get("citations", []))
            all_sources.extend(result.get("sources", []))

            # 🔥 STEP 3: Multi-hop follow-up
            for _ in range(max_hops):
                followup_prompt = f"""
                    Based on the answer below, generate a follow-up query to improve understanding.

                    Rules:
                    - Stay relevant to the original question
                    - Do NOT introduce unrelated concepts
                    - Keep it short

                    Original Question:
                    {query}

                    Current Answer:
                    {all_answers[-1]}

                    Follow-up query:
                """

                try:
                    followup_query, _, _ = self.pipeline.generator.llm_router.generate(
                        followup_prompt
                    )

                    followup_query = followup_query.strip()

                    # 🔥 safety check
                    if len(followup_query.split()) < 3:
                        break

                    followup_result = self.pipeline.run(
                        query=followup_query,
                        original_query=query,
                        metadata_filters=filters,
                    )
                    followup_result = self._normalize_chat_result(followup_result)

                    all_answers.append(followup_result["answer"])
                    all_citations.extend(followup_result.get("citations", []))
                    all_sources.extend(followup_result.get("sources", []))

                except Exception:
                    logger.exception("Multi-hop follow-up failed for query: %s", query)
                    break

        final_answer = self.aggregate_answers(query, all_answers)
        unique_sources = list(
            {
                (s.get("document_id"), s.get("section")): s
                for s in all_sources
                if isinstance(s, dict)
            }.values()
        )
        result = self._normalize_chat_result(
            {
                "answer": final_answer,
                "citations": list(set(all_citations)),  # or extract later
                "confidence": min(1.0, len(all_answers) / 3),
                "sources": unique_sources,
            }
        )
        self.response_cache.set(query, result, document_id)
        return result

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
