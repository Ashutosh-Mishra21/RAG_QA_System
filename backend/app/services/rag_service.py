import logging
from typing import Any, Dict, List

from backend.app.agent import AgenticQueryRunner
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
    EvidenceTools,
)
from backend.app.core.config import settings

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
            embedder=registry.get_embedder(),
        )
        # 🔥 inject rewriter
        self.retriever.query_rewriter = self.query_rewriter
        self.evidence_tools = EvidenceTools(
            self.retriever,
            storage_dir=storage_dir,
            page_parser=self._parse_targeted_evidence if self.ingestion_service else None,
        )
        self.agent = AgenticQueryRunner(
            evidence_tools=self.evidence_tools,
            query_rewriter=self.query_rewriter,
            query_decomposer=self.decomposer,
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

    def _parse_targeted_evidence(self, request, action: str):
        if not self.ingestion_service:
            return []
        return self.ingestion_service.refine_page(
            document_id=request.document_id,
            page_id=request.page_id,
            page_number=request.page_number,
            action=action,
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
        )

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
            logger.exception(
                "Answer aggregation failed; returning concatenated answers"
            )
            return " ".join(answers)

    # =========================
    # 🔹 QUERY → ANSWER (Phase 2)
    # =========================
    def answer(
        self,
        query: str,
        metadata_filters=None,
        document_id: str | None = None,
        tenant_id: str = "default",
        workspace_id: str = "default",
        document_version: str | None = None,
    ) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty")
        if len(query) > settings.RETRIEVAL_MAX_QUERY_CHARS:
            raise ValueError("Query exceeds the maximum supported length")
        filters = dict(metadata_filters or {})
        if document_id:
            filters["document_id"] = document_id
        filters["tenant_id"] = tenant_id
        filters["workspace_id"] = workspace_id
        if document_version:
            filters["document_version"] = document_version

        cached = self.response_cache.get(query, document_id, filters)
        if cached:
            logger.info("Response cache hit for query (document_id=%s)", document_id)
            return self._normalize_chat_result(cached)

        plan, decisions, searches = self.agent.run(query, filters)
        logger.info(
            "Agent query plan type=%s subqueries=%s decisions=%s",
            plan.query_type,
            len(plan.subqueries),
            len(decisions),
        )

        all_answers = []
        all_citations = []
        all_sources = []
        all_confidences = []

        # Solve each planned subquery using the evidence selected by the agent.
        for search in searches:
            rewritten_sq = search["query"]
            sufficiency = search["sufficiency"]
            if sufficiency.sufficient:
                result = self.pipeline.run(
                    query=rewritten_sq,
                    original_query=query,
                    metadata_filters=filters,
                    retrieved=search["chunks"],
                )
            else:
                result = {
                    "answer": "I don't know based on the available evidence.",
                    "citations": [],
                    "confidence": sufficiency.score,
                    "sources": [
                        {
                            "document_id": item.document_id,
                            "page_id": item.page_id,
                            "page_number": item.provenance.source_page,
                            "evidence_id": item.evidence_id,
                            "evidence_type": item.evidence_type,
                        }
                        for item in search["evidence"][:5]
                    ],
                }

            result = self._normalize_chat_result(result)

            all_answers.append(result["answer"])
            all_citations.extend(result.get("citations", []))
            all_sources.extend(result.get("sources", []))
            all_confidences.append(result["confidence"])

        final_answer = (
            all_answers[0]
            if len(all_answers) == 1
            else self.aggregate_answers(query, all_answers)
        )
        unique_sources = list(
            {
                (s.get("evidence_id"), s.get("document_id"), s.get("page_id")): s
                for s in all_sources
                if isinstance(s, dict)
            }.values()
        )
        result = self._normalize_chat_result(
            {
                "answer": final_answer,
                "citations": list(set(all_citations)),  # or extract later
                "confidence": (
                    sum(all_confidences) / len(all_confidences)
                    if all_confidences
                    else 0.0
                ),
                "sources": unique_sources,
            }
        )
        self.response_cache.set(query, result, document_id, filters)
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
