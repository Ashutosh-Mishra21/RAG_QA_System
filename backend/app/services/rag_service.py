import logging
from typing import Any, Dict
from langsmith import traceable, get_current_run_tree

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
            page_parser=(
                self._parse_targeted_evidence if self.ingestion_service else None
            ),
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

        normalized = {
            "answer": str(result.get("answer") or ""),
            "context": str(result.get("context") or ""),
            "citations": [str(citation) for citation in citations],
            "confidence": min(1.0, max(0.0, confidence)),
            "sources": [
                source
                for source in (self._normalize_source(item) for item in sources)
                if source
            ],
        }
        for field in ("model", "provider", "fallback_triggered", "reason"):
            if field in result:
                normalized[field] = result[field]
        return normalized

    @staticmethod
    def _stream_final_data(result: dict) -> dict:
        """Return chat-facing fields without exposing the assembled prompt context."""
        fields = (
            "answer",
            "citations",
            "confidence",
            "sources",
            "model",
            "provider",
            "fallback_triggered",
        )
        return {field: result[field] for field in fields if field in result}

    @traceable(name="rag.answer.stream", run_type="chain")
    def stream_answer(
        self,
        query: str,
        metadata_filters=None,
        document_id: str | None = None,
        tenant_id: str = "default",
        workspace_id: str = "default",
        document_version: str | None = None,
    ):
        """Yield answer chunks and final metadata using the existing RAG flow."""
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.metadata.update(
                {
                    "tenant_id": tenant_id,
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "document_version": document_version,
                    "streaming": True,
                }
            )
            run_tree.tags.extend(["rag", "production", "streaming"])

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
            logger.info(
                "Response cache hit for streaming query (document_id=%s)",
                document_id,
            )
            result = self._normalize_chat_result(cached)
            if result["answer"]:
                yield {"event": "chunk", "data": {"text": result["answer"]}}
            yield {"event": "final", "data": self._stream_final_data(result)}
            return

        plan, decisions, searches = self.agent.run(query, filters)
        logger.info(
            "Streaming agent query plan type=%s subqueries=%s decisions=%s",
            plan.query_type,
            len(plan.subqueries),
            len(decisions),
        )

        retrieved = []
        seen_chunks = set()
        for search in searches:
            for chunk in search["chunks"]:
                chunk_key = getattr(chunk, "id", None) or (
                    (chunk.metadata or {}).get("document_id"),
                    (chunk.metadata or {}).get("page_id"),
                    chunk.content,
                )
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    retrieved.append(chunk)

        if not retrieved:
            result = {
                "answer": "I don't know based on the available evidence.",
                "context": "",
                "citations": [],
                "confidence": 0.0,
                "sources": [],
            }
            try:
                self.response_cache.set(query, result, document_id, filters)
            except Exception:
                logger.exception("Unable to cache completed empty-evidence response")
            yield {"event": "chunk", "data": {"text": result["answer"]}}
            yield {"event": "final", "data": self._stream_final_data(result)}
            return

        result = None
        for event in self.pipeline.stream(
            query=query,
            original_query=query,
            metadata_filters=filters,
            retrieved=retrieved,
        ):
            if event["event"] == "chunk":
                yield event
            elif event["event"] == "final":
                result = self._normalize_chat_result(event["data"])

        if result is None:
            raise RuntimeError("Generation stream ended without final metadata")

        # Cache only a completed, validated response. Interrupted/failed streams
        # leave no partial answer in the response cache.
        try:
            self.response_cache.set(query, result, document_id, filters)
        except Exception:
            logger.exception("Unable to cache completed streaming response")
        yield {"event": "final", "data": self._stream_final_data(result)}

    # =========================
    # 🔹 QUERY → ANSWER (Phase 2)
    # =========================
    @traceable(name="rag.answer", run_type="chain")
    def answer(
        self,
        query: str,
        metadata_filters=None,
        document_id: str | None = None,
        tenant_id: str = "default",
        workspace_id: str = "default",
        document_version: str | None = None,
    ) -> dict:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.metadata.update(
                {
                    "tenant_id": tenant_id,
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "document_version": document_version,
                }
            )

            run_tree.tags.extend(
                [
                    "rag",
                    "production",
                ]
            )

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

        # Decomposition is for finding evidence. Generate once from the union so
        # the original question, citation map, and answer validation stay aligned.
        retrieved = []
        seen_chunks = set()
        for search in searches:
            for chunk in search["chunks"]:
                chunk_key = getattr(chunk, "id", None) or (
                    (chunk.metadata or {}).get("document_id"),
                    (chunk.metadata or {}).get("page_id"),
                    chunk.content,
                )
                if chunk_key in seen_chunks:
                    continue
                seen_chunks.add(chunk_key)
                retrieved.append(chunk)

        if not retrieved:
            result = {
                "answer": "I don't know based on the available evidence.",
                "context": "",
                "citations": [],
                "confidence": 0.0,
                "sources": [],
            }
        else:
            result = self.pipeline.run(
                query=query,
                original_query=query,
                metadata_filters=filters,
                retrieved=retrieved,
            )
        result = self._normalize_chat_result(result)
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
