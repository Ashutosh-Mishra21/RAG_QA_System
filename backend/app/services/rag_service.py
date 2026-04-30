from typing import Any, Dict, List

from backend.app.core import ModelRegistry
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


class RagService:
    def __init__(self, storage_dir=None) -> None:
        registry = ModelRegistry.instance()

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
            final_answer, _, _ = self.generator.llm_router.generate(prompt)
            return final_answer
        except:
            return " ".join(answers)

    # =========================
    # 🔹 QUERY → ANSWER (Phase 2)
    # =========================
    def answer(self, query: str, metadata_filters=None) -> dict:

        # 🔥 STEP 1: Decompose
        subqueries = self.decomposer.decompose(query)

        print(f"\n[DECOMPOSITION]")
        for i, sq in enumerate(subqueries):
            print(f"{i+1}. {sq}")

        all_answers = []
        all_sources = []

        # 🔥 STEP 2: Solve each subquery
        max_hops = 1  # 🔥 control explosion

        for sq in subqueries:
            rewritten_sq = self.query_rewriter.rewrite(sq)

            result = self.pipeline.run(
                query=rewritten_sq,
                original_query=query,
                metadata_filters=metadata_filters or {},
            )

            all_answers.append(result["answer"])
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
                    {result["answer"]}

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
                        metadata_filters=metadata_filters or {},
                    )

                    all_answers.append(followup_result["answer"])
                    all_sources.extend(followup_result.get("sources", []))

                except Exception:
                    break

        final_answer = self.aggregate_answers(query, all_answers)

        return {
            "answer": final_answer,
            "sources": all_sources,
        }

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
