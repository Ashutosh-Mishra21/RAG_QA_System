import logging
import time
from typing import Dict, Any, Optional
from langsmith import get_current_run_tree, traceable

logger = logging.getLogger(__name__)


class GenerationPipeline:
    def __init__(
        self, retriever, reranker, context_builder, prompt_builder, generator, validator,
        sufficiency_checker=None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.validator = validator
        self.sufficiency_checker = sufficiency_checker

    def _assess_final_context(self, query, chunks, run_tree):
        if self.sufficiency_checker is None:
            return None
        assessment = self.sufficiency_checker(query, chunks)
        reasons = list(getattr(assessment, "reasons", []) or [])
        score = float(getattr(assessment, "score", 0.0) or 0.0)
        sufficient = bool(getattr(assessment, "sufficient", False))
        logger.info(
            "[PIPELINE] Final context sufficiency sufficient=%s score=%.4f reasons=%s",
            sufficient,
            score,
            reasons,
        )
        if run_tree:
            run_tree.metadata.update(
                {
                    "final_context_sufficient": sufficient,
                    "final_context_sufficiency_score": score,
                    "final_context_sufficiency_reasons": reasons,
                }
            )
        return assessment

    @staticmethod
    def _insufficient_context_result(context, top_chunks, assessment):
        reasons = list(getattr(assessment, "reasons", []) or [])
        reason = "; ".join(reasons) or "selected context did not meet the sufficiency threshold"
        return {
            "answer": "I don't know based on the available evidence.",
            "context": context,
            "citations": [],
            "confidence": 0.0,
            "sources": [chunk.metadata for chunk in top_chunks],
            "reason": "final context insufficient: " + reason,
        }

    @traceable(name="generation.run", run_type="chain")
    def run(
        self,
        query: str,
        original_query: Optional[str] = None,
        metadata_filters=None,
        retrieved=None,
    ) -> Dict[str, Any]:

        run_tree = get_current_run_tree()
        retrieval_started = time.perf_counter()
        if retrieved is None:
            retrieved = self.retriever.retrieve(
                query=query,
                original_query=original_query,
                top_k=40,
                metadata_filters=metadata_filters or {},
            )
        logger.info(
            "[PIPELINE] Retrieved chunks: %s (stage_ms=%.1f)",
            len(retrieved),
            (time.perf_counter() - retrieval_started) * 1000,
        )

        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(query, retrieved)
        top_chunks = reranked[:5]
        logger.info(
            "[PIPELINE] Reranking completed stage_ms=%.1f top_chunks=%s",
            (time.perf_counter() - rerank_started) * 1000,
            len(top_chunks),
        )
        logger.info("[PIPELINE] Top chunks after rerank: %s", len(top_chunks))
        if run_tree:
            run_tree.metadata.update(
                {
                    "retrieved_count": len(retrieved),
                    "top_chunks_count": len(top_chunks),
                }
            )

        context_started = time.perf_counter()
        context, citation_map = self.context_builder.build(top_chunks)
        logger.info(
            "[PIPELINE] Context assembly completed stage_ms=%.1f context_chars=%s",
            (time.perf_counter() - context_started) * 1000,
            len(context),
        )
        context_assessment = self._assess_final_context(query, top_chunks, run_tree)
        if context_assessment is not None and not context_assessment.sufficient:
            return self._insufficient_context_result(
                context, top_chunks, context_assessment
            )

        llm = self.generator.llm_router
        logger.info("[PIPELINE] llm.primary is None: %s", llm.primary is None)

        expected_provider = "api" if llm.primary else "local"
        # first try with API-style (default)
        initial_prompt = self.prompt_builder.build_prompt(
            query, context, expected_provider
        )

        generation_started = time.perf_counter()
        answer, used_model, provider = self.generator.generate(initial_prompt)
        logger.info("[PIPELINE] Provider used for initial generation: %s", provider)

        fallback_triggered = expected_provider == "api" and provider == "local"
        logger.info(
            "[PIPELINE] Fallback triggered: %s (expected=%s, actual=%s)",
            fallback_triggered,
            expected_provider,
            provider,
        )

        # If provider differs from the prompt style assumption, only rebuild once when
        # this was not an API->local fallback. Re-running through the router after a
        # fallback would attempt the API a second time for the same user query.
        if provider != expected_provider and not fallback_triggered:
            prompt = self.prompt_builder.build_prompt(query, context, provider)
            answer, used_model, provider = self.generator.generate(prompt)
            logger.info(
                "[PIPELINE] Provider used after prompt adaptation: %s", provider
            )
        elif fallback_triggered:
            logger.info(
                "[PIPELINE] Skipping prompt adaptation rerun after API fallback to avoid duplicate provider attempts"
            )

        logger.info(
            "[PIPELINE] Generation completed stage_ms=%.1f provider=%s",
            (time.perf_counter() - generation_started) * 1000,
            provider,
        )
        validation = self.validator.validate(answer, citation_map, top_chunks)
        if validation["confidence"] < 0.2 and len(answer.split()) < 5:
            return {
                "answer": "I don't know",
                "context": context,  # 🔥 ADD THIS
                "citations": [],
                "confidence": validation["confidence"],
                "sources": [c.metadata for c in top_chunks],
                "reason": validation["reason"],
                "provider": provider,
                "fallback_triggered": fallback_triggered,
            }

        if run_tree:
            run_tree.metadata.update(
                {
                    "provider": provider,
                    "model": used_model,
                    "fallback_triggered": fallback_triggered,
                }
            )
        return {
            "answer": answer,
            "context": context,
            "citations": validation["citations"],
            "confidence": validation["confidence"],
            "sources": [c.metadata for c in top_chunks],
            "model": used_model,
            "provider": provider,
            "fallback_triggered": fallback_triggered,
        }

    @traceable(name="generation.stream", run_type="chain")
    def stream(
        self,
        query: str,
        original_query: Optional[str] = None,
        metadata_filters=None,
        retrieved=None,
    ):
        """Stream generated text, then yield the validated final result."""
        run_tree = get_current_run_tree()
        retrieval_started = time.perf_counter()
        if retrieved is None:
            retrieved = self.retriever.retrieve(
                query=query,
                original_query=original_query,
                top_k=40,
                metadata_filters=metadata_filters or {},
            )
        logger.info(
            "[PIPELINE] Retrieved chunks for stream: %s (stage_ms=%.1f)",
            len(retrieved),
            (time.perf_counter() - retrieval_started) * 1000,
        )

        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(query, retrieved)
        top_chunks = reranked[:5]
        logger.info(
            "[PIPELINE] Stream reranking completed stage_ms=%.1f top_chunks=%s",
            (time.perf_counter() - rerank_started) * 1000,
            len(top_chunks),
        )
        if run_tree:
            run_tree.metadata.update(
                {
                    "retrieved_count": len(retrieved),
                    "top_chunks_count": len(top_chunks),
                    "streaming": True,
                }
            )

        context_started = time.perf_counter()
        context, citation_map = self.context_builder.build(top_chunks)
        logger.info(
            "[PIPELINE] Stream context assembly completed stage_ms=%.1f context_chars=%s",
            (time.perf_counter() - context_started) * 1000,
            len(context),
        )
        context_assessment = self._assess_final_context(query, top_chunks, run_tree)
        if context_assessment is not None and not context_assessment.sufficient:
            rejected = self._insufficient_context_result(
                context, top_chunks, context_assessment
            )
            yield {"event": "chunk", "data": {"text": rejected["answer"]}}
            yield {"event": "final", "data": rejected}
            return

        llm_router = self.generator.llm_router
        expected_provider = "api" if llm_router.primary else "local"
        prompt = self.prompt_builder.build_prompt(query, context, expected_provider)

        generation_started = time.perf_counter()
        answer_parts = []
        used_model = None
        provider = None
        for text, used_model, provider in self.generator.stream(prompt):
            answer_parts.append(text)
            yield {"event": "chunk", "data": {"text": text}}

        answer = "".join(answer_parts)
        logger.info(
            "[PIPELINE] Stream generation completed stage_ms=%.1f provider=%s",
            (time.perf_counter() - generation_started) * 1000,
            provider,
        )
        validation = self.validator.validate(answer, citation_map, top_chunks)
        fallback_triggered = expected_provider == "api" and provider == "local"

        if run_tree:
            run_tree.metadata.update(
                {
                    "provider": provider,
                    "model": used_model,
                    "fallback_triggered": fallback_triggered,
                }
            )

        yield {
            "event": "final",
            "data": {
                "answer": answer,
                "context": context,
                "citations": validation["citations"],
                "confidence": validation["confidence"],
                "sources": [chunk.metadata for chunk in top_chunks],
                "model": used_model,
                "provider": provider,
                "fallback_triggered": fallback_triggered,
                "reason": validation["reason"],
            },
        }
