import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GenerationPipeline:
    def __init__(
        self, retriever, reranker, context_builder, prompt_builder, generator, validator
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.validator = validator

    def run(self, query: str, metadata_filters=None) -> Dict[str, Any]:
        retrieved = self.retriever.retrieve(
            query=query, top_k=40, metadata_filters=metadata_filters or {}
        )
        print("Retrieved:", len(retrieved))

        reranked = self.reranker.rerank(query, retrieved)  # ✅ first define

        top_chunks = reranked[:5]  # ✅ then use it
        print("Top chunks:", len(top_chunks))

        context, citation_map = self.context_builder.build(top_chunks)

        llm = self.generator.llm_router
        logger.info("[PIPELINE] llm.primary is None: %s", llm.primary is None)

        expected_provider = "api" if llm.primary else "local"
        # first try with API-style (default)
        initial_prompt = self.prompt_builder.build_prompt(
            query, context, expected_provider
        )

        answer, used_model, provider = self.generator.generate(initial_prompt)
        logger.info("[PIPELINE] Provider used for initial generation: %s", provider)

        fallback_triggered = expected_provider == "api" and provider == "local"
        print(
            f"[PIPELINE] Provider result: expected={expected_provider}, actual={provider}, fallback_triggered={fallback_triggered}"
        )
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
