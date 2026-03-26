from typing import Dict, Any


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
        reranked = self.reranker.rerank(query, retrieved)
        top_chunks = reranked[:5]

        context, citation_map = self.context_builder.build(top_chunks)

        llm = self.generator.llm_router

        if llm.primary:
            model_name = llm.primary.model
        elif llm.fallback:
            model_name = llm.fallback.model
        else:
            model_name = "unknown"

        prompt = self.prompt_builder.build_prompt(query, context, model_name)

        answer, used_model = self.generator.generate(prompt)

        validation = self.validator.validate(answer, citation_map, top_chunks)
        if validation["confidence"] < 0.5:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": validation["confidence"],
                "sources": [c.metadata for c in top_chunks],
                "reason": validation["reason"],
            }

        return {
            "answer": answer,
            "citations": validation["citations"],
            "confidence": validation["confidence"],
            "sources": [c.metadata for c in top_chunks],
            "model": used_model,
        }
