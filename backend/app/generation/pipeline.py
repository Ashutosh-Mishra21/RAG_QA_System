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
        prompt = self.prompt_builder.build_prompt(query, context)
        answer = self.generator.generate(prompt)

        validation = self.validator.validate(answer, citation_map, top_chunks)
        if not validation["valid"]:
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
        }
