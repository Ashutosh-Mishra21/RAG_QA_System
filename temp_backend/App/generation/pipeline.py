class GenerationPipeline:

    def __init__(
        self, retriever, context_builder, prompt_builder, generator, validator
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.validator = validator

    def run(self, query):

        retrieved = self.retriever.retrieve(query)

        context, citation_map = self.context_builder.build(retrieved)

        prompt = self.prompt_builder.build_prompt(query, context)

        answer = self.generator.generate(prompt)

        validation = self.validator.validate(answer, citation_map)

        if not validation["valid"]:
            return {"answer": "I don't know", "reason": validation["reason"]}

        confidence = sum(chunk["score"] for chunk in retrieved[:3]) / 3

        return {
            "answer": answer,
            "citations": validation["citations"],
            "context": context,
            "confidence": confidence,
        }
