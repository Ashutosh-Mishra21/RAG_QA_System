class GenerationPipeline:

    def __init__(
        self, retriever, context_builder, prompt_builder, generator, citation_manager
    ):

        self.retriever = retriever
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.citation_manager = citation_manager

    def run(self, query):

        retrieved = self.retriever.retrieve(query)

        context = self.context_builder.build(retrieved)

        prompt = self.prompt_builder.build_prompt(query, context)

        answer = self.generator.generate(prompt)

        citations = self.citation_manager.extract(answer)

        return {"answer": answer, "citations": citations, "context": context}
