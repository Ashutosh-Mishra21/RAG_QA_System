class Generator:
    def __init__(self, llm_router):
        self.llm_router = llm_router

    def generate(self, prompt: str):
        answer, model, provider = self.llm_router.generate(prompt)

        if answer.strip().lower() == "i don't know":
            retry_prompt = (
                f"{prompt}\n\n"
                "Re-check the provided context carefully and answer with the most relevant facts "
                'from context with citations. Return "I don\'t know" only if context is truly unrelated.'
            )
            answer, model, provider = self.llm_router.generate(retry_prompt)

        return answer, model, provider

    def stream(self, prompt: str):
        """Yield text fragments and their provider metadata from the LLM router."""
        yield from self.llm_router.stream(prompt)
