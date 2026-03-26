class Generator:
    def __init__(self, llm_router):
        self.llm_router = llm_router

    def generate(self, prompt: str):
        answer, model = self.llm_router.generate(prompt)
        return answer, model
