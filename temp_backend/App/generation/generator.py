class Generator:

    def __init__(self, llm):
        self.llm = llm

    def generate(self, prompt):

        response = self.llm.generate(prompt)

        return response
