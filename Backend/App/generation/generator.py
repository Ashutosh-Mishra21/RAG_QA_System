class Generator:

    def __init__(self, llm_client):
        self.llm = llm_client

    def generate(self, prompt):

        response = self.llm.generate(prompt)

        return response
