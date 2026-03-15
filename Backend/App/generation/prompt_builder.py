class PromptBuilder:

    def build_prompt(self, query, context):

        system = """
You are a helpful AI assistant that answers questions using the provided context.

Rules:
- Only answer using the context
- If the answer cannot be found, say "I don't know"
- Cite sources using [number]
"""

        prompt = f"""
SYSTEM:
{system}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

        return prompt
