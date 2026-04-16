class QueryRewriter:
    def __init__(self, llm_router):
        self.llm = llm_router

    def rewrite(self, query: str) -> str:

        prompt = f"""
You are an expert in search systems.

Rewrite the user query into a more specific and retrieval-friendly query.

- Make it clear
- Add missing context if needed
- Keep the meaning same
- Do NOT answer the query

USER QUERY:
{query}

REWRITTEN QUERY:
"""

        try:
            rewritten, _, _ = self.llm.generate(prompt)

            # basic cleanup
            rewritten = rewritten.strip().replace("\n", " ")

            # fallback if bad output
            if len(rewritten) < 5:
                return query

            return rewritten

        except Exception:
            return query
