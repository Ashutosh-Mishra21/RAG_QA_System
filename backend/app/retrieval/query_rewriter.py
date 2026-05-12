import logging
import re
from typing import List

logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(self, llm_router):
        self.llm = llm_router

    # -------------------------
    # 🔹 Single Query Rewrite
    # -------------------------
    def rewrite(self, query: str) -> str:
        prompt = f"""
You are an expert in search systems.

Rewrite the user query into a more specific and retrieval-friendly query.

Rules:
- Keep it SHORT (max 1 sentence)
- Do NOT add new terms (like model names, examples)
- Do NOT expand the scope
- Only improve clarity
- Keep keywords close to original

USER QUERY:
{query}

REWRITTEN QUERY:
"""

        try:
            rewritten, _, _ = self.llm.generate(prompt)

            rewritten = rewritten.strip().replace("\n", " ")

            if len(rewritten) < 5:
                logger.warning("Query rewrite too short; using original query")
                return query

            logger.info("Query rewritten successfully")
            return rewritten

        except Exception:
            logger.exception("Query rewrite failed; using original query")
            return query

    # -------------------------
    # 🔥 Multi Query Generation
    # -------------------------
    def generate_multi_queries(self, query: str) -> List[str]:

        prompt = f"""
Generate 3 alternative search queries for the following question.

Rules:
- Preserve original meaning
- Use different wording
- Keep them concise
- Do NOT introduce new concepts
- Do NOT answer the question

Original query:
{query}

Return ONLY a list (one per line).
"""

        try:
            response, _, _ = self.llm.generate(prompt)

            # 🔥 Parse output safely
            queries = []

            for line in response.split("\n"):
                line = line.strip()

                # remove numbering (1., -, etc.)
                line = re.sub(r"^\d+[\).\s-]*", "", line)

                if len(line) > 5:
                    queries.append(line)

            # fallback safety
            if not queries:
                logger.warning("Multi-query generation returned no usable queries")
                return [query]

            # limit
            logger.info("Generated %s alternative search queries", len(queries[:3]))
            return queries[:3]

        except Exception:
            logger.exception("Multi-query generation failed; using original query")
            return [query]
