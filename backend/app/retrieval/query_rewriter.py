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
Generate 3 alternative search queries for the following question. The queries
will be used to search a technical reference document.

Rules:
- Preserve original meaning
- Make the first a concise semantic paraphrase
- Make the second keyword-focused, using likely canonical technical terms or
  terminology a reference document may use for the requested concept
- Make the third a different concise formulation
- Do not introduce unrelated entities or details
- Do NOT answer the question
- Return three distinct queries, not minor rewordings of the same sentence

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

                # Ignore common model preambles so they do not consume one of
                # the three query slots and truncate a real search variant.
                if re.match(
                    r"^(?:here are\b|alternative search queries\b|search queries\s*:)",
                    line,
                    flags=re.IGNORECASE,
                ):
                    continue

                # remove numbering and list markers (1., -, *, etc.)
                line = re.sub(r"^(?:\d+[\).\s-]*|[-*•]\s*)", "", line)

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
