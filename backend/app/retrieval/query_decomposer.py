import logging
from typing import List

logger = logging.getLogger(__name__)


class QueryDecomposer:
    def __init__(self, llm_router):
        self.llm = llm_router

    def decompose(self, query: str) -> List[str]:
        prompt = f"""
            Break the following query into smaller sub-questions.

            Rules:
            - Each sub-question should be answerable independently
            - Do NOT add new information
            - Keep them simple and clear
            - If query is simple, return original query only

            Query:
            {query}

            Return each sub-question on a new line.
        """

        try:
            response, _, _ = self.llm.generate(prompt)

            subqueries = []

            for line in response.split("\n"):
                line = line.strip()
                if len(line) > 5:
                    subqueries.append(line)

            if not subqueries:
                logger.warning("Query decomposition returned no usable subqueries")
                return [query]

            logger.info(
                "Query decomposition produced %s subquery/subqueries",
                len(subqueries[:3]),
            )
            return subqueries[:3]

        except Exception:
            logger.exception("Query decomposition failed; using original query")
            return [query]
