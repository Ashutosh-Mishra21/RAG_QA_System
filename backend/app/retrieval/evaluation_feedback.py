import logging

logger = logging.getLogger(__name__)


def evaluate_strategy(agentic_retriever, test_queries, evaluator):

    for q in test_queries:

        output = agentic_retriever.retrieve(q)
        results = output["results"]

        metrics = evaluator.evaluate(q, results)

        logger.info(
            "Evaluation feedback | query=%s | strategy=%s | metrics=%s",
            q,
            output["strategy"],
            metrics,
        )
