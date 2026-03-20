def evaluate_strategy(agentic_retriever, test_queries, evaluator):

    for q in test_queries:

        output = agentic_retriever.retrieve(q)
        results = output["results"]

        metrics = evaluator.evaluate(q, results)

        print("Query:", q)
        print("Strategy:", output["strategy"])
        print(metrics)
