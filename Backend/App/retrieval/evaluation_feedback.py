def evaluate_strategy(agentic_retriever, test_queries):
    for q in test_queries:
        output = agentic_retriever.retrieve(q)
        # compute recall, mrr, ndcg
        # log strategy vs performance
