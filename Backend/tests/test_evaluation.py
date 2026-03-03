from backend.evaluation.evaluator import RetrievalEvaluator

evaluator = RetrievalEvaluator(
    test_file="backend/evaluation/test_queries.json", top_k=5
)

print("Dense Only:")
print(evaluator.evaluate(rerank=False))

print("\nDense + Reranker:")
print(evaluator.evaluate(rerank=True))
