from backend.evaluation.evaluator import RetrievalEvaluator
from backend.app.retrieval.semantic_retriever import SemanticRetriever


def test_evaluator_runs():
    evaluator = RetrievalEvaluator(
        test_file="backend/evaluation/test_queries.json", top_k=5
    )
    result = evaluator.evaluate_retriever(SemanticRetriever())
    assert "Recall@K" in result
