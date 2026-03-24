from backend.evaluation import RetrievalEvaluator
from backend.app.models import Chunk


class FakeRetriever:
    def retrieve(self, query, top_k=5):
        return [
            Chunk(
                id="1",
                content="x",
                metadata={"document_id": "embedding_doc"},
                score=0.9,
            )
        ]


def test_evaluator_runs():
    evaluator = RetrievalEvaluator(
        test_file="backend/evaluation/test_queries.json", top_k=5
    )
    result = evaluator.evaluate_retriever(FakeRetriever())
    assert "Recall@K" in result
