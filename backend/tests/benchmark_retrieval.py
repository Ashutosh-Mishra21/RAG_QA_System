from backend.app.retrieval import (
    HybridRetriever,
    SemanticRetriever,
    CrossEncoderReranker,
)
from backend.app.indexing import KeywordIndex
from backend.app.models import Chunk
from backend.evaluation import RetrievalEvaluator


def run_benchmark():
    evaluator = RetrievalEvaluator(
        test_file="backend/evaluation/test_queries.json", top_k=5
    )

    dense = SemanticRetriever()
    reranker = CrossEncoderReranker()

    keyword = KeywordIndex()
    keyword.add(
        [
            Chunk(
                id="1",
                content="embedding models",
                metadata={"document_id": "embedding_doc"},
            ),
            Chunk(
                id="2",
                content="hierarchical chunking",
                metadata={"document_id": "chunking_doc"},
            ),
        ]
    )

    hybrid = HybridRetriever(dense, keyword)

    print("=" * 60)
    print("Hybrid")
    print(evaluator.evaluate_retriever(hybrid, reranker=reranker))


if __name__ == "__main__":
    run_benchmark()
