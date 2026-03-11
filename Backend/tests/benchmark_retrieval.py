from backend.app.retrieval.semantic_retriever import SemanticRetriever
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.strategy_controller import AgenticRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.evaluation.evaluator import RetrievalEvaluator
from backend.app.indexing.vector_store import VectorStore


def run_benchmark():

    evaluator = RetrievalEvaluator(
        test_file="backend/evaluation/test_queries.json", top_k=5
    )

    dense = SemanticRetriever()
    reranker = CrossEncoderReranker()

    # Build keyword corpus from vector store
    vector_store = VectorStore()
    documents = vector_store.fetch_all_documents()  # You must implement this
    keyword = BM25Retriever(documents)

    hybrid = HybridRetriever(dense, keyword)
    agentic = AgenticRetriever(documents)

    print("=" * 60)
    print("Dense")
    print(evaluator.evaluate_retriever(dense))

    print("=" * 60)
    print("Dense + Rerank")
    print(evaluator.evaluate_retriever(dense, reranker=reranker))

    print("=" * 60)
    print("Keyword")
    print(evaluator.evaluate_retriever(keyword))

    print("=" * 60)
    print("Hybrid (Metadata + Dynamic)")
    print(evaluator.evaluate_retriever(hybrid))

    print("=" * 60)
    print("Hybrid + Rerank")
    print(evaluator.evaluate_retriever(hybrid, reranker=reranker))

    print("=" * 60)
    print("Agentic Hybrid")
    print(evaluator.evaluate_agentic(agentic))


if __name__ == "__main__":
    run_benchmark()
