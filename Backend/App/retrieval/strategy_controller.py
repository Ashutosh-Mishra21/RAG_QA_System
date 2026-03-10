from backend.app.retrieval.semantic_retriever import SemanticRetriever
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.query_analyzer import QueryAnalyzer


class AgenticRetriever:
    def __init__(self, documents):
        self.dense = SemanticRetriever()
        self.keyword = BM25Retriever(documents)
        self.hybrid = HybridRetriever(self.dense, self.keyword)

        self.reranker = CrossEncoderReranker()
        self.analyzer = QueryAnalyzer()

    def retrieve(self, query: str):

        strategy = self.analyzer.analyze(query)

        # --- Retrieval Mode Selection ---
        if strategy.query_type == "definition":
            results = self.hybrid.retrieve(query, top_k=strategy.top_k)

        elif strategy.query_type == "explanation":
            results = self.hybrid.retrieve(query, top_k=strategy.top_k)

        else:
            results = self.dense.retrieve(
                query=query,
                top_k=strategy.top_k,
                metadata_filters=strategy.metadata_filter,
                rerank=False,
            )

        # --- Optional Rerank ---
        if strategy.use_rerank:
            results = self.reranker.rerank(query, results)

        return {
            "strategy": strategy,
            "results": results,
        }
