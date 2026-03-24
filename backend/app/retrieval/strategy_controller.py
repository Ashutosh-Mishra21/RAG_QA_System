from backend.app.retrieval.semantic_retriever import SemanticRetriever
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.query_analyzer import QueryAnalyzer
from backend.app.indexing.keyword_index import KeywordIndex


class AgenticRetriever:
    def __init__(self, documents):
        self.dense = SemanticRetriever()
        keyword = KeywordIndex()
        keyword.add(documents)
        self.hybrid = HybridRetriever(self.dense, keyword)

        self.reranker = CrossEncoderReranker()
        self.analyzer = QueryAnalyzer()

    def retrieve(self, query: str):
        strategy = self.analyzer.analyze(query)

        if strategy.query_type in {"definition", "explanation"}:
            results = self.hybrid.retrieve(
                query,
                top_k=strategy.top_k,
                metadata_filters=strategy.metadata_filter,
            )
        else:
            results = self.dense.retrieve(
                query=query,
                top_k=strategy.top_k,
                metadata_filters=strategy.metadata_filter,
            )

        if strategy.use_rerank:
            results = self.reranker.rerank(query, results)

        return {
            "strategy": strategy,
            "results": results,
        }
