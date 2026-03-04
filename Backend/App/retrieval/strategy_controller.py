from backend.app.retrieval.semantic_retriever import SemanticRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.query_analyzer import QueryAnalyzer


class AgenticRetriever:
    def __init__(self):
        self.retriever = SemanticRetriever()
        self.reranker = CrossEncoderReranker()
        self.analyzer = QueryAnalyzer()

    def retrieve(self, query: str):
        strategy = self.analyzer.analyze(query)

        results = self.retriever.retrieve(
            query=query,
            top_k=strategy.top_k,
            metadata_filters=strategy.metadata_filters,
        )

        if strategy.use_rerank:
            results = self.reranker.rerank(query, results)
            
        return {
            "strategy": strategy,
            "results": results
        }