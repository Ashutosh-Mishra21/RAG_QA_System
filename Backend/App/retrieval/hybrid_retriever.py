import numpy as np
from backend.app.retrieval.semantic_retriever import SemanticRetriever
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.reranker import CrossEncoderReranker


class HybridRetriever:

    def __init__(self, documents):

        self.dense_retriever = SemanticRetriever()
        self.bm25_retriever = BM25Retriever(documents)
        self.reranker = CrossEncoderReranker()
        # fusion weights
        self.rrf_k = 60

    def reciprocal_rank_fusion(self, dense_results, bm25_results):

        score_dict = {}

        # dense ranks
        for rank, r in enumerate(dense_results):
            score = 1 / (self.rrf_k + rank + 1)

            if r["id"] not in score_dict:
                score_dict[r["id"]] = {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "score": 0,
                }

            score_dict[r["id"]]["score"] += score

        # bm25 ranks
        for rank, r in enumerate(bm25_results):
            score = 1 / (self.rrf_k + rank + 1)

            if r["id"] not in score_dict:
                score_dict[r["id"]] = {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "score": 0,
                }

            score_dict[r["id"]]["score"] += score

        return list(score_dict.values())

    def retrieve(self, query, top_k=5):

        dense_results = self.dense_retriever.retrieve(
            query, top_k=top_k * 5, rerank=False
        )
        bm25_results = self.bm25_retriever.retrieve(
            query, top_k=top_k * 5, rerank=False
        )

        fused = self.reciprocal_rank_fusion(dense_results, bm25_results)

        fused.sort(key=lambda x: x["score"], reverse=True)

        top_candidates = fused[: top_k * 4]

        reranked = self.reranker.rerank(query, top_candidates)

        return reranked[:top_k]
