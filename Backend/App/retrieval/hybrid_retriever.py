from rank_bm25 import BM25Okapi
import numpy as np

from app.retrieval.semantic_retriever import SemanticRetriever


class HybridRetriever:

    def __init__(self):
        self.dense = SemanticRetriever()

        self.dense_weight = 0.6
        self.bm25_weight = 0.4

    def retrieve(self, query, top_k=5, candidate_k=40):

        # Step 1: dense retrieval from Qdrant
        dense_results = self.dense.retrieve(query, top_k=candidate_k)

        corpus = [r["text"] for r in dense_results]
        tokenized_corpus = [doc.split() for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.split()

        bm25_scores = bm25.get_scores(tokenized_query)

        dense_scores = np.array([r["score"] for r in dense_results])

        if dense_scores.max() > 0:
            dense_scores = dense_scores / dense_scores.max()

        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        fused_results = []

        for i, r in enumerate(dense_results):

            final_score = (
                self.dense_weight * dense_scores[i] + self.bm25_weight * bm25_scores[i]
            )

            fused_results.append(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "score": final_score,
                }
            )

        fused_results.sort(key=lambda x: x["score"], reverse=True)

        return fused_results[:top_k]
