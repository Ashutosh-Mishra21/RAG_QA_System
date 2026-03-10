import numpy as np


class HybridRetriever:

    def __init__(self, dense_retriever, keyword_index):
        self.dense = dense_retriever
        self.keyword = keyword_index

        self.dense_weight = 0.6
        self.keyword_weight = 0.4

    def retrieve(self, query, top_k=5):

        dense_results = self.dense.retrieve(query, top_k=top_k * 3)
        keyword_results = self.keyword.search(query, top_k=top_k * 3)

        score_dict = {}

        # Normalize dense
        dense_scores = np.array([r["score"] for r in dense_results])
        if dense_scores.size > 0 and dense_scores.max() > 0:
            dense_scores = np.exp(dense_scores)
            dense_scores = dense_scores / dense_scores.sum()

        for i, r in enumerate(dense_results):
            score_dict[r["id"]] = {
                "text": r["text"],
                "metadata": r["metadata"],
                "dense_score": dense_scores[i] if dense_scores.size > 0 else 0,
                "keyword_score": 0,
            }

        # Normalize keyword
        keyword_scores = np.array([r["score"] for r in keyword_results])
        if keyword_scores.size > 0 and keyword_scores.max() > 0:
            keyword_scores = keyword_scores / keyword_scores.max()

        for i, r in enumerate(keyword_results):
            if r["id"] not in score_dict:
                score_dict[r["id"]] = {
                    "text": r["content"],
                    "metadata": r["metadata"],
                    "dense_score": 0,
                    "keyword_score": (
                        keyword_scores[i] if keyword_scores.size > 0 else 0
                    ),
                }
            else:
                score_dict[r["id"]]["keyword_score"] = (
                    keyword_scores[i] if keyword_scores.size > 0 else 0
                )

        fused_results = []

        for doc_id, data in score_dict.items():

            final_score = (
                self.dense_weight * data["dense_score"]
                + self.keyword_weight * data["keyword_score"]
            )

            fused_results.append(
                {
                    "id": doc_id,
                    "text": data["text"],
                    "metadata": data["metadata"],
                    "score": final_score,
                }
            )

        fused_results.sort(key=lambda x: x["score"], reverse=True)

        return fused_results[:top_k]
