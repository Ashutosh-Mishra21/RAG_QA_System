import numpy as np


class HybridRetriever:

    def __init__(self, dense_retriever, keyword_index, temperature: float = 1.0):
        self.dense = dense_retriever
        self.keyword = keyword_index

        self.dense_weight = 0.6
        self.keyword_weight = 0.4
        self.temperature = temperature

    # ---------- Softmax ----------
    def _softmax(self, scores):
        if not scores:
            return np.array([])

        scores = np.array(scores)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores / self.temperature)
        return exp_scores / np.sum(exp_scores)

    # ---------- Metadata Importance ----------
    def _importance_weight(self, metadata):

        hierarchy = metadata.get("hierarchy_path", [])

        depth = len(hierarchy)

        if depth == 0:
            return 1.3  # likely title-level
        elif depth == 1:
            return 1.2
        elif depth >= 3:
            return 0.9
        else:
            return 1.0

    # ---------- Dynamic Weight Selection ----------
    def _dynamic_weights(self, query):

        q = query.lower()
        tokens = q.split()

        # Keyword-heavy queries
        if len(tokens) <= 3:
            return 0.4, 0.6

        # Definition-style → dense slightly stronger
        if q.startswith("what is") or q.startswith("define"):
            return 0.65, 0.35

        # Explanation → balanced
        if "explain" in q or "how" in q:
            return 0.6, 0.4

        # Default
        return 0.6, 0.4

    # ---------- Retrieval ----------
    def retrieve(self, query, top_k=5):

        dense_results = self.dense.retrieve(query, top_k=top_k * 3)
        keyword_results = self.keyword.retrieve(query, top_k=top_k * 3)

        dense_weight, keyword_weight = self._dynamic_weights(query)

        score_dict = {}

        # --- Softmax normalize dense ---
        dense_scores = self._softmax([r["score"] for r in dense_results])

        for i, r in enumerate(dense_results):
            score_dict[r["id"]] = {
                "text": r["text"],
                "metadata": r["metadata"],
                "dense_score": dense_scores[i] if len(dense_scores) else 0,
                "keyword_score": 0,
            }

        # --- Softmax normalize keyword ---
        keyword_scores = self._softmax([r["score"] for r in keyword_results])

        for i, r in enumerate(keyword_results):
            if r["id"] not in score_dict:
                score_dict[r["id"]] = {
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "dense_score": 0,
                    "keyword_score": keyword_scores[i] if len(keyword_scores) else 0,
                }
            else:
                score_dict[r["id"]]["keyword_score"] = (
                    keyword_scores[i] if len(keyword_scores) else 0
                )

        # --- Fusion with Metadata Weight ---
        fused_results = []

        for doc_id, data in score_dict.items():

            fusion_score = (
                dense_weight * data["dense_score"]
                + keyword_weight * data["keyword_score"]
            )

            importance = self._importance_weight(data["metadata"])

            final_score = fusion_score * importance

            fused_results.append(
                {
                    "id": doc_id,
                    "text": data["text"],
                    "metadata": data["metadata"],
                    "score": float(final_score),
                }
            )

        fused_results.sort(key=lambda x: x["score"], reverse=True)

        return fused_results[:top_k]
