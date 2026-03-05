from rank_bm25 import BM25Okapi
import numpy as np


class BM25Retriever:
    def __init__(self, documents):
        """
        documents = list of dicts
        {
            "id": "...",
            "text": "...",
            "metadata": {...}
        }
        """
        self.documents = documents
        self.corpus = [doc["text"] for doc in documents]
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query, top_k=5):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in ranked_idx:
            results.append(
                {
                    "id": self.documents[idx]["id"],
                    "text": self.documents[idx]["text"],
                    "metadata": self.documents[idx]["metadata"],
                    "bm25_score": float(scores[idx]),
                }
            )

        return results
