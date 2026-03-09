from typing import Dict, List
from rank_bm25 import BM25Okapi


class KeywordIndex:
    def __init__(self):
        self.documents = List[Dict[str, object]] = []
        self.bm25 = None
        self.tokenized_corpus = List[List[str]] = []

    def add(self, documents: List[Dict[str, object]]) -> None:
        """
        documents: List of dicts with keys:
            - id
            - content
            - metadata
        """
        if not documents:
            return
        self.documents.extend(documents)
        self.tokenized_corpus = [
            doc["content"].lower().split() for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []

        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append(
                {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "score": float(scores[idx]),
                }
            )

        return results
