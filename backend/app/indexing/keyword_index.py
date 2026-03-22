from typing import List
from rank_bm25 import BM25Okapi
from backend.app.models import Chunk


class KeywordIndex:
    def __init__(self):
        self.documents: List[Chunk] = []
        self.bm25 = None

    def add(self, documents: List[Chunk]) -> None:
        if not documents:
            return
        self.documents.extend(documents)
        tokenized_corpus = [doc.content.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: List[Chunk] = []
        for idx in ranked_indices:
            chunk = self.documents[idx].model_copy(deep=True)
            chunk.score = float(scores[idx])
            results.append(chunk)

        return results
