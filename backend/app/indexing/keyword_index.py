from typing import List, Optional, Dict, Any
from rank_bm25 import BM25Okapi
from backend.app.models import Chunk


class KeywordIndex:
    def __init__(self):
        self.documents: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None

    def _rebuild(self) -> None:
        tokenized_corpus = [doc.content.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def add(self, documents: List[Chunk]) -> None:
        if not documents:
            return
        self.documents.extend(documents)
        self._rebuild()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        if not self.bm25:
            return []

        candidates = self.documents
        if metadata_filters:
            candidates = [
                d
                for d in self.documents
                if all(d.metadata.get(k) == v for k, v in metadata_filters.items())
            ]
            if not candidates:
                return []
            tokenized_corpus = [doc.content.lower().split() for doc in candidates]
            bm25 = BM25Okapi(tokenized_corpus)
        else:
            bm25 = self.bm25

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: float(scores[i]), reverse=True
        )[:top_k]

        results: List[Chunk] = []
        for idx in ranked_indices:
            chunk = candidates[idx].model_copy(deep=True)
            chunk.score = float(scores[idx])
            results.append(chunk)
        return results
