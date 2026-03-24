from typing import List
import torch
from sentence_transformers import CrossEncoder
from backend.app.models import Chunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, retrieved_chunks: List[Chunk]) -> List[Chunk]:
        if not retrieved_chunks:
            return []

        pairs = [(query, chunk.content) for chunk in retrieved_chunks]
        scores = self.model.predict(pairs)

        reranked: List[Chunk] = []
        for chunk, score in zip(retrieved_chunks, scores):
            c = chunk.model_copy(deep=True)
            c.score = float(score)
            reranked.append(c)

        return sorted(reranked, key=lambda x: x.score or 0.0, reverse=True)
