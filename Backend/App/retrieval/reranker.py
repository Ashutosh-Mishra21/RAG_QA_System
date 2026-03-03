import decimal
from turtle import mode

from sentence_transformers import CrossEncoder
import torch
from typing import List, Dict


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, retrieved_chunks: List[Dict]) -> List[Dict]:
        if not retrieved_chunks:
            return []

        # Prepare query-chunk pairs
        pairs = [(query, chunk["content"]) for chunk in retrieved_chunks]

        # Get relevance cross encode scores for each pair
        scores = self.model.predict(pairs)

        # Attach scores to chunks and sort by score
        for chunk, score in zip(retrieved_chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(
            retrieved_chunks,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked
