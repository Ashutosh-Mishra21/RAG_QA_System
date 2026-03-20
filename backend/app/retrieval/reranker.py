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

        # Use standardized key: "text"
        pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]

        scores = self.model.predict(pairs)

        for chunk, score in zip(retrieved_chunks, scores):
            chunk["score"] = float(score)

        return sorted(
            retrieved_chunks,
            key=lambda x: x["score"],
            reverse=True,
        )
