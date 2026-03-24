from typing import Dict, List, Optional, Any
import numpy as np
from backend.app.models import Chunk


class HybridRetriever:
    def __init__(self, dense_retriever, keyword_index, temperature: float = 1.0):
        self.dense = dense_retriever
        self.keyword = keyword_index
        self.temperature = temperature

    def _softmax(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        arr = np.array(scores, dtype=float)
        arr = arr - np.max(arr)
        exp_scores = np.exp(arr / self.temperature)
        denom = np.sum(exp_scores)
        if denom == 0:
            return [0.0 for _ in scores]
        return (exp_scores / denom).tolist()

    def _importance_weight(self, metadata: Dict[str, Any]) -> float:
        hierarchy = metadata.get("hierarchy_path")
        if isinstance(hierarchy, str):
            depth = len([p for p in hierarchy.split(" > ") if p])
        elif isinstance(hierarchy, list):
            depth = len(hierarchy)
        else:
            depth = 0

        if depth == 0:
            return 1.2
        if depth >= 3:
            return 0.9
        return 1.0

    def _dynamic_weights(self, query: str) -> tuple[float, float]:
        q = query.lower().strip()
        tokens = q.split()
        if len(tokens) <= 3:
            return 0.4, 0.6
        if q.startswith("what is") or q.startswith("define"):
            return 0.65, 0.35
        if "explain" in q or "how" in q:
            return 0.6, 0.4
        return 0.6, 0.4

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        dense_results = self.dense.retrieve(
            query=query, top_k=top_k * 3, metadata_filters=metadata_filters
        )
        keyword_results = self.keyword.retrieve(
            query=query, top_k=top_k * 3, metadata_filters=metadata_filters
        )

        dense_weight, keyword_weight = self._dynamic_weights(query)
        dense_scores = self._softmax([c.score or 0.0 for c in dense_results])
        keyword_scores = self._softmax([c.score or 0.0 for c in keyword_results])

        merged: Dict[str, Dict[str, Any]] = {}

        for idx, chunk in enumerate(dense_results):
            merged[chunk.id] = {
                "chunk": chunk.model_copy(deep=True),
                "dense": dense_scores[idx] if idx < len(dense_scores) else 0.0,
                "keyword": 0.0,
            }

        for idx, chunk in enumerate(keyword_results):
            entry = merged.setdefault(
                chunk.id,
                {
                    "chunk": chunk.model_copy(deep=True),
                    "dense": 0.0,
                    "keyword": 0.0,
                },
            )
            entry["keyword"] = keyword_scores[idx] if idx < len(keyword_scores) else 0.0

        fused: List[Chunk] = []
        for entry in merged.values():
            chunk = entry["chunk"]
            fusion_score = (
                dense_weight * entry["dense"] + keyword_weight * entry["keyword"]
            )
            chunk.score = float(fusion_score * self._importance_weight(chunk.metadata))
            fused.append(chunk)

        fused.sort(key=lambda c: c.score or 0.0, reverse=True)
        return fused[:top_k]
