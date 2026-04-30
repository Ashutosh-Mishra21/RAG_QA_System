from typing import Dict, List, Optional, Any
import numpy as np
from backend.app.models import Chunk


class HybridRetriever:
    def __init__(
        self,
        dense_retriever,
        keyword_index,
        query_rewriter=None,
        temperature: float = 1.0,
    ):
        self.dense = dense_retriever
        self.keyword = keyword_index
        self.query_rewriter = query_rewriter
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

    def _similarity(self, c1, c2):
        if not hasattr(c1, "embedding") or not hasattr(c2, "embedding"):
            return 0.0

        v1 = np.array(c1.embedding)
        v2 = np.array(c2.embedding)

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0

        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _mmr(self, chunks: List[Chunk], k: int = 5, lambda_param: float = 0.85):
        if not chunks:
            return []

        if lambda_param is None:
            if len(chunks) < 10:
                lambda_param = 0.9  # fewer chunks → prioritize relevance
            else:
                lambda_param = 0.8  # more chunks → allow diversity

        selected = [chunks[0]]
        candidates = chunks[1:]

        while candidates and len(selected) < k:
            best_score = -1
            best_idx = 0

            for i, c in enumerate(candidates):
                relevance = c.score or 0.0

                diversity = (
                    max([self._similarity(c, s) for s in selected]) if selected else 0
                )

                score = lambda_param * relevance - (1 - lambda_param) * diversity

                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(candidates.pop(best_idx))

        return selected

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        original_query: Optional[str] = None,
    ) -> List[Chunk]:

        # =========================
        # 🔥 STEP 1: Multi-query
        # =========================
        queries = []

        if original_query:
            queries.append(original_query)

        queries.append(query)  # rewritten

        if self.query_rewriter:
            multi_queries = self.query_rewriter.generate_multi_queries(
                original_query or query
            )
            queries.extend(multi_queries)

        # dedupe queries
        queries = list(dict.fromkeys(queries))

        base = []
        rest = []

        for q in queries:
            if q == original_query or q == query:
                base.append(q)
            else:
                rest.append(q)

        queries = base + rest[:3]

        # =========================
        # 🔥 STEP 2: Retrieve
        # =========================
        query_weights = []

        for q in queries:
            if original_query and q == original_query:
                query_weights.append(1.0)
            elif q == query:
                query_weights.append(0.95)
            else:
                query_weights.append(0.85)

        all_dense_results = []
        all_keyword_results = []

        per_query_k = max(6, top_k * 2)

        for i, q in enumerate(queries):
            weight = query_weights[i]

            dense = self.dense.retrieve(
                q, top_k=per_query_k, metadata_filters=metadata_filters
            )
            for d in dense:
                d = d.model_copy(deep=True)
                d.score = (d.score or 0.0) * weight
                all_dense_results.append(d)

            keyword = self.keyword.retrieve(
                q, top_k=per_query_k, metadata_filters=metadata_filters
            )
            for k in keyword:
                k = k.model_copy(deep=True)
                k.score = (k.score or 0.0) * weight
                all_keyword_results.append(k)

        # =========================
        # 🔥 STEP 3: Deduplicate
        # =========================
        def deduplicate(results):
            seen = set()
            unique = []

            for r in results:
                key = (r.metadata.get("document_id"), getattr(r, "id", None))

                if key not in seen:
                    seen.add(key)
                    unique.append(r)

            return unique

        dense_results = deduplicate(all_dense_results)
        keyword_results = deduplicate(all_keyword_results)

        # =========================
        # 🔥 STEP 4: Normalize scores
        # =========================
        dense_scores = self._softmax([c.score or 0.0 for c in dense_results])
        keyword_scores = self._softmax([c.score or 0.0 for c in keyword_results])

        # =========================
        # 🔥 STEP 5: Dynamic weighting
        # =========================
        dense_weight, keyword_weight = self._dynamic_weights(query)

        # =========================
        # 🔥 STEP 6: Fusion
        # =========================
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

        # =========================
        # 🔥 STEP 7: Final scoring
        # =========================
        fused: List[Chunk] = []

        for entry in merged.values():
            chunk = entry["chunk"]

            fusion_score = (
                dense_weight * entry["dense"] + keyword_weight * entry["keyword"]
            )

            chunk.score = float(fusion_score * self._importance_weight(chunk.metadata))

            fused.append(chunk)

        # =========================
        # 🔥 STEP 8: Sort + return
        # =========================

        fused.sort(key=lambda c: c.score or 0.0, reverse=True)
        candidates = fused[: top_k * 3]  # restrict pool

        return self._mmr(candidates, k=top_k)
