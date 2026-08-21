import logging
import math
from typing import Dict, List, Optional, Any
import numpy as np
from backend.app.models import Chunk
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        dense_retriever,
        keyword_index,
        query_rewriter=None,
        embedder=None,
        temperature: float = 1.0,
    ):
        self.dense = dense_retriever
        self.keyword = keyword_index
        self.query_rewriter = query_rewriter
        self.embedder = embedder
        self.temperature = temperature
        self.rrf_k = settings.RRF_K
        self.mmr_lambda = settings.MMR_LAMBDA

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
        if c1.embedding is None or c2.embedding is None:
            return 0.0

        v1 = np.array(c1.embedding)
        v2 = np.array(c2.embedding)

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0

        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _mmr(self, chunks: List[Chunk], k: int = 5, lambda_param: float | None = None):
        if not chunks:
            return []

        lambda_param = self.mmr_lambda if lambda_param is None else lambda_param
        lambda_param = min(1.0, max(0.0, lambda_param))

        selected = [chunks[0]]
        candidates = chunks[1:]

        while candidates and len(selected) < k:
            best_score = float("-inf")
            best_idx = 0

            for i, c in enumerate(candidates):
                relevance = c.score or 0.0

                diversity = max(
                    (self._similarity(c, s) for s in selected),
                    default=0.0,
                )

                score = lambda_param * relevance - (1 - lambda_param) * diversity

                if score > best_score:
                    best_score = score
                    best_idx = i

            selected.append(candidates.pop(best_idx))

        return selected

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[Chunk],
        keyword_results: list[Chunk],
        dense_weight: float = 0.6,
        keyword_weight: float = 0.4,
        k: int | None = None,
    ) -> list[Chunk]:
        k = self.rrf_k if k is None else max(1, k)
        dense_results = self._best_by_id(dense_results)
        keyword_results = self._best_by_id(keyword_results)
        merged: dict[str, dict] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            entry = merged.setdefault(
                chunk.id,
                {"chunk": chunk.model_copy(deep=True), "score": 0.0},
            )
            entry["score"] += dense_weight / (k + rank)

        for rank, chunk in enumerate(keyword_results, start=1):
            entry = merged.setdefault(
                chunk.id,
                {"chunk": chunk.model_copy(deep=True), "score": 0.0},
            )
            entry["score"] += keyword_weight / (k + rank)

        fused = []
        for entry in merged.values():
            chunk = entry["chunk"]
            chunk.score = entry["score"] * self._importance_weight(chunk.metadata)
            fused.append(chunk)

        return sorted(
            fused,
            key=lambda chunk: (-(chunk.score or 0.0), chunk.id),
        )

    def _weighted_query_rrf(
        self,
        dense_rankings: list[tuple[float, List[Chunk]]],
        keyword_rankings: list[tuple[float, List[Chunk]]],
        dense_weight: float,
        keyword_weight: float,
    ) -> list[Chunk]:
        merged: dict[str, dict[str, Any]] = {}

        for modality_weight, rankings in (
            (dense_weight, dense_rankings),
            (keyword_weight, keyword_rankings),
        ):
            for query_weight, candidates in rankings:
                for rank, chunk in enumerate(candidates, start=1):
                    entry = merged.setdefault(
                        chunk.id,
                        {"chunk": chunk.model_copy(deep=True), "score": 0.0},
                    )
                    entry["score"] += (
                        modality_weight * query_weight / (self.rrf_k + rank)
                    )

        fused = []
        for entry in merged.values():
            chunk = entry["chunk"]
            chunk.score = float(
                entry["score"] * self._importance_weight(chunk.metadata)
            )
            fused.append(chunk)
        return sorted(fused, key=lambda chunk: (-(chunk.score or 0.0), chunk.id))

    @staticmethod
    def _valid_candidates(results: List[Chunk]) -> List[Chunk]:
        valid = []
        for chunk in results:
            if (
                not isinstance(chunk, Chunk)
                or not chunk.id
                or not chunk.content.strip()
            ):
                continue
            if chunk.score is None or not math.isfinite(float(chunk.score)):
                continue
            valid.append(chunk)
        return valid

    @staticmethod
    def _best_by_id(results: List[Chunk]) -> List[Chunk]:
        best: dict[str, Chunk] = {}
        for chunk in HybridRetriever._valid_candidates(results):
            current = best.get(chunk.id)
            if current is None or (chunk.score, chunk.id) > (current.score, current.id):
                best[chunk.id] = chunk
        unique_content: dict[str, Chunk] = {}
        for chunk in sorted(best.values(), key=lambda c: (-(c.score or 0.0), c.id)):
            content_key = " ".join(chunk.content.casefold().split())
            unique_content.setdefault(content_key, chunk)
        return list(unique_content.values())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        original_query: Optional[str] = None,
    ) -> List[Chunk]:
        if not isinstance(query, str) or not query.strip():
            return []
        if len(query) > settings.RETRIEVAL_MAX_QUERY_CHARS:
            raise ValueError("Query exceeds the maximum supported length")
        if metadata_filters and any(
            not isinstance(key, str) or not key or len(key) > 100
            for key in metadata_filters
        ):
            raise ValueError("Invalid metadata filter")

        # =========================
        # 🔥 STEP 1: Multi-query
        # =========================
        top_k = min(max(1, top_k), settings.RETRIEVAL_MAX_TOP_K)
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

        dense_rankings: list[tuple[float, List[Chunk]]] = []
        keyword_rankings: list[tuple[float, List[Chunk]]] = []

        per_query_k = max(6, top_k * 2)

        for i, q in enumerate(queries):
            weight = query_weights[i]

            try:
                dense = self.dense.retrieve(
                    q, top_k=per_query_k, metadata_filters=metadata_filters
                )
            except Exception:
                logger.error("Dense retrieval failed for query variant", exc_info=True)
                dense = []
            dense_rankings.append((weight, self._best_by_id(dense)))

            try:
                keyword = self.keyword.retrieve(
                    q, top_k=per_query_k, metadata_filters=metadata_filters
                )
            except Exception:
                logger.exception("Sparse retrieval failed for query variant")
                keyword = []
            keyword_rankings.append((weight, self._best_by_id(keyword)))

        # =========================
        # 🔥 STEP 3: Dynamic weighting
        # =========================
        dense_weight, keyword_weight = self._dynamic_weights(query)

        # =========================
        # 🔥 STEP 4: Weighted query-level Reciprocal Rank Fusion
        # =========================
        fused = self._weighted_query_rrf(
            dense_rankings=dense_rankings,
            keyword_rankings=keyword_rankings,
            dense_weight=dense_weight,
            keyword_weight=keyword_weight,
        )

        # =========================
        # 🔥 STEP 5: Sort + return
        # =========================

        fused.sort(key=lambda c: (-(c.score or 0.0), c.id))
        candidates = fused[: top_k * 3]  # restrict pool

        if self.embedder is not None:
            missing = [chunk for chunk in candidates if chunk.embedding is None]
            if missing:
                vectors = self.embedder.embed_documents(
                    [chunk.content for chunk in missing]
                )
                for chunk, vector in zip(missing, vectors):
                    chunk.embedding = vector

        return self._mmr(candidates, k=top_k)
