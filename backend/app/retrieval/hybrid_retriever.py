import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
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
        # This executor is shared by requests handled by this retriever instance;
        # total in-flight dense/BM25 calls stay bounded across those requests.
        self._retrieval_executor = ThreadPoolExecutor(
            max_workers=max(1, settings.RETRIEVAL_MAX_CONCURRENCY),
            thread_name_prefix="rag-retrieval",
        )

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

        ordered = sorted(chunks, key=lambda chunk: (-(chunk.score or 0.0), chunk.id))
        # RRF scores are small rank-fusion values, whereas cosine similarity is
        # approximately in [-1, 1]. Use fused rank for MMR relevance so the two
        # terms in the MMR objective have comparable ranges. Keep chunk.score
        # untouched: it remains the original RRF score for diagnostics/downstream.
        last_rank = max(len(ordered) - 1, 1)
        relevance_by_id = {
            chunk.id: 1.0 - (rank / last_rank)
            for rank, chunk in enumerate(ordered)
        }
        selected = [ordered[0]]
        candidates = ordered[1:]

        logger.debug(
            "MMR decision=1 candidate_id=%s rrf_rank=1 rrf_score=%.7f "
            "normalized_relevance=%.7f max_cosine_similarity=None mmr_score=None selected=True",
            ordered[0].id,
            ordered[0].score or 0.0,
            relevance_by_id[ordered[0].id],
        )
        decision = 2
        rrf_rank_by_id = {chunk.id: rank for rank, chunk in enumerate(ordered, start=1)}
        while candidates and len(selected) < k:
            best_score = float("-inf")
            best_idx = 0
            decision_rows = []

            for i, c in enumerate(candidates):
                relevance = relevance_by_id[c.id]

                diversity = max(
                    (self._similarity(c, s) for s in selected),
                    default=0.0,
                )

                score = lambda_param * relevance - (1 - lambda_param) * diversity
                decision_rows.append((c, relevance, diversity, score))

                if score > best_score or (
                    score == best_score and c.id < candidates[best_idx].id
                ):
                    best_score = score
                    best_idx = i

            selected_id = candidates[best_idx].id
            for chunk, relevance, diversity, score in decision_rows:
                logger.debug(
                    "MMR decision=%d candidate_id=%s rrf_rank=%d rrf_score=%.7f "
                    "normalized_relevance=%.7f max_cosine_similarity=%.7f "
                    "mmr_score=%.7f selected=%s",
                    decision,
                    chunk.id,
                    rrf_rank_by_id[chunk.id],
                    chunk.score or 0.0,
                    relevance,
                    diversity,
                    score,
                    chunk.id == selected_id,
                )
            selected.append(candidates.pop(best_idx))
            decision += 1

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

    def retrieve_target(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Retrieve one planned query without spawning additional rewrites."""
        return self.retrieve(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            expand_queries=False,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        original_query: Optional[str] = None,
        expand_queries: bool = True,
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

        if expand_queries and self.query_rewriter:
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

        dense_results: list[list[Chunk]] = [[] for _ in queries]
        keyword_results: list[list[Chunk]] = [[] for _ in queries]
        per_query_k = max(6, top_k * 2)
        started = time.perf_counter()
        futures = {}

        def run_backend(backend, backend_name: str, query_index: int, text: str):
            task_started = time.perf_counter()
            try:
                results = backend.retrieve(
                    text, top_k=per_query_k, metadata_filters=metadata_filters
                )
                error = None
            except Exception as exc:
                logger.warning(
                    "Retrieval backend failed backend=%s query_index=%s error=%s",
                    backend_name,
                    query_index,
                    type(exc).__name__,
                    exc_info=True,
                )
                results = []
                error = type(exc).__name__
            duration_ms = (time.perf_counter() - task_started) * 1000
            logger.info(
                "Retrieval task backend=%s query_index=%s duration_ms=%.1f "
                "result_count=%s error=%s",
                backend_name,
                query_index,
                duration_ms,
                len(results),
                error or "none",
            )
            return results

        # Query planning/rewriting is complete before this point. Dense and BM25
        # calls are independent, so submit them together to a shared bounded pool.
        for i, text in enumerate(queries):
            if self.dense is not None:
                futures[self._retrieval_executor.submit(
                    run_backend, self.dense, "dense", i, text
                )] = ("dense", i)
            if self.keyword is not None:
                futures[self._retrieval_executor.submit(
                    run_backend, self.keyword, "bm25", i, text
                )] = ("bm25", i)

        timeout_seconds = max(0.1, settings.RETRIEVAL_TASK_TIMEOUT_SECONDS)
        completed = set()
        try:
            for future in as_completed(futures, timeout=timeout_seconds):
                completed.add(future)
                backend_name, query_index = futures[future]
                try:
                    results = future.result()
                except Exception:
                    logger.exception(
                        "Unexpected retrieval task failure backend=%s query_index=%s",
                        backend_name,
                        query_index,
                    )
                    results = []
                if backend_name == "dense":
                    dense_results[query_index] = self._best_by_id(results)
                else:
                    keyword_results[query_index] = self._best_by_id(results)
        except FuturesTimeoutError:
            for future, (backend_name, query_index) in futures.items():
                if future not in completed:
                    future.cancel()
                    logger.warning(
                        "Retrieval task timed out backend=%s query_index=%s timeout_seconds=%.1f",
                        backend_name,
                        query_index,
                        timeout_seconds,
                    )

        dense_rankings = [
            (weight, results)
            for weight, results in zip(query_weights, dense_results)
        ]
        keyword_rankings = [
            (weight, results)
            for weight, results in zip(query_weights, keyword_results)
        ]
        logger.info(
            "Hybrid retrieval completed query_count=%s submitted_tasks=%s "
            "completed_tasks=%s concurrency_limit=%s duration_ms=%.1f",
            len(queries),
            len(futures),
            len(completed),
            max(1, settings.RETRIEVAL_MAX_CONCURRENCY),
            (time.perf_counter() - started) * 1000,
        )

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
