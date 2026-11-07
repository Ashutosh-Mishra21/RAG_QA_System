from backend.app.models import Chunk
from backend.app.retrieval import HybridRetriever
from backend.app.core.config import settings


def test_rrf_rewards_chunks_present_in_both_rankings():
    retriever = HybridRetriever(None, None)
    dense = [
        Chunk(id="dense-only", content="dense", metadata={}, score=0.9),
        Chunk(id="both", content="both", metadata={}, score=0.8),
    ]
    keyword = [
        Chunk(id="both", content="both", metadata={}, score=0.9),
        Chunk(id="keyword-only", content="keyword", metadata={}, score=0.8),
    ]

    results = retriever._reciprocal_rank_fusion(dense, keyword)

    assert [chunk.id for chunk in results][0] == "both"
    assert {chunk.id for chunk in results} == {
        "dense-only",
        "both",
        "keyword-only",
    }


def test_weighted_query_rrf_preserves_query_weights():
    retriever = HybridRetriever(None, None)
    strong = Chunk(id="strong", content="strong", metadata={})
    weak = Chunk(id="weak", content="weak", metadata={})

    results = retriever._weighted_query_rrf(
        dense_rankings=[(1.0, [strong]), (0.1, [weak])],
        keyword_rankings=[],
        dense_weight=1.0,
        keyword_weight=0.0,
    )

    assert [chunk.id for chunk in results] == ["strong", "weak"]


def test_mmr_uses_embeddings_to_select_diverse_results():
    retriever = HybridRetriever(None, None)
    chunks = [
        Chunk(id="a", content="a", metadata={}, score=1.0, embedding=[1.0, 0.0]),
        Chunk(id="b", content="b", metadata={}, score=0.99, embedding=[1.0, 0.0]),
        Chunk(id="c", content="c", metadata={}, score=0.8, embedding=[0.0, 1.0]),
    ]

    results = retriever._mmr(chunks, k=2, lambda_param=0.5)

    assert [chunk.id for chunk in results] == ["a", "c"]


def test_mmr_normalizes_rrf_relevance_and_keeps_failure_case_evidence(caplog):
    retriever = HybridRetriever(None, None)
    seed = Chunk(
        id="72123411-ee28-5e7f-85a2-890c1d0d3b33",
        content="Figure 1: The Transformer - model architecture.",
        metadata={},
        score=0.0669606,
        embedding=[1.0, 0.0, 0.0],
    )
    diverse_candidate = Chunk(
        id="96dab554-56b8-5690-bef1-8b4ec4209ff4",
        content="Evaluation details about translation results.",
        metadata={},
        score=0.0661327,
        embedding=[0.0, 1.0, 0.0],
    )
    # This is an answer-bearing chunk from the logged failure case. Its raw RRF
    # score is much smaller than cosine similarity, but it ranks above the final
    # filler candidate and should survive MMR after relevance normalization.
    answer_chunk = Chunk(
        id="8a8f89a3-52d2-527e-957c-60769c40ec5f",
        content=(
            "The dominant sequence transduction models include an encoder and a "
            "decoder. The Transformer is based solely on attention mechanisms."
        ),
        metadata={},
        score=0.0419111,
        embedding=[1.0, 0.0, 0.0],
    )
    filler = Chunk(
        id="filler",
        content="A low-ranked chunk unrelated to the architecture question.",
        metadata={},
        score=0.0200000,
        embedding=[0.0, 0.0, 1.0],
    )
    original_scores = {chunk.id: chunk.score for chunk in (seed, diverse_candidate, answer_chunk, filler)}

    # Reproduce the pre-patch MMR objective on the same candidates to assert
    # that this fixture represents the score-scale failure being fixed.
    baseline_selected = [seed]
    baseline_candidates = [diverse_candidate, answer_chunk, filler]
    while baseline_candidates and len(baseline_selected) < 3:
        best_index = max(
            range(len(baseline_candidates)),
            key=lambda i: (
                0.85 * (baseline_candidates[i].score or 0.0)
                - 0.15 * max(
                    retriever._similarity(baseline_candidates[i], chosen)
                    for chosen in baseline_selected
                ),
                tuple(-ord(char) for char in baseline_candidates[i].id),
            ),
        )
        baseline_selected.append(baseline_candidates.pop(best_index))
    assert answer_chunk.id not in {chunk.id for chunk in baseline_selected}

    with caplog.at_level("DEBUG", logger="backend.app.retrieval.hybrid_retriever"):
        selected = retriever._mmr(
            [filler, answer_chunk, diverse_candidate, seed],
            k=3,
            lambda_param=0.85,
        )

    assert answer_chunk.id in {chunk.id for chunk in selected}
    assert answer_chunk.id in {chunk.id for chunk in selected} - {chunk.id for chunk in baseline_selected}
    assert {chunk.id: chunk.score for chunk in (seed, diverse_candidate, answer_chunk, filler)} == original_scores
    assert "rrf_rank=3" in caplog.text
    assert "normalized_relevance=0.3333333" in caplog.text
    assert "max_cosine_similarity=1.0000000" in caplog.text
    assert "selected=True" in caplog.text


def test_rrf_and_mmr_break_ties_by_stable_chunk_id():
    retriever = HybridRetriever(None, None)
    left = Chunk(id="a", content="left", metadata={}, score=1.0, embedding=[1.0, 0.0])
    right = Chunk(id="b", content="right", metadata={}, score=1.0, embedding=[0.0, 1.0])

    fused = retriever._weighted_query_rrf(
        dense_rankings=[(1.0, [left])],
        keyword_rankings=[(1.0, [right])],
        dense_weight=0.5,
        keyword_weight=0.5,
    )
    mmr = retriever._mmr([right, left], k=2, lambda_param=0.5)

    assert [chunk.id for chunk in fused] == ["a", "b"]
    assert [chunk.id for chunk in mmr] == ["a", "b"]


def test_dense_and_bm25_first_pass_run_concurrently(monkeypatch):
    from threading import Barrier

    monkeypatch.setattr(settings, "RETRIEVAL_MAX_CONCURRENCY", 2)
    barrier = Barrier(2)

    class ConcurrentBackend:
        def __init__(self, chunk_id):
            self.chunk_id = chunk_id

        def retrieve(self, query, top_k=5, metadata_filters=None):
            barrier.wait(timeout=2)
            return [
                Chunk(
                    id=self.chunk_id,
                    content=f"evidence from {self.chunk_id}",
                    metadata={},
                    score=1.0,
                )
            ]

    retriever = HybridRetriever(
        ConcurrentBackend("dense"),
        ConcurrentBackend("bm25"),
    )
    results = retriever.retrieve("query", top_k=2)

    assert {chunk.id for chunk in results} == {"dense", "bm25"}


def test_planned_target_skips_additional_query_expansion():
    class UnexpectedRewriter:
        def generate_multi_queries(self, query):
            raise AssertionError("planned targets must not trigger nested expansions")

    class Backend:
        def __init__(self, chunk_id):
            self.chunk_id = chunk_id

        def retrieve(self, query, top_k=5, metadata_filters=None):
            return [
                Chunk(
                    id=self.chunk_id,
                    content=f"target evidence {self.chunk_id}",
                    metadata={},
                    score=1.0,
                )
            ]

    retriever = HybridRetriever(
        Backend("dense"),
        Backend("bm25"),
        query_rewriter=UnexpectedRewriter(),
    )
    results = retriever.retrieve_target("planned target", top_k=2)

    assert {chunk.id for chunk in results} == {"dense", "bm25"}


def test_hybrid_falls_back_to_sparse_when_dense_fails():
    class BrokenDense:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("unavailable")

    class Sparse:
        def retrieve(self, *args, **kwargs):
            return [Chunk(id="s", content="sparse result", metadata={}, score=1.0)]

    results = HybridRetriever(BrokenDense(), Sparse()).retrieve("query", top_k=1)

    assert [chunk.id for chunk in results] == ["s"]


def test_hybrid_rejects_unbounded_queries_and_invalid_filter_keys():
    retriever = HybridRetriever(None, None)

    try:
        retriever.retrieve("x" * (settings.RETRIEVAL_MAX_QUERY_CHARS + 1))
        assert False, "Expected oversized query to be rejected"
    except ValueError:
        pass

    try:
        retriever.retrieve("query", metadata_filters={"": "value"})
        assert False, "Expected invalid filter key to be rejected"
    except ValueError:
        pass
