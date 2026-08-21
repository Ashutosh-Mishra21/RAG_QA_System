from backend.app.models import Chunk
from backend.app.retrieval import HybridRetriever


def test_rrf_rewards_chunks_present_in_both_rankings():
    retriever = HybridRetriever(None, None)
    dense = [
        Chunk(id="dense-only", content="dense", metadata={}),
        Chunk(id="both", content="both", metadata={}),
    ]
    keyword = [
        Chunk(id="both", content="both", metadata={}),
        Chunk(id="keyword-only", content="keyword", metadata={}),
    ]

    results = retriever._reciprocal_rank_fusion(dense, keyword)

    assert [chunk.id for chunk in results][0] == "both"
    assert {chunk.id for chunk in results} == {
        "dense-only",
        "both",
        "keyword-only",
    }
