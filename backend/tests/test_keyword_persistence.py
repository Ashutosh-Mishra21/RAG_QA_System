from backend.app.indexing.keyword_index import KeywordIndex
from backend.app.models import Chunk


def test_bm25_index_survives_restart(tmp_path):
    index_path = tmp_path / "bm25.json"
    document = Chunk(
        id="doc-1",
        content="BGE embeddings and hybrid retrieval",
        metadata={"document_id": "doc-1"},
    )

    KeywordIndex(index_path).add([document])
    restarted = KeywordIndex(index_path)

    results = restarted.retrieve("hybrid retrieval", top_k=1)

    assert [chunk.id for chunk in results] == ["doc-1"]
