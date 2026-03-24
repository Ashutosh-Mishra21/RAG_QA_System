from pathlib import Path

from backend.app.models import Chunk
from backend.app.services import IngestionService


class FakeParser:
    def parse(self, file_path):
        return "doc"


class FakeBuilder:
    def build_tree(self, document):
        return ["root"]


class FakeChunker:
    def merge_chunks(self, node):
        return None


def test_ingest_and_index(monkeypatch, tmp_path):
    storage = tmp_path / "data"
    raw = storage / "raw"
    raw.mkdir(parents=True)
    file_path = raw / "doc1.md"
    file_path.write_text("content", encoding="utf-8")

    monkeypatch.setattr(
        "backend.app.services.ingestion_service.DoclingParser", lambda: FakeParser()
    )
    monkeypatch.setattr(
        "backend.app.services.ingestion_service.StructureBuilder", lambda: FakeBuilder()
    )
    monkeypatch.setattr(
        "backend.app.services.ingestion_service.NodeChunker", lambda: FakeChunker()
    )
    monkeypatch.setattr(
        "backend.app.services.ingestion_service.flatten_tree",
        lambda tree: [
            {
                "text": "chunk one",
                "heading_path": [{"heading": "Intro"}],
            }
        ],
    )

    class FakeEnricher:
        def enrich(self, chunk: Chunk) -> Chunk:
            return chunk

    class FakeEmbedder:
        def embed_texts(self, texts):
            return [[0.1, 0.2] for _ in texts]

    class FakeKeyword:
        def __init__(self):
            self.docs = []

        def add(self, docs):
            self.docs.extend(docs)

    class FakeRegistry:
        def __init__(self):
            self.keyword = FakeKeyword()

        def get_enricher(self):
            return FakeEnricher()

        def get_embedder(self):
            return FakeEmbedder()

        def get_keyword_index(self):
            return self.keyword

    class FakeStore:
        def upsert_chunks(self, chunks, embeddings):
            assert len(chunks) == len(embeddings) == 1

    monkeypatch.setattr(
        "backend.app.services.ingestion_service.ModelRegistry.instance",
        lambda: FakeRegistry(),
    )
    monkeypatch.setattr(
        "backend.app.services.ingestion_service.VectorStore", lambda: FakeStore()
    )

    service = IngestionService(storage)
    result = service.ingest_and_index(str(file_path))

    assert result["document_id"] == "doc1"
    assert result["chunks_indexed"] == 1
