import uuid
from pathlib import Path
from datetime import datetime
from typing import List
from backend.app.core.model_registry import ModelRegistry
from backend.app.ingestion import (
    DoclingParser,
    StructureBuilder,
    NodeChunker,
    flatten_tree,
)
from backend.app.indexing import VectorStore
from backend.app.models import Chunk, ChunkMetadata


class IngestionService:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.raw_dir = self.storage_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, filename: str, file_bytes: bytes):
        document_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        extension = Path(filename).suffix
        new_filename = f"{document_id}{extension}"
        file_path = self.raw_dir / new_filename

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        return {
            "document_id": document_id,
            "filename": filename,
            "stored_as": new_filename,
            "file_path": str(file_path),
            "timestamp": timestamp,
        }

    def ingest_and_index(self, file_path: str) -> dict:
        fp = Path(file_path)
        document_id = fp.stem

        parser = DoclingParser()
        builder = StructureBuilder()
        chunker = NodeChunker()

        document = parser.parse(fp)
        tree = builder.build_tree(document)
        for root in tree:
            chunker.merge_chunks(root)

        flat_chunks = flatten_tree(tree)

        chunks: List[Chunk] = []
        for idx, flat in enumerate(flat_chunks):
            heading_path = flat.get("heading_path", [])
            hierarchy_path = [h.get("heading", "") for h in heading_path]
            section = hierarchy_path[-1] if hierarchy_path else None
            title = hierarchy_path[0] if hierarchy_path else None

            chunk = Chunk(
                id=str(uuid.uuid4()),
                content=flat.get("text", ""),
            )
            chunk.attach_metadata(
                ChunkMetadata(
                    document_id=document_id,
                    source_file=fp.name,
                    document_type=fp.suffix.lstrip(".") or "unknown",
                    title=title,
                    section=section,
                    subsection=flat.get("subsection"),
                    hierarchy_path=hierarchy_path,
                    page_number=flat.get("page_number"),
                    chunk_index=idx,
                    summary=flat.get("summary"),
                    language=flat.get("language", "en"),
                )
            )
            chunks.append(chunk)

        registry = ModelRegistry.instance()
        enricher = registry.get_enricher()
        enriched = [enricher.enrich(chunk) for chunk in chunks]

        embedder = registry.get_embedder()
        embeddings = embedder.embed_texts([c.content for c in enriched])

        vector_store = VectorStore()
        if embeddings:
            try:
                vector_store.upsert_chunks(enriched, embeddings)
            except Exception as e:
                # Retrieval can still work via keyword index when vector db is unavailable.
                print("❌ Qdrant upsert failed:", e)
                raise

        registry.get_keyword_index().add(enriched)

        return {
            "document_id": document_id,
            "chunks_indexed": len(enriched),
        }
