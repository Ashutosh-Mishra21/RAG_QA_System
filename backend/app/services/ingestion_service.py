import logging
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from backend.app.core.model_registry import ModelRegistry
from backend.app.indexing import VectorStore
from backend.app.ingestion import (
    DoclingParser,
    StructureBuilder,
    NodeChunker,
    flatten_tree,
)
from backend.app.models import Chunk, ChunkMetadata
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


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

        logger.info(
            "Saved uploaded file: filename=%s stored_as=%s document_id=%s",
            filename,
            new_filename,
            document_id,
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "stored_as": new_filename,
            "file_path": str(file_path),
            "timestamp": timestamp,
        }

    def ingest_and_index(self, file_path: str) -> dict:
        fp = Path(file_path).resolve()
        if self.raw_dir.resolve() not in fp.parents:
            raise ValueError("file_path must remain inside the raw data directory")
        document_id = fp.stem

        registry = ModelRegistry.instance()
        embedder = registry.get_embedder()
        parser = DoclingParser()
        builder = StructureBuilder()
        model = getattr(embedder, "model", None)
        chunker = NodeChunker(
            embedding_model=model, batch_size=settings.EMBEDDING_BATCH_SIZE
        )

        document = parser.parse(fp)
        tree = builder.build_tree(document)
        for root in tree:
            chunker.merge_chunks(root)

        document_type = fp.suffix.lstrip(".") or "unknown"
        flat_chunks = flatten_tree(
            tree,
            document_id=document_id,
            source_file=fp.name,
            document_type=document_type,
        )

        chunks: List[Chunk] = []
        for idx, flat in enumerate(flat_chunks):
            heading_path = flat.get("heading_path", [])
            hierarchy_path = [h.get("heading", "") for h in heading_path]
            content = flat.get("text", "")
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            stable_key = ":".join(
                [
                    document_id,
                    " > ".join(hierarchy_path),
                    str(flat.get("chunk_index", idx)),
                    content_hash,
                ]
            )
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))

            chunk = Chunk(
                id=chunk_id,
                content=content,
            )
            chunk.attach_metadata(
                ChunkMetadata(
                    document_id=document_id,
                    source_file=fp.name,
                    document_type=document_type,
                    title=flat.get("title"),
                    section=flat.get("section"),
                    subsection=flat.get("subsection"),
                    hierarchy_path=hierarchy_path,
                    page_number=flat.get("page_number"),
                    chunk_index=flat.get("chunk_index", idx),
                    summary=flat.get("summary"),
                    language=flat.get("language", "en"),
                )
            )
            chunks.append(chunk)

        enricher = registry.get_enricher()
        enriched = [enricher.enrich(chunk) for chunk in chunks]

        embedding_texts = [
            flat.get("embedding_text", c.content)
            for flat, c in zip(flat_chunks, enriched)
        ]
        embed_documents = getattr(embedder, "embed_documents", embedder.embed_texts)
        embeddings = embed_documents(embedding_texts)

        vector_store = VectorStore()
        if embeddings:
            try:
                vector_store.upsert_chunks(enriched, embeddings)
            except Exception as exc:
                # Retrieval can still work via keyword index when vector db is unavailable.
                logger.exception("Qdrant upsert failed: %s", exc)
                raise

        registry.get_keyword_index().add(enriched)

        logger.info(
            "Ingestion completed: document_id=%s chunks_indexed=%s",
            document_id,
            len(enriched),
        )

        return {
            "document_id": document_id,
            "chunks_indexed": len(enriched),
        }
