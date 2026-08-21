import json
from pathlib import Path

from .docling_parser import DoclingParser
from .structure_builder import StructureBuilder
from .node_chunker import NodeChunker
from .tree_flattener import flatten_tree
from .enrichment import ChunkEnricher

from backend.app.indexing import Embedder, VectorStore
from backend.app.core.config import settings
from backend.app.core.embedding_provider import EmbeddingModelProvider


class IngestionOrchestrator:

    def __init__(
        self,
        data_dir: Path,
    ):

        self.raw_dir = data_dir / "raw"

        self.processed_dir = data_dir / "processed"

        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # DEVICE
        # =====================================================

        # =====================================================
        # PARSER
        # =====================================================

        self.parser = DoclingParser()

        # =====================================================
        # STRUCTURE BUILDER
        # =====================================================

        self.builder = StructureBuilder()

        # =====================================================
        # SHARED BGE MODEL
        # =====================================================

        self.embedding_model = EmbeddingModelProvider(
            settings.EMBEDDING_MODEL
        ).get_model()

        # =====================================================
        # SEMANTIC CHUNKER
        # =====================================================

        self.chunker = NodeChunker(
            embedding_model=self.embedding_model,
            max_tokens=settings.CHUNK_MAX_TOKENS,
            min_tokens=settings.CHUNK_MIN_TOKENS,
            similarity_threshold=settings.CHUNK_SIMILARITY_THRESHOLD,
        )

    def run(
        self,
        document_id: str,
        stored_filename: str,
    ):

        file_path = (self.raw_dir / stored_filename).resolve()
        if self.raw_dir.resolve() not in file_path.parents:
            raise ValueError(
                "stored_filename must remain inside the raw data directory"
            )

        # 1. Parse
        document = self.parser.parse(file_path)

        # 2. Build hierarchical tree
        tree = self.builder.build_tree(document)

        # 3. Semantic chunking
        for root in tree:

            self.chunker.merge_chunks(root)

        # 4. Flatten
        flat_chunks = flatten_tree(
            tree,
            document_id=document_id,
            source_file=file_path.name,
            document_type=file_path.suffix.lstrip(".") or "unknown",
        )

        # 5. Save tree
        tree_output_path = self.processed_dir / f"{document_id}_tree.json"

        with open(
            tree_output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                [node.model_dump() for node in tree],
                f,
                indent=2,
                ensure_ascii=False,
            )

        # 6. Save flat chunks
        flat_output_path = self.processed_dir / f"{document_id}_flat.json"

        with open(
            flat_output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                flat_chunks,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "document_id": document_id,
            "tree_file": str(tree_output_path),
            "flat_file": str(flat_output_path),
            "num_chunks": len(flat_chunks),
        }


class EmbeddingPipeline:

    def __init__(self):

        # =====================================================
        # ONE SHARED BGE MODEL
        # =====================================================

        self.embedding_model = EmbeddingModelProvider(
            settings.EMBEDDING_MODEL
        ).get_model()

        # =====================================================
        # ENRICHMENT
        # =====================================================

        self.enricher = ChunkEnricher(embedding_model=self.embedding_model)

        # =====================================================
        # EMBEDDING WRAPPER
        # =====================================================

        self.embedder = Embedder(
            model=self.embedding_model,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )

        # =====================================================
        # VECTOR STORE
        # =====================================================

        self.vector_store = VectorStore()

    def process_chunks(
        self,
        chunks,
    ):

        if not chunks:
            return

        # =====================================================
        # 1. ENRICH
        # =====================================================

        enriched = [self.enricher.enrich(chunk) for chunk in chunks]

        # =====================================================
        # 2. EMBEDDING TEXT
        # =====================================================

        texts = [
            getattr(
                chunk,
                "embedding_text",
                chunk.content,
            )
            for chunk in enriched
        ]

        # =====================================================
        # 3. EMBEDDINGS
        # =====================================================

        embeddings = self.embedder.embed_documents(texts)

        if not embeddings:
            return

        # =====================================================
        # 4. VECTOR DB
        # =====================================================

        self.vector_store.upsert_chunks(
            enriched,
            embeddings,
        )
