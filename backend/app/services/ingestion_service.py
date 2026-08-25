import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from backend.app.core.model_registry import ModelRegistry
from backend.app.indexing import VectorStore
from backend.app.ingestion import (
    DoclingParser,
    PageScanner,
    Canonicalizer,
    PageRoutingPolicy,
    PageActionExecutor,
    EvidenceIndexer,
)
from backend.app.models import CanonicalDocument, Chunk, DocumentManifest, PageRoutingDecision
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.raw_dir = self.storage_dir / "raw"
        self.processed_dir = self.storage_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.page_scanner = PageScanner(
            max_bytes=settings.MAX_UPLOAD_BYTES,
            max_pages=settings.MAX_PDF_PAGES,
            max_page_text_chars=settings.MAX_PAGE_TEXT_CHARS,
            max_page_blocks=settings.MAX_PAGE_BLOCKS,
        )
        self.page_routing = PageRoutingPolicy()

    def save_file(
        self,
        filename: str,
        file_bytes: bytes,
        tenant_id: str = "default",
        workspace_id: str = "default",
    ):
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
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
        }

    def ingest_and_index(
        self,
        file_path: str,
        tenant_id: str = "default",
        workspace_id: str = "default",
        document_version: str | None = None,
    ) -> dict:
        fp = Path(file_path).resolve()
        if self.raw_dir.resolve() not in fp.parents:
            raise ValueError("file_path must remain inside the raw data directory")
        document_id = fp.stem

        scan_supported = fp.suffix.lower() in {".pdf", ".md", ".markdown", ".txt"}
        manifest_path = (
            self.processed_dir / f"{document_id}_manifest.json"
            if scan_supported
            else None
        )
        canonical_path = self.processed_dir / f"{document_id}_canonical.json"
        canonicalizer = Canonicalizer()
        if scan_supported:
            manifest = self.page_scanner.load_or_scan(
                fp, manifest_path, document_id=document_id
            )
            self.page_routing.decide(manifest)
            self.page_scanner.persist(manifest, manifest_path)
            base_canonical = canonicalizer.from_manifest(manifest)
            canonical = base_canonical.model_copy(
                update={
                    "tenant_id": tenant_id,
                    "workspace_id": workspace_id,
                    "document_version": document_version or manifest.document_hash or "1",
                    "evidence": [
                        item.model_copy(
                            update={
                                "tenant_id": tenant_id,
                                "workspace_id": workspace_id,
                                "document_version": document_version
                                or manifest.document_hash
                                or "1",
                            }
                        )
                        for item in base_canonical.evidence
                    ],
                }
            )

        if scan_supported:
            if fp.suffix.lower() == ".pdf":
                canonical = PageActionExecutor().execute(fp, manifest, canonical)
        else:
            parser = DoclingParser()
            canonical = canonicalizer.from_docling(
                parser.parse(fp),
                document_id=document_id,
                source_file=fp.name,
                document_type=fp.suffix.lstrip(".") or "unknown",
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                document_version=document_version or "1",
            )

        canonical_path.write_text(
            json.dumps(canonical.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

        chunks = self._index_canonical(canonical)

        logger.info(
            "Ingestion completed: document_id=%s chunks_indexed=%s",
            document_id,
            len(chunks),
        )

        return {
            "document_id": document_id,
            "chunks_indexed": len(chunks),
            "manifest_file": str(manifest_path) if manifest_path else None,
            "canonical_file": str(canonical_path),
        }

    def refine_page(
        self,
        document_id: str,
        page_id: str | None,
        page_number: int | None,
        action: str,
        tenant_id: str = "default",
        workspace_id: str = "default",
    ) -> list:
        if action not in {"parse_table", "parse_visual"}:
            raise ValueError("Unsupported page action")
        if Path(document_id).name != document_id:
            raise ValueError("Invalid document_id")
        manifest_path = self.processed_dir / f"{document_id}_manifest.json"
        canonical_path = self.processed_dir / f"{document_id}_canonical.json"
        if not manifest_path.is_file() or not canonical_path.is_file():
            return []

        manifest = DocumentManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        canonical = CanonicalDocument.model_validate_json(
            canonical_path.read_text(encoding="utf-8")
        )
        target = next(
            (
                page
                for page in manifest.pages
                if page.page_id == page_id
                or (page_id is None and page.page_number == page_number)
            ),
            None,
        )
        if target is None:
            return []
        if canonical.tenant_id != tenant_id or canonical.workspace_id != workspace_id:
            return []

        updated_routing = []
        for decision in manifest.routing:
            if decision.page_id == target.page_id:
                updated_routing.append(
                    PageRoutingDecision(
                        page_id=decision.page_id,
                        actions=list(dict.fromkeys([*decision.actions, action])),
                        reason=decision.reason,
                        priority=max(decision.priority, 8),
                    )
                )
            else:
                updated_routing.append(decision)
        if not any(item.page_id == target.page_id for item in updated_routing):
            updated_routing.append(
                PageRoutingDecision(
                    page_id=target.page_id,
                    actions=[action],
                    reason="targeted evidence request",
                    priority=8,
                )
            )
        manifest.routing = updated_routing

        source_files = list(self.raw_dir.glob(f"{document_id}.*"))
        if len(source_files) != 1:
            raise FileNotFoundError("Raw source for targeted parse was not found")
        refined = PageActionExecutor().execute(
            source_files[0], manifest, canonical, page_ids={target.page_id}
        )
        self.page_scanner.persist(manifest, manifest_path)
        canonical_path.write_text(
            json.dumps(refined.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        self._index_canonical(refined, replace_document=True)
        expected_types = (
            {"table"} if action == "parse_table" else {"figure", "chart", "image"}
        )
        return [
            item
            for item in refined.evidence
            if item.page_id == target.page_id and item.evidence_type in expected_types
        ]

    @staticmethod
    def _safe_delete_document(vector_store, document_id: str) -> None:
        try:
            vector_store.delete_document(document_id)
        except Exception:
            logger.warning("Could not clear previous vector evidence for %s", document_id)

    def _index_canonical(
        self, canonical: CanonicalDocument, replace_document: bool = False
    ) -> List[Chunk]:
        registry = ModelRegistry.instance()
        embedder = registry.get_embedder()
        chunks: List[Chunk] = EvidenceIndexer().build_chunks(canonical)
        embedding_texts = [chunk.content for chunk in chunks]
        embed_documents = getattr(embedder, "embed_documents", embedder.embed_texts)
        embeddings = embed_documents(embedding_texts)

        vector_store = VectorStore()
        if replace_document:
            self._safe_delete_document(vector_store, canonical.document_id)
        if embeddings:
            try:
                vector_store.upsert_chunks(chunks, embeddings)
            except Exception as exc:
                # Retrieval can still work via keyword index when vector db is unavailable.
                logger.exception("Qdrant upsert failed: %s", exc)
                raise

        keyword_index = registry.get_keyword_index()
        if replace_document and hasattr(keyword_index, "delete_document"):
            keyword_index.delete_document(canonical.document_id)
        keyword_index.add(chunks)
        return chunks
