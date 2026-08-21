from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from backend.app.models import Chunk, ChunkMetadata


class ChunkFactory:
    """Create final chunks from metadata-complete flattened records."""

    def create(
        self,
        flat_record: dict[str, Any],
        document_id: str,
        source_file: str,
        document_type: str,
    ) -> Chunk:
        content = str(flat_record.get("text", "")).strip()
        hierarchy = flat_record.get("heading_path", [])
        hierarchy_path = [
            str(item.get("heading", "")) for item in hierarchy if item.get("heading")
        ]
        chunk_index = int(flat_record.get("chunk_index", 0))
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        stable_key = ":".join(
            [
                document_id,
                str(flat_record.get("node_id", "")),
                str(chunk_index),
                content_hash,
            ]
        )
        chunk = Chunk(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
            content=content,
        )
        chunk.attach_metadata(
            ChunkMetadata(
                document_id=document_id,
                source_file=source_file,
                document_type=document_type,
                title=flat_record.get("title")
                or (hierarchy_path[0] if hierarchy_path else None),
                section=flat_record.get("section")
                or (hierarchy_path[-1] if hierarchy_path else None),
                subsection=flat_record.get("subsection"),
                hierarchy_path=hierarchy_path,
                page_number=flat_record.get("page_number"),
                chunk_index=chunk_index,
                summary=flat_record.get("summary"),
                language=flat_record.get("language", "en"),
            )
        )
        return chunk
