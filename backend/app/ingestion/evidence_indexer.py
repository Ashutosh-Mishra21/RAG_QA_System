from __future__ import annotations

from backend.app.models import CanonicalDocument, Chunk, ChunkMetadata, EvidenceItem


class EvidenceIndexer:
    """Project canonical evidence into the existing index record contract."""

    def build_chunks(self, document: CanonicalDocument) -> list[Chunk]:
        return [
            self._build_chunk(document, evidence, index)
            for index, evidence in enumerate(document.evidence)
            if evidence.text_content().strip()
        ]

    @staticmethod
    def _build_chunk(
        document: CanonicalDocument, evidence: EvidenceItem, index: int
    ) -> Chunk:
        chunk = Chunk(id=evidence.evidence_id, content=evidence.text_content())
        chunk.attach_metadata(
            ChunkMetadata(
                document_id=evidence.document_id,
                source_file=document.source_file,
                document_type=document.document_type,
                evidence_id=evidence.evidence_id,
                page_id=evidence.page_id,
                bbox=evidence.bbox or evidence.provenance.source_bbox,
                evidence_type=evidence.evidence_type,
                parser_name=evidence.parser_name,
                parser_version=evidence.parser_version,
                confidence=evidence.confidence,
                tenant_id=evidence.tenant_id,
                workspace_id=evidence.workspace_id,
                document_version=evidence.document_version,
                section=next(
                    (
                        page.section
                        for page in document.pages
                        if page.page_id == evidence.page_id
                    ),
                    None,
                ),
                page_number=evidence.provenance.source_page,
                chunk_index=index,
            )
        )
        return chunk
