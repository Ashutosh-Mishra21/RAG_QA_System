from __future__ import annotations

import hashlib
import uuid
from typing import Any

from backend.app.models import (
    CanonicalDocument,
    DocumentManifest,
    EvidenceItem,
    EvidenceProvenance,
)


class Canonicalizer:
    """Convert scanner/parser output into stable application-owned models."""

    def from_manifest(self, manifest: DocumentManifest) -> CanonicalDocument:
        evidence: list[EvidenceItem] = []
        for page in manifest.pages:
            if not page.text:
                continue
            blocks = page.blocks or [
                type(
                    "PageText", (), {"text": page.text, "bbox": page.source.page_bbox}
                )()
            ]
            for block in blocks:
                digest = hashlib.sha256(
                    f"{manifest.document_id}:{page.page_id}:{block.text}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                evidence.append(
                    EvidenceItem(
                        evidence_id=str(uuid.uuid5(uuid.NAMESPACE_URL, digest)),
                        document_id=manifest.document_id,
                        page_id=page.page_id,
                        evidence_type="text",
                        content=block.text,
                        parser_name="pymupdf",
                        confidence=1.0,
                        bbox=block.bbox or None,
                        provenance=EvidenceProvenance(
                            source_page=page.page_number,
                            source_bbox=block.bbox or page.source.page_bbox,
                            storage_path=page.source.storage_path,
                            page_id=page.page_id,
                        ),
                    )
                )
        return CanonicalDocument(
            document_id=manifest.document_id,
            source_file=manifest.source_file,
            document_type=manifest.document_type,
            pages=manifest.pages,
            evidence=evidence,
        )

    def from_docling(
        self,
        document: Any,
        document_id: str,
        source_file: str,
        document_type: str,
        tenant_id: str = "default",
        workspace_id: str = "default",
        document_version: str = "1",
    ) -> CanonicalDocument:
        """Create a canonical text document from Docling-compatible items.

        Detailed tables and visuals are added by specialist adapters in later stages.
        """
        evidence: list[EvidenceItem] = []
        for index, (element, _) in enumerate(document.iterate_items(), start=1):
            text = str(getattr(element, "text", "")).strip()
            if not text:
                continue
            page_number = getattr(element, "page_number", None) or 1
            source_bbox = self._source_bbox(element)
            evidence_type = self._evidence_type(element)
            evidence_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{document_id}:{evidence_type}:{index}:{text}",
                )
            )
            evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    document_id=document_id,
                    page_id=f"{document_id}_p{page_number}",
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    document_version=document_version,
                    evidence_type=evidence_type,
                    content=text,
                    parser_name="docling",
                    confidence=1.0,
                    bbox=source_bbox,
                    provenance=EvidenceProvenance(
                        source_page=page_number,
                        source_bbox=source_bbox,
                        page_id=f"{document_id}_p{page_number}",
                    ),
                )
            )
        return CanonicalDocument(
            document_id=document_id,
            source_file=source_file,
            document_type=document_type,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            document_version=document_version,
            evidence=evidence,
        )

    @staticmethod
    def _source_bbox(element: Any) -> list[float] | None:
        bbox = getattr(element, "source_bbox", None)
        if isinstance(bbox, dict):
            bbox = bbox.get("bbox")
        if bbox is None:
            provenance = getattr(element, "prov", None) or []
            if provenance:
                bbox = getattr(provenance[-1], "bbox", None)
                if hasattr(bbox, "model_dump"):
                    bbox = bbox.model_dump()
                if isinstance(bbox, dict):
                    bbox = bbox.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return [float(value) for value in bbox]
        return None

    @staticmethod
    def _evidence_type(element: Any) -> str:
        label = str(getattr(getattr(element, "label", None), "name", "")).upper()
        if "TABLE" in label:
            return "table"
        if any(token in label for token in ("FIGURE", "PICTURE", "IMAGE", "CHART")):
            return "figure"
        if "FORMULA" in label:
            return "formula"
        if "CAPTION" in label:
            return "caption"
        return "text"
