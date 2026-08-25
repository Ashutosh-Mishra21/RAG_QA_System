from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EvidenceProvenance(BaseModel):
    source_page: int = Field(ge=1)
    source_bbox: Optional[list[float]] = None
    storage_path: Optional[str] = None
    page_id: Optional[str] = None


class EvidenceItem(BaseModel):
    evidence_id: str
    document_id: str
    page_id: str
    tenant_id: str = "default"
    workspace_id: str = "default"
    document_version: str = "1"
    evidence_type: Literal[
        "text", "table", "figure", "chart", "image", "formula", "caption"
    ]
    content: str | dict[str, Any]
    bbox: Optional[list[float]] = None
    parser_name: str
    parser_version: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: EvidenceProvenance
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def text_content(self) -> str:
        """Return the representation used by lexical and vector indexes."""
        if isinstance(self.content, str):
            return self.content
        import json

        return json.dumps(self.content, sort_keys=True, ensure_ascii=True)
