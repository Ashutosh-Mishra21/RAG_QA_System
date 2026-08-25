from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .evidence import EvidenceItem
from .page_manifest import PageManifest


class CanonicalDocument(BaseModel):
    schema_version: int = 1
    document_id: str
    source_file: str
    document_type: str
    tenant_id: str = "default"
    workspace_id: str = "default"
    document_version: str = "1"
    title: Optional[str] = None
    pages: list[PageManifest] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
