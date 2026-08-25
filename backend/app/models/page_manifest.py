from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class PageSignals(BaseModel):
    text: bool = False
    table: bool = False
    image: bool = False
    chart: bool = False
    formula: bool = False
    scanned: bool = False


class PageStatistics(BaseModel):
    text_chars: int = Field(default=0, ge=0)
    text_blocks: int = Field(default=0, ge=0)
    image_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    numeric_token_count: int = Field(default=0, ge=0)
    text_density: float = Field(default=0.0, ge=0.0, le=1.0)
    heading_count: int = Field(default=0, ge=0)
    list_item_count: int = Field(default=0, ge=0)


class PageSource(BaseModel):
    page_bbox: list[float] = Field(default_factory=list)
    storage_path: Optional[str] = None
    page_rotation: Optional[int] = None


class PageBlock(BaseModel):
    block_id: str
    text: str = ""
    bbox: list[float] = Field(default_factory=list)
    block_type: str = "text"
    line_start: Optional[int] = Field(default=None, ge=1)
    line_end: Optional[int] = Field(default=None, ge=1)


class PageManifest(BaseModel):
    document_id: str
    page_id: str
    page_number: int = Field(ge=1)
    text: str = ""
    blocks: list[PageBlock] = Field(default_factory=list)
    signals: PageSignals = Field(default_factory=PageSignals)
    statistics: PageStatistics = Field(default_factory=PageStatistics)
    section: Optional[str] = None
    section_level: Optional[int] = Field(default=None, ge=0)
    source: PageSource = Field(default_factory=PageSource)


class PageRoutingDecision(BaseModel):
    page_id: str
    actions: list[str] = Field(default_factory=list)
    reason: str
    priority: int = Field(default=0, ge=0, le=10)


class DocumentManifest(BaseModel):
    schema_version: int = 1
    scanner_name: str = "pymupdf"
    scanner_version: Optional[str] = None
    document_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    document_id: str
    source_file: str
    document_type: str
    page_count: int = Field(ge=0)
    pages: list[PageManifest] = Field(default_factory=list)
    routing: list[PageRoutingDecision] = Field(default_factory=list)
