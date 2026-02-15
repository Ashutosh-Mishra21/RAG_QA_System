from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field


class ContentBlock(BaseModel):
    id: str = Field(..., description="Unique block identifier")
    text: str = Field(..., description="Raw text content")
    role: Literal[
        "title", "heading", "paragraph", "list", "table", "figure", "reference"
    ]
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class BaseDocument(BaseModel):
    document_id: str
    source: str
    document_type: str

    title: Optional[str] = None
    authors: Optional[List[str]] = None
    language: Optional[str] = "en"

    content_blocks: List[ContentBlock]

    extra_metadata: Dict[str, str] = Field(default_factory=dict)


class PolicyDocument(BaseDocument):
    document_type: Literal["policy"] = "policy"

    effective_date: Optional[str] = None
    department: Optional[str] = None

    policy_sections: Optional[List[str]] = None


class StudyDocument(BaseDocument):
    document_type: Literal["study"] = "study"

    subject: Optional[str] = None
    level: Optional[Literal["school", "college", "competitive"]] = None

    chapters: Optional[List[str]] = None


class ResearchArticleDocument(BaseDocument):
    document_type: Literal["research"] = "research"

    journal: Optional[str] = None
    publication_year: Optional[int] = None
    doi: Optional[str] = None

    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
