from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from backend.app.ingestion.evidence_cache import EvidenceCache
from backend.app.models import Chunk, EvidenceItem, EvidenceProvenance


class SearchHybridInput(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=50)
    metadata_filters: dict[str, object] = Field(default_factory=dict)


class GetPageInput(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    page_id: Optional[str] = None
    page_number: Optional[int] = Field(default=None, ge=1)
    tenant_id: str = "default"
    workspace_id: str = "default"

    @model_validator(mode="after")
    def require_page_selector(self):
        if self.page_id is None and self.page_number is None:
            raise ValueError("page_id or page_number is required")
        return self


class GetRegionInput(GetPageInput):
    bbox: list[float] = Field(min_length=4, max_length=4)


class ParseEvidenceInput(GetPageInput):
    pass


class EvidenceSufficiency(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    sufficient: bool
    evidence_count: int = Field(ge=0)
    covered_query_terms: int = Field(ge=0)
    query_terms: int = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)


class EvidenceSearchResult(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)
    sufficiency: EvidenceSufficiency
    chunks: list[Chunk] = Field(default_factory=list, exclude=True)


class EvidenceSufficiencyScorer:
    """Deterministic gate based on query coverage and provenance completeness."""

    threshold = 0.55

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"\b\w{3,}\b", text)
        }

    def score(self, query: str, evidence: list[EvidenceItem]) -> EvidenceSufficiency:
        terms = self._terms(query)
        valid = [item for item in evidence if item.text_content().strip()]
        corpus_terms = self._terms(" ".join(item.text_content() for item in valid))
        covered = len(terms & corpus_terms)
        coverage = covered / len(terms) if terms else 1.0
        provenance = (
            sum(
                bool(item.document_id)
                and bool(item.page_id)
                and item.provenance.source_page >= 1
                for item in valid
            )
            / len(valid)
            if valid
            else 0.0
        )
        page_diversity = min(len({item.page_id for item in valid}) / 3.0, 1.0)
        count_score = min(len(valid) / 3.0, 1.0)
        score = round(
            0.35 * coverage
            + 0.30 * provenance
            + 0.20 * page_diversity
            + 0.15 * count_score,
            4,
        )
        reasons = []
        if not valid:
            reasons.append("no retrievable evidence")
        if coverage < 0.5:
            reasons.append("weak query-term coverage")
        if provenance < 1.0:
            reasons.append("incomplete evidence provenance")
        sufficient = bool(valid) and score >= self.threshold
        if not sufficient and not reasons:
            reasons.append("evidence score below threshold")
        return EvidenceSufficiency(
            score=score,
            sufficient=sufficient,
            evidence_count=len(valid),
            covered_query_terms=covered,
            query_terms=len(terms),
            reasons=reasons,
        )


class EvidenceRepository:
    """Read canonical evidence by document/page/region with a local selection cache."""

    def __init__(self, storage_dir: str | Path):
        self.processed_dir = Path(storage_dir) / "processed"
        self.cache = EvidenceCache(self.processed_dir / "evidence_cache")

    def _load_document(self, document_id: str):
        if Path(document_id).name != document_id:
            raise ValueError("Invalid document_id")
        path = self.processed_dir / f"{document_id}_canonical.json"
        if not path.is_file():
            return None
        from backend.app.models import CanonicalDocument

        return CanonicalDocument.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _matches_scope(
        evidence: EvidenceItem, tenant_id: str, workspace_id: str
    ) -> bool:
        return (
            evidence.tenant_id == tenant_id
            and evidence.workspace_id == workspace_id
        )

    def get_page(self, request: GetPageInput) -> list[EvidenceItem]:
        document = self._load_document(request.document_id)
        if document is None:
            return []
        selected_page_id = request.page_id or next(
            (
                page.page_id
                for page in document.pages
                if page.page_number == request.page_number
            ),
            f"{request.document_id}_p{request.page_number}",
        )
        return self.cache.get_or_load(
            request.document_id,
            selected_page_id,
            None,
            "get_page",
            request.tenant_id,
            request.workspace_id,
            lambda: [
                item
                for item in document.evidence
                if item.page_id == selected_page_id
                and self._matches_scope(item, request.tenant_id, request.workspace_id)
            ],
        )

    def get_region(self, request: GetRegionInput) -> list[EvidenceItem]:
        page_evidence = self.get_page(request)
        page_id = request.page_id or (
            page_evidence[0].page_id
            if page_evidence
            else f"{request.document_id}_p{request.page_number}"
        )
        return self.cache.get_or_load(
            request.document_id,
            page_id,
            request.bbox,
            "get_region",
            request.tenant_id,
            request.workspace_id,
            lambda: [
                item
                for item in page_evidence
                if item.bbox is None or self._intersects(item.bbox, request.bbox)
            ],
        )

    @staticmethod
    def _intersects(first: list[float], second: list[float]) -> bool:
        return not (
            first[2] < second[0]
            or second[2] < first[0]
            or first[3] < second[1]
            or second[3] < first[1]
        )


class EvidenceTools:
    """Bounded tool surface for evidence-first retrieval and inspection."""

    def __init__(
        self,
        retriever,
        storage_dir: str | Path | None = None,
        page_parser: Optional[
            Callable[[ParseEvidenceInput, Literal["parse_table", "parse_visual"]], list[EvidenceItem]]
        ] = None,
    ):
        self.retriever = retriever
        self.repository = EvidenceRepository(storage_dir) if storage_dir else None
        self.page_parser = page_parser
        self.sufficiency = EvidenceSufficiencyScorer()

    @staticmethod
    def evidence_from_chunk(chunk: Chunk) -> EvidenceItem:
        metadata = chunk.metadata or {}
        document_id = str(metadata.get("document_id", "unknown"))
        page_number = int(metadata.get("page_number") or 1)
        page_id = str(metadata.get("page_id") or f"{document_id}_p{page_number}")
        bbox = metadata.get("bbox")
        return EvidenceItem(
            evidence_id=str(metadata.get("evidence_id") or chunk.id),
            document_id=document_id,
            page_id=page_id,
            tenant_id=str(metadata.get("tenant_id", "default")),
            workspace_id=str(metadata.get("workspace_id", "default")),
            document_version=str(metadata.get("document_version", "1")),
            evidence_type=str(metadata.get("evidence_type", "text")),
            content=chunk.content,
            bbox=bbox if isinstance(bbox, list) else None,
            parser_name=str(metadata.get("parser_name", "legacy-index")),
            parser_version=metadata.get("parser_version"),
            confidence=float(metadata.get("confidence", 1.0) or 0.0),
            provenance=EvidenceProvenance(
                source_page=page_number,
                source_bbox=bbox if isinstance(bbox, list) else None,
                page_id=page_id,
            ),
        )

    def search_hybrid(self, request: SearchHybridInput) -> EvidenceSearchResult:
        chunks = self.retriever.retrieve(
            request.query,
            top_k=request.top_k,
            metadata_filters=request.metadata_filters,
        )
        evidence = [self.evidence_from_chunk(chunk) for chunk in chunks]
        return EvidenceSearchResult(
            evidence=evidence,
            sufficiency=self.sufficiency.score(request.query, evidence),
            chunks=chunks,
        )

    def get_page(self, request: GetPageInput) -> list[EvidenceItem]:
        return self._require_repository().get_page(request)

    def get_region(self, request: GetRegionInput) -> list[EvidenceItem]:
        return self._require_repository().get_region(request)

    def parse_table(self, request: ParseEvidenceInput) -> list[EvidenceItem]:
        return self._parse(request, "parse_table")

    def parse_visual(self, request: ParseEvidenceInput) -> list[EvidenceItem]:
        return self._parse(request, "parse_visual")

    def verify_evidence(
        self, query: str, evidence: list[EvidenceItem]
    ) -> EvidenceSufficiency:
        return self.sufficiency.score(query, evidence)

    def _parse(
        self, request: ParseEvidenceInput, action: Literal["parse_table", "parse_visual"]
    ) -> list[EvidenceItem]:
        if self.page_parser is not None:
            return self.page_parser(request, action)
        evidence = self.get_page(request)
        expected_type = "table" if action == "parse_table" else "figure"
        return [item for item in evidence if item.evidence_type == expected_type]

    def _require_repository(self) -> EvidenceRepository:
        if self.repository is None:
            raise RuntimeError("Evidence page tools require a local storage directory")
        return self.repository
