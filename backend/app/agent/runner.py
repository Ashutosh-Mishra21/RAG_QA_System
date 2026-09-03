from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.app.retrieval import EvidenceTools, QueryAnalyzer, QueryDecomposer

from .models import AgentDecision, AgentPlan, AgentSearchOutcome
from backend.app.models import Chunk
logger = logging.getLogger(__name__)


class AgenticQueryRunner:
    """Execute bounded retrieval plans before handing evidence to generation."""

    def __init__(
        self,
        evidence_tools: "EvidenceTools",
        query_rewriter,
        query_decomposer: "QueryDecomposer",
        query_analyzer: "QueryAnalyzer | None" = None,
    ) -> None:
        from backend.app.retrieval import QueryAnalyzer

        self.evidence_tools = evidence_tools
        self.query_rewriter = query_rewriter
        self.query_decomposer = query_decomposer
        self.query_analyzer = query_analyzer or QueryAnalyzer()

    def _initial_plan(self, query: str) -> AgentPlan:
        from backend.app.core.config import settings
        from backend.app.retrieval import QueryAnalyzer

        strategy = self.query_analyzer.analyze(query)
        is_complex = len(query.split()) > 12 or any(
            marker in query.casefold() for marker in (" and ", "compare", "versus", " vs ")
        )
        subqueries = (
            self.query_decomposer.decompose(query) if is_complex else [query]
        )
        actions = ["decompose" if is_complex else "search", "search"]
        if strategy.query_type in {"definition", "explanation"}:
            actions.append("rewrite")
        return AgentPlan(
            query=query,
            query_type=strategy.query_type,
            subqueries=subqueries[: settings.AGENT_MAX_SUBQUERIES],
            actions=actions,
        )

    def _search(
        self,
        query: str,
        metadata_filters: dict[str, object],
        top_k: int,
    ) -> AgentSearchOutcome:
        from backend.app.retrieval import SearchHybridInput

        from backend.app.core.config import settings

        result = self.evidence_tools.search_hybrid(
            SearchHybridInput(
                query=query,
                top_k=min(top_k, settings.RETRIEVAL_MAX_TOP_K),
                metadata_filters=metadata_filters,
            )
        )
        return AgentSearchOutcome(query, result.chunks, result.sufficiency)

    @staticmethod
    def _parse_targeted(
        query: str,
        outcome: AgentSearchOutcome,
        evidence_tools: EvidenceTools,
    ) -> AgentSearchOutcome:
        query_lower = query.casefold()
        action = "parse_table" if "table" in query_lower else "parse_visual"
        if action == "parse_visual" and not any(
            marker in query_lower for marker in ("figure", "chart", "diagram", "visual")
        ):
            return outcome

        from backend.app.retrieval import ParseEvidenceInput

        parsed = []
        seen: set[tuple[str, str]] = set()
        for chunk in outcome.chunks:
            metadata = chunk.metadata or {}
            document_id = str(metadata.get("document_id", ""))
            page_id = metadata.get("page_id")
            page_number = metadata.get("page_number")
            selector = (document_id, str(page_id or page_number))
            if not document_id or selector in seen or (page_id is None and page_number is None):
                continue
            seen.add(selector)
            parsed.extend(
                getattr(evidence_tools, action)(
                    ParseEvidenceInput(
                        document_id=document_id,
                        page_id=str(page_id) if page_id else None,
                        page_number=int(page_number) if page_number else None,
                        tenant_id=str(metadata.get("tenant_id", "default")),
                        workspace_id=str(metadata.get("workspace_id", "default")),
                    )
                )
            )
        if not parsed:
            return outcome
        parsed_chunks = [
            Chunk(
                id=item.evidence_id,
                content=item.text_content(),
                metadata={
                    "document_id": item.document_id,
                    "page_id": item.page_id,
                    "page_number": item.provenance.source_page,
                    "evidence_id": item.evidence_id,
                    "evidence_type": item.evidence_type,
                    "parser_name": item.parser_name,
                    "parser_version": item.parser_version,
                    "confidence": item.confidence,
                    "tenant_id": item.tenant_id,
                    "workspace_id": item.workspace_id,
                    "document_version": item.document_version,
                },
                score=1.0,
            )
            for item in parsed
        ]
        return AgentSearchOutcome(
            query=query,
            chunks=parsed_chunks,
            sufficiency=evidence_tools.verify_evidence(query, parsed),
        )

    def run(
        self, query: str, metadata_filters: dict[str, object]
    ) -> tuple[AgentPlan, list[AgentDecision], list[dict[str, Any]]]:
        from backend.app.core.config import settings

        plan = self._initial_plan(query)
        decisions: list[AgentDecision] = []
        results: list[dict[str, Any]] = []

        for subquery in plan.subqueries:
            current_query = subquery
            outcome: AgentSearchOutcome | None = None
            for attempt in range(1, settings.AGENT_MAX_ATTEMPTS + 1):
                if attempt > 1:
                    current_query = self.query_rewriter.rewrite(current_query)
                    decisions.append(
                        AgentDecision(
                            action="rewrite",
                            query=current_query,
                            attempt=attempt,
                            reason="previous evidence was insufficient",
                        )
                    )
                outcome = self._search(
                    current_query,
                    metadata_filters,
                    settings.AGENT_INITIAL_TOP_K * attempt,
                )
                decisions.append(
                    AgentDecision(
                        action="search",
                        query=current_query,
                        attempt=attempt,
                        reason="retrieve candidate evidence",
                        result_count=len(outcome.chunks),
                        sufficient=outcome.sufficiency.sufficient,
                    )
                )
                if outcome.sufficiency.sufficient:
                    break
                parsed = self._parse_targeted(current_query, outcome, self.evidence_tools)
                if parsed is not outcome:
                    outcome = parsed
                    decisions.append(
                        AgentDecision(
                            action=(
                                "parse_table"
                                if "table" in current_query.casefold()
                                else "parse_visual"
                            ),
                            query=current_query,
                            attempt=attempt,
                            reason="inspect targeted page evidence",
                            sufficient=outcome.sufficiency.sufficient,
                        )
                    )
                    if outcome.sufficiency.sufficient:
                        break

            assert outcome is not None
            if outcome.sufficiency.sufficient:
                results.append(
                    {
                        "query": current_query,
                        "chunks": outcome.chunks,
                        "evidence": [
                            self.evidence_tools.evidence_from_chunk(chunk)
                            for chunk in outcome.chunks
                        ],
                        "sufficiency": outcome.sufficiency,
                    }
                )
            else:
                results.append(
                    {
                        "query": current_query,
                        "chunks": outcome.chunks,
                        "evidence": [
                            self.evidence_tools.evidence_from_chunk(chunk)
                            for chunk in outcome.chunks
                        ],
                        "sufficiency": outcome.sufficiency,
                    }
                )

        logger.info(
            "Agent plan completed query_type=%s subqueries=%s decisions=%s",
            plan.query_type,
            len(plan.subqueries),
            len(decisions),
        )
        return plan, decisions, results
