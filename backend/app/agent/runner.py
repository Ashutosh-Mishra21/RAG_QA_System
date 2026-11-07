from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any
from langsmith import traceable, get_current_run_tree

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

    @traceable(name="agent.initial_plan", run_type="chain")
    def _initial_plan(self, query: str) -> AgentPlan:
        from backend.app.core.config import settings

        strategy = self.query_analyzer.analyze(query)
        is_complex = len(query.split()) > 12 or any(
            marker in query.casefold()
            for marker in (" and ", "compare", "versus", " vs ")
        )
        decomposed = self.query_decomposer.decompose(query) if is_complex else []
        # Always search the original question; decomposition adds targets rather
        # than replacing the user intent. The configured count applies to extras.
        targets = [query.strip()]
        seen = {query.strip().casefold()}
        for candidate in decomposed:
            candidate = str(candidate).strip()
            key = candidate.casefold()
            if not candidate or key in seen:
                continue
            if len(targets) - 1 >= max(0, settings.AGENT_MAX_SUBQUERIES):
                break
            seen.add(key)
            targets.append(candidate)

        actions = ["decompose" if is_complex else "search", "search"]
        if strategy.query_type in {"definition", "explanation"}:
            actions.append("rewrite")
        return AgentPlan(
            query=query,
            query_type=strategy.query_type,
            subqueries=targets,
            actions=actions,
        )

    @traceable(name="agent.search", run_type="retriever")
    def _search(
        self,
        query: str,
        metadata_filters: dict[str, object],
        top_k: int,
    ) -> AgentSearchOutcome:
        from backend.app.retrieval import SearchHybridInput

        from backend.app.core.config import settings

        started = time.perf_counter()
        result = self.evidence_tools.search_hybrid(
            SearchHybridInput(
                query=query,
                top_k=min(top_k, settings.RETRIEVAL_MAX_TOP_K),
                metadata_filters=metadata_filters,
            )
        )
        logger.info(
            "Agent search completed duration_ms=%.1f result_count=%s sufficient=%s reasons=%s",
            (time.perf_counter() - started) * 1000,
            len(result.chunks),
            result.sufficiency.sufficient,
            result.sufficiency.reasons,
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
            if (
                not document_id
                or selector in seen
                or (page_id is None and page_number is None)
            ):
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

    @staticmethod
    def _follow_up_gap(outcome: AgentSearchOutcome) -> str | None:
        """Return an actionable evidence gap; provenance-only gaps do not retry."""
        reasons = outcome.sufficiency.reasons
        missing = next(
            (reason for reason in reasons if reason.startswith("missing query terms:")),
            None,
        )
        if missing:
            return missing
        if "no retrievable evidence" in reasons or "no query-term overlap" in reasons:
            return "no relevant passage found for the query terms"
        if "weak query-term coverage" in reasons:
            return "weak query-term coverage"
        return None

    @traceable(name="agent.run", run_type="chain")
    def run(
        self, query: str, metadata_filters: dict[str, object]
    ) -> tuple[AgentPlan, list[AgentDecision], list[dict[str, Any]]]:
        from backend.app.core.config import settings

        run_started = time.perf_counter()
        run_tree = get_current_run_tree()
        plan_started = time.perf_counter()
        plan = self._initial_plan(query)
        plan_duration_ms = (time.perf_counter() - plan_started) * 1000
        max_tasks = max(1, settings.AGENT_MAX_SEARCH_TASKS)
        targets = plan.subqueries[:max_tasks]
        first_pass_concurrency = max(
            1, min(settings.AGENT_FIRST_PASS_CONCURRENCY, len(targets) or 1)
        )
        if run_tree:
            run_tree.metadata.update(
                {
                    "query_type": plan.query_type,
                    "subquery_count": len(plan.subqueries),
                    "search_target_count": len(targets),
                    "max_attempts": min(2, max(1, settings.AGENT_MAX_ATTEMPTS)),
                    "max_search_tasks": max_tasks,
                    "first_pass_concurrency": first_pass_concurrency,
                    "planning_duration_ms": round(plan_duration_ms, 2),
                }
            )

        decisions: list[AgentDecision] = []
        results: list[dict[str, Any]] = []
        first_pass: dict[str, AgentSearchOutcome] = {}

        # Independent first-pass targets execute concurrently. Each underlying
        # dense/BM25 call still goes through HybridRetriever's shared bounded pool.
        with ThreadPoolExecutor(
            max_workers=first_pass_concurrency,
            thread_name_prefix="rag-agent-first-pass",
        ) as executor:
            future_to_query = {
                executor.submit(
                    self._search,
                    target,
                    metadata_filters,
                    settings.AGENT_INITIAL_TOP_K,
                ): target
                for target in targets
            }
            for future in as_completed(future_to_query):
                target = future_to_query[future]
                first_pass[target] = future.result()

        # Record all first-pass decisions before follow-ups so traces distinguish
        # independent initial searches from conditional second-hop work.
        for target in targets:
            outcome = first_pass[target]
            reason = "; ".join(outcome.sufficiency.reasons) or "no sufficiency gaps reported"
            decisions.append(
                AgentDecision(
                    action="search",
                    query=target,
                    attempt=1,
                    reason="first-pass retrieval: " + reason,
                    result_count=len(outcome.chunks),
                    sufficient=outcome.sufficiency.sufficient,
                )
            )

        search_tasks_used = len(targets)
        follow_up_count = 0
        max_attempts = min(2, max(1, settings.AGENT_MAX_ATTEMPTS))

        for target in targets:
            current_query = target
            outcome = first_pass[target]

            if not outcome.sufficiency.sufficient:
                parsed = self._parse_targeted(
                    current_query, outcome, self.evidence_tools
                )
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
                            attempt=1,
                            reason="inspect targeted page evidence; "
                            + ("; ".join(outcome.sufficiency.reasons) or "no gaps reported"),
                            sufficient=outcome.sufficiency.sufficient,
                        )
                    )

            gap = None if outcome.sufficiency.sufficient else self._follow_up_gap(outcome)
            if (
                not outcome.sufficiency.sufficient
                and gap
                and max_attempts > 1
                and search_tasks_used < max_tasks
            ):
                missing = next(
                    (
                        reason.split(":", 1)[1].strip()
                        for reason in outcome.sufficiency.reasons
                        if reason.startswith("missing query terms:")
                    ),
                    None,
                )
                focused_query = (
                    f"{current_query}. Focus on passages explicitly mentioning "
                    f"these missing terms and their relationship: {missing}."
                    if missing
                    else current_query
                )
                rewritten_query = focused_query
                if self.query_rewriter is not None:
                    try:
                        rewritten_query = self.query_rewriter.rewrite(focused_query)
                    except Exception:
                        logger.warning(
                            "Follow-up query rewrite failed; using targeted gap query",
                            exc_info=True,
                        )
                current_query = str(rewritten_query).strip() if rewritten_query else focused_query
                decisions.append(
                    AgentDecision(
                        action="rewrite",
                        query=current_query,
                        attempt=2,
                        reason=f"conditional one-hop follow-up for evidence gap: {gap}",
                    )
                )
                outcome = self._search(
                    current_query,
                    metadata_filters,
                    settings.AGENT_INITIAL_TOP_K * 2,
                )
                search_tasks_used += 1
                follow_up_count += 1
                decisions.append(
                    AgentDecision(
                        action="search",
                        query=current_query,
                        attempt=2,
                        reason="follow-up retrieval: "
                        + ("; ".join(outcome.sufficiency.reasons) or "no sufficiency gaps reported"),
                        result_count=len(outcome.chunks),
                        sufficient=outcome.sufficiency.sufficient,
                    )
                )
            elif not outcome.sufficiency.sufficient and gap:
                logger.info(
                    "Skipping follow-up search gap=%s task_budget_used=%s task_budget=%s",
                    gap,
                    search_tasks_used,
                    max_tasks,
                )

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

        total_duration_ms = (time.perf_counter() - run_started) * 1000
        for decision in decisions:
            logger.info(
                "Agent decision action=%s query=%r attempt=%s reason=%s "
                "result_count=%s sufficient=%s",
                decision.action,
                decision.query,
                decision.attempt,
                decision.reason,
                decision.result_count,
                decision.sufficient,
            )
        logger.info(
            "Agent plan completed query_type=%s targets=%s searches=%s follow_ups=%s "
            "task_budget=%s concurrency=%s planning_ms=%.1f duration_ms=%.1f",
            plan.query_type,
            len(targets),
            search_tasks_used,
            follow_up_count,
            max_tasks,
            first_pass_concurrency,
            plan_duration_ms,
            total_duration_ms,
        )
        if run_tree:
            run_tree.metadata.update(
                {
                    "search_tasks_used": search_tasks_used,
                    "follow_up_count": follow_up_count,
                    "total_search_duration_ms": round(total_duration_ms, 2),
                }
            )
        return plan, decisions, results
