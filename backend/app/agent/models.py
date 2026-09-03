from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from backend.app.models import Chunk
    from backend.app.retrieval import EvidenceSufficiency

AgentAction = Literal["search", "rewrite", "decompose", "parse_table", "parse_visual"]


@dataclass
class AgentPlan:
    query: str
    query_type: str
    subqueries: list[str] = field(default_factory=list)
    actions: list[AgentAction] = field(default_factory=list)


@dataclass
class AgentDecision:
    action: AgentAction
    query: str
    attempt: int
    reason: str
    result_count: int = 0
    sufficient: bool | None = None


@dataclass
class AgentSearchOutcome:
    query: str
    chunks: list[Chunk]
    sufficiency: EvidenceSufficiency
