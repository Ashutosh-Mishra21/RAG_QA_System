from dataclasses import dataclass
from typing import Optional, Dict

from torch.backends.opt_einsum import strategy


@dataclass
class RetrievalStrategy:
    top_k: int
    use_rerank: bool
    metadata_filter: Optional[Dict]
    query_type: str


class QueryAnalyzer:
    def analyze(self, query: str) -> RetrievalStrategy:
        # Placeholder implementation, replace with actual analysis logic
        q = query.lower()
        tokens = q.split()

        strategy = RetrievalStrategy(
            top_k=5, use_rerank=False, metadata_filter=None, query_type="general"
        )

        # Long Query -> increase candidate pool
        if len(tokens) > 8:
            strategy.top_k = 8

        # Definition-style queries → rerank
        if q.startswith("what is") or q.startswith("define"):
            strategy.use_rerank = True
            strategy.query_type = "definition"

        # Explainaion query
        if "explain" in q or "in detail" in q:
            strategy.use_rerank = True
            strategy.top_k = 10
            strategy.query_type = "explanation"

        # Metadata Filtering
        if "figure" in q:
            strategy.metadata_filter = {"type": "figure"}
        elif "table" in q:
            strategy.metadata_filter = {"type": "table"}

        return strategy
