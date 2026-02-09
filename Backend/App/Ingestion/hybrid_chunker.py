from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .agentic_chunker import AgenticChunker
from .rule_chunker import RuleBasedChunker


def should_use_agentic(text: str, role: str) -> bool:
    token_estimate = len(text.split())

    if token_estimate < 300:
        return False

    if role in ["heading", "title"]:
        return False

    policy_keywords = [
        "shall",
        "must",
        "policy",
        "procedure",
        "entitled",
        "eligible",
        "exception",
    ]

    if any(word in text.lower() for word in policy_keywords):
        return True

    return token_estimate > 600


@dataclass
class HybridChunker:
    agentic_chunker: AgenticChunker
    rule_chunker: RuleBasedChunker

    @staticmethod
    def _chunk_id(metadata: Dict[str, str], idx: int) -> str:
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "na")
        return f"{Path(source).name}_{page}_{idx}"

    def chunk_documents(
        self, documents: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        final_chunks: List[Dict[str, object]] = []

        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]
            role = metadata.get("role", "paragraph")

            if should_use_agentic(text, role):
                chunks = self.agentic_chunker.chunk(text)
                method = "agentic"
            else:
                chunks = self.rule_chunker.chunk(text)
                method = "rule"

            for idx, chunk in enumerate(chunks):
                final_chunks.append(
                    {
                        "text": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_id": self._chunk_id(metadata, idx),
                            "chunking_method": method,
                        },
                    }
                )

        return final_chunks
