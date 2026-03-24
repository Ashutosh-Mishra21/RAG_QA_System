from typing import List, Dict, Tuple
from backend.app.models import Chunk


class ContextBuilder:
    def __init__(self, max_chunks: int = 5):
        self.max_chunks = max_chunks

    def build(self, retrieved_chunks: List[Chunk]) -> Tuple[str, Dict[str, Chunk]]:
        context_blocks = []
        citation_map: Dict[str, Chunk] = {}

        for i, chunk in enumerate(retrieved_chunks[: self.max_chunks]):
            idx = i + 1
            meta = chunk.metadata or {}
            doc_id = meta.get("document_id", "doc") or meta.get("doc_id") or "doc"
            page = meta.get("page_number", "?")
            section = meta.get("section", "")
            citation_header = f"[{idx}] {doc_id} | p{page} | {section}"
            context_blocks.append(f"{citation_header}\n{chunk.content}")
            citation_map[str(idx)] = chunk

        return "\n\n".join(context_blocks), citation_map
