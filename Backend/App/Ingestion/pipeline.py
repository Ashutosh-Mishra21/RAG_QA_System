from dataclasses import dataclass
from typing import Dict, List

from .orchestrator import HybridChunker
from .parser import DocumentLoader
from .semantic_chunker import AgenticChunker
from .structural_chunker import RuleBasedChunker


@dataclass
class ChunkingPipeline:
    loader: DocumentLoader
    chunker: HybridChunker

    def load_and_chunk(self, file_path: str) -> List[Dict[str, object]]:
        documents = self.loader.load_pdf(file_path)
        return self.chunker.chunk_documents(documents)


def build_default_pipeline() -> ChunkingPipeline:
    loader = DocumentLoader()
    agentic = AgenticChunker()
    rule_based = RuleBasedChunker()
    hybrid = HybridChunker(agentic_chunker=agentic, rule_chunker=rule_based)
    return ChunkingPipeline(loader=loader, chunker=hybrid)
