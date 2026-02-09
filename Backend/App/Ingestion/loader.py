from dataclasses import dataclass, field
from typing import Dict, List

from docling.document_converter import DocumentConverter

from .agentic_chunker import AgenticChunker
from .hybrid_chunker import HybridChunker
from .rule_chunker import RuleBasedChunker


@dataclass
class DocumentLoader:
    converter: DocumentConverter = field(default_factory=DocumentConverter)

    def load_pdf(self, file_path: str) -> List[Dict[str, object]]:
        """
        Uses Docling to extract structured content from PDF.
        """
        result = self.converter.convert(file_path)

        documents: List[Dict[str, object]] = []

        for block in result.document.text_blocks:
            documents.append(
                {
                    "text": block.text,
                    "metadata": {
                        "source": file_path,
                        "page": block.page,
                        "role": block.role,  # heading, paragraph, table, etc.
                    },
                }
            )

        return documents


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
