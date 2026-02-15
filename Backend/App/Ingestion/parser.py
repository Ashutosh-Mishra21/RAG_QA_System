from dataclasses import dataclass, field
from typing import Dict, List

from docling.document_converter import DocumentConverter

from ..indexing.embedder import SentenceTransformerEmbedder


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
