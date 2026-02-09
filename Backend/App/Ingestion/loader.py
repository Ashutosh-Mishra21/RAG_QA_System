from docling.document_converter import DocumentConverter
from typing import List
from .agentic_chunker import agentic_chunk


def load_pdf_with_docling(file_path: str) -> List[dict]:
    """
    Uses Docling to extract structured content from PDF.
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)

    documents = []

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


def agentic_chunk_documents(documents: List[dict]) -> List[dict]:
    final_chunks = []

    for doc in documents:
        chunks = agentic_chunk(doc["text"])

        for idx, chunk in enumerate(chunks):
            final_chunks.append(
                {
                    "text": chunk,
                    "metadata": {**doc["metadata"], "chunk_id": f"{doc['metadata']['page']}_{idx}"},
                }
            )

    return final_chunks
