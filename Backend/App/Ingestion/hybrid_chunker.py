from pathlib import Path
from .rule_chunker import rule_based_chunk
from .agentic_chunker import agentic_chunk


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


def _chunk_id(metadata: dict, idx: int) -> str:
    source = metadata.get("source", "unknown")
    page = metadata.get("page", "na")
    return f"{Path(source).name}_{page}_{idx}"


def hybrid_chunk_documents(documents: list[dict]) -> list[dict]:
    final_chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        role = metadata.get("role", "paragraph")

        if should_use_agentic(text, role):
            chunks = agentic_chunk(text)
            method = "agentic"
        else:
            chunks = rule_based_chunk(text)
            method = "rule"

        for idx, chunk in enumerate(chunks):
            final_chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_id": _chunk_id(metadata, idx),
                        "chunking_method": method,
                    },
                }
            )

    return final_chunks
