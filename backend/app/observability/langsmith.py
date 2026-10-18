"""
Centralized LangSmith observability utilities.

This module is the ONLY place where application code should need
to know about LangSmith-specific tracing configuration.

Responsibilities:
- Trace configuration
- Trace naming
- Metadata sanitization
- Evidence metadata extraction
- Content capture policy
- Standardized tracing helpers
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from langsmith import traceable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANGSMITH_TRACING_ENABLED = (
    os.getenv("LANGSMITH_TRACING", "false").lower()
    in {"1", "true", "yes", "on"}
)

LANGSMITH_PROJECT = os.getenv(
    "LANGCHAIN_PROJECT",
    os.getenv("LANGSMITH_PROJECT", "rag-agent"),
)

# Maximum amount of text we are willing to send to LangSmith.
MAX_CAPTURE_CHARS = int(
    os.getenv("LANGSMITH_MAX_CAPTURE_CHARS", "4000")
)

# Whether actual document/query/answer content should be captured.
CAPTURE_CONTENT = (
    os.getenv("LANGSMITH_CAPTURE_CONTENT", "true").lower()
    in {"1", "true", "yes", "on"}
)


# ---------------------------------------------------------------------------
# Sensitive metadata
# ---------------------------------------------------------------------------

SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "token",
    "cookie",
    "session",
}


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def _is_sensitive_key(key: str) -> bool:
    """
    Determine whether a metadata key contains sensitive information.
    """
    normalized = key.lower().replace("-", "_")

    return (
        normalized in SENSITIVE_KEYS
        or any(
            sensitive in normalized
            for sensitive in SENSITIVE_KEYS
        )
    )


def sanitize_metadata(
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Remove sensitive values and normalize metadata into a
    LangSmith-friendly dictionary.
    """

    if not metadata:
        return {}

    sanitized: Dict[str, Any] = {}

    for key, value in metadata.items():
        key_str = str(key)

        if _is_sensitive_key(key_str):
            sanitized[key_str] = "[REDACTED]"
            continue

        # Avoid sending huge arbitrary objects to tracing.
        if isinstance(value, str):
            sanitized[key_str] = truncate_text(value)

        elif isinstance(value, (int, float, bool)) or value is None:
            sanitized[key_str] = value

        elif isinstance(value, (list, tuple)):
            sanitized[key_str] = [
                truncate_text(str(item))
                for item in value[:20]
            ]

        elif isinstance(value, Mapping):
            sanitized[key_str] = sanitize_metadata(value)

        else:
            sanitized[key_str] = truncate_text(str(value))

    return sanitized


def truncate_text(
    value: Optional[str],
    max_chars: int = MAX_CAPTURE_CHARS,
) -> Optional[str]:
    """
    Prevent very large text payloads from being sent to LangSmith.
    """

    if value is None:
        return None

    value = str(value)

    if len(value) <= max_chars:
        return value

    return value[:max_chars] + "...[TRUNCATED]"


# ---------------------------------------------------------------------------
# Trace naming
# ---------------------------------------------------------------------------

def trace_name(
    component: str,
    operation: str,
) -> str:
    """
    Generate standardized trace names.

    Example:
        trace_name("retrieval", "vector_search")
        -> "RAG / Retrieval / vector_search"
    """

    component = component.strip().title()
    operation = re.sub(r"\s+", "_", operation.strip())

    return f"RAG / {component} / {operation}"


# ---------------------------------------------------------------------------
# Content capture policy
# ---------------------------------------------------------------------------

def capture_content(
    content: Any,
    *,
    max_chars: int = MAX_CAPTURE_CHARS,
) -> Any:
    """
    Apply the global content capture policy.

    If content capture is disabled, return a lightweight placeholder.
    """

    if not CAPTURE_CONTENT:
        return "[CONTENT_CAPTURE_DISABLED]"

    if content is None:
        return None

    if isinstance(content, str):
        return truncate_text(content, max_chars)

    if isinstance(content, Mapping):
        return {
            str(key): capture_content(value, max_chars=max_chars)
            for key, value in content.items()
        }

    if isinstance(content, (list, tuple)):
        return [
            capture_content(item, max_chars=max_chars)
            for item in content[:20]
        ]

    return truncate_text(str(content), max_chars)


# ---------------------------------------------------------------------------
# Evidence metadata
# ---------------------------------------------------------------------------

def extract_evidence_metadata(
    documents: Optional[Iterable[Any]],
) -> List[Dict[str, Any]]:
    """
    Extract lightweight metadata about retrieved evidence.

    This intentionally avoids sending entire documents unless the
    content capture policy allows it.

    Expected document shape:
        {
            "page_content": "...",
            "metadata": {...}
        }

    or LangChain-style:
        Document(
            page_content="...",
            metadata={...}
        )
    """

    if not documents:
        return []

    evidence: List[Dict[str, Any]] = []

    for index, document in enumerate(documents):

        # LangChain Document
        if hasattr(document, "page_content"):
            content = getattr(document, "page_content", "")
            metadata = getattr(document, "metadata", {}) or {}

        # Dictionary representation
        elif isinstance(document, Mapping):
            content = document.get("page_content", "")
            metadata = document.get("metadata", {}) or {}

        else:
            content = str(document)
            metadata = {}

        evidence_item = {
            "rank": index + 1,
            "source": metadata.get("source"),
            "document_id": metadata.get("document_id"),
            "chunk_id": metadata.get("chunk_id"),
            "page": metadata.get("page"),
            "title": metadata.get("title"),
            "score": metadata.get("score"),
        }

        if CAPTURE_CONTENT:
            evidence_item["content_preview"] = truncate_text(
                content,
                max_chars=500,
            )

        evidence.append(
            sanitize_metadata(evidence_item)
        )

    return evidence


# ---------------------------------------------------------------------------
# Standard metadata builders
# ---------------------------------------------------------------------------

def rag_request_metadata(
    *,
    query: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Standard metadata for a top-level RAG request.
    """

    metadata = {
        "component": "rag",
        "query": capture_content(query, max_chars=2000),
        "user_id": user_id,
        "session_id": session_id,
        **extra,
    }

    return sanitize_metadata(metadata)


def retrieval_metadata(
    *,
    query: Optional[str] = None,
    retriever: Optional[str] = None,
    top_k: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Standard metadata for retrieval traces.
    """

    metadata = {
        "component": "retrieval",
        "query": capture_content(query, max_chars=2000),
        "retriever": retriever,
        "top_k": top_k,
        **extra,
    }

    return sanitize_metadata(metadata)


def generation_metadata(
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Standard metadata for generation traces.
    """

    metadata = {
        "component": "generation",
        "model": model,
        "temperature": temperature,
        **extra,
    }

    return sanitize_metadata(metadata)


# ---------------------------------------------------------------------------
# Traceable wrapper
# ---------------------------------------------------------------------------

def traced(
    *,
    name: str,
    run_type: str = "chain",
):

    def decorator(func):
        if not LANGSMITH_TRACING_ENABLED:
            return func

        return traceable(
            name=name,
            run_type=run_type,
            project_name=LANGSMITH_PROJECT,
        )(func)

    return decorator


# ---------------------------------------------------------------------------
# Public tracing decorators
# ---------------------------------------------------------------------------

def trace_rag_request(func):
    """
    Trace the complete RAG request.
    """

    return traced(
        name=trace_name("RAG", "request"),
        run_type="chain",
    )(func)


def trace_agent(func):
    """
    Trace agent execution.
    """

    return traced(
        name=trace_name("Agent", "execution"),
        run_type="chain",
    )(func)


def trace_tool(func):
    """
    Trace tool execution.
    """

    return traced(
        name=trace_name("Tool", func.__name__),
        run_type="tool",
    )(func)


def trace_retrieval(func):
    """
    Trace retrieval.
    """

    return traced(
        name=trace_name("Retrieval", func.__name__),
        run_type="retriever",
    )(func)


def trace_generation(func):
    """
    Trace LLM/answer generation.
    """

    return traced(
        name=trace_name("Generation", func.__name__),
        run_type="llm",
    )(func)


def trace_evaluation(func):
    """
    Trace evaluation execution.
    """

    return traced(
        name=trace_name("Evaluation", func.__name__),
        run_type="chain",
    )(func)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def trace_context(
    *,
    name: str,
    run_type: str = "chain",
    metadata: Optional[Mapping[str, Any]] = None,
    inputs: Optional[Mapping[str, Any]] = None,
) -> Iterator[None]:

    if not LANGSMITH_TRACING_ENABLED:
        yield
        return

    from langsmith import trace

    with trace(
        name=name,
        run_type=run_type,
        project_name=LANGSMITH_PROJECT,
        metadata=sanitize_metadata(metadata),
        inputs=capture_content(inputs),
    ):
        yield


# ---------------------------------------------------------------------------
# Explicit trace helpers
# ---------------------------------------------------------------------------

def trace_retrieval_result(
    documents: Iterable[Any],
) -> Dict[str, Any]:
    """
    Convert retrieved documents into safe tracing metadata.
    """

    evidence = extract_evidence_metadata(documents)

    return {
        "num_results": len(evidence),
        "evidence": evidence,
    }


def trace_generation_result(
    answer: str,
) -> Dict[str, Any]:
    """
    Convert generated answer into safe tracing output.
    """

    return {
        "answer": capture_content(answer),
    }


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def configure_langsmith() -> None:
    """
    Validate LangSmith configuration.

    LangSmith reads LANGCHAIN_API_KEY / LANGSMITH_API_KEY and
    project configuration from environment variables.

    This function intentionally does not make network calls.
    """

    if not LANGSMITH_TRACING_ENABLED:
        return

    api_key = (
        os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "LangSmith tracing is enabled, but no API key was found. "
            "Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY."
        )


__all__ = [
    "LANGSMITH_TRACING_ENABLED",
    "LANGSMITH_PROJECT",
    "configure_langsmith",
    "sanitize_metadata",
    "truncate_text",
    "capture_content",
    "extract_evidence_metadata",
    "trace_name",
    "rag_request_metadata",
    "retrieval_metadata",
    "generation_metadata",
    "traced",
    "trace_rag_request",
    "trace_agent",
    "trace_tool",
    "trace_retrieval",
    "trace_generation",
    "trace_evaluation",
    "trace_context",
    "trace_retrieval_result",
    "trace_generation_result",
]