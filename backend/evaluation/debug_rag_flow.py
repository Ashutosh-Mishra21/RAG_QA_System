"""Trace upload, retrieval, reranking, prompt context, generation, and validation."""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".docx", ".xlsx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="append",
        type=Path,
        default=[],
        help="Optional document to upload and index before asking questions. Repeatable.",
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        required=True,
        help="Question to trace. Repeat this option for more than one question.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the complete assembled context sent to the answer model.",
    )
    parser.add_argument(
        "--bypass-llm-cache",
        action="store_true",
        help="Ignore and do not write the prompt-level LLM cache for this run.",
    )
    parser.add_argument(
        "--chunks-per-search",
        type=int,
        default=5,
        help="Number of chunks to display for each agent search (default: 5).",
    )
    return parser.parse_args()


def _resolve_upload(path: Path) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Upload file not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported upload type: {resolved.suffix}")
    return resolved


def _excerpt(value: str, limit: int = 450) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + " …"


def _source_line(chunk: Any, rank: int, limit: int = 450) -> str:
    metadata = chunk.metadata or {}
    source_file = metadata.get("source_file") or metadata.get("title") or "?"
    section = metadata.get("section") or metadata.get("subsection") or "?"
    page = metadata.get("page_number") or metadata.get("page_id") or "?"
    return (
        f"#{rank} score={chunk.score!r} doc={metadata.get('document_id', '?')} "
        f"file={source_file!r} page={page} section={section!r} "
        f"chunk={chunk.id}\n    {_excerpt(chunk.content, limit)}"
    )


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _prompt_kind(prompt: str) -> str:
    normalized = " ".join(prompt.split())
    if prompt.lstrip().startswith("Break the following query"):
        return "decomposition"
    if "alternative search queries" in prompt.lower():
        return "query rewrite"
    if prompt.lstrip().startswith("You are a retrieval-augmented"):
        return "answer generation"
    return normalized[:90]


def _instrument_llm_cache(router: Any, bypass: bool) -> dict[str, int]:
    cache = router.cache
    original_get = cache.get
    original_set = cache.set
    totals = {"hits": 0, "misses": 0, "bypassed": 0, "writes": 0}

    def traced_get(prompt: str, model: str = ""):
        if bypass:
            totals["bypassed"] += 1
            print(f"[LLM CACHE BYPASS] model={model}; kind={_prompt_kind(prompt)}")
            return None
        value = original_get(prompt, model=model)
        result = "HIT" if value else "MISS"
        totals["hits" if value else "misses"] += 1
        print(f"[LLM CACHE {result}] model={model}; kind={_prompt_kind(prompt)}")
        return value

    def traced_set(
        prompt: str,
        response: str,
        provider: str,
        model: str = "",
        cache_model: str | None = None,
    ):
        if bypass:
            return
        totals["writes"] += 1
        print(f"[LLM CACHE WRITE] provider={provider}; model={model}")
        original_set(
            prompt,
            response,
            provider=provider,
            model=model,
            cache_model=cache_model,
        )

    cache.get = traced_get
    cache.set = traced_set
    return totals


def _print_searches(searches: list[dict], chunks_per_search: int) -> list[Any]:
    candidates = []
    for number, search in enumerate(searches, start=1):
        sufficiency = search["sufficiency"]
        chunks = search["chunks"]
        print(f"\n[AGENT SEARCH {number}] query={search['query']!r}")
        print(
            "  sufficiency="
            f"{sufficiency.sufficient} score={sufficiency.score:.4f} "
            f"evidence_count={sufficiency.evidence_count} "
            f"coverage={sufficiency.covered_query_terms}/{sufficiency.query_terms} "
            f"reasons={sufficiency.reasons}"
        )
        print(f"  returned_chunks={len(chunks)}")
        for rank, chunk in enumerate(chunks[:chunks_per_search], start=1):
            print("  " + _source_line(chunk, rank).replace("\n", "\n  "))
        candidates.extend(chunks)
    return candidates


def _unique_candidates(chunks: list[Any]) -> list[Any]:
    unique = []
    seen = set()
    for chunk in chunks:
        key = getattr(chunk, "id", None) or (
            (chunk.metadata or {}).get("document_id"),
            (chunk.metadata or {}).get("page_id"),
            chunk.content,
        )
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def _run_question(
    rag: Any,
    question: str,
    args: argparse.Namespace,
    llm_cache_totals: dict[str, int],
) -> None:
    filters = {"tenant_id": "default", "workspace_id": "default"}
    print("\n" + "=" * 100)
    print(f"QUESTION: {question}")
    print("=" * 100)

    cached_response = rag.response_cache.get(question, None, filters)
    if cached_response:
        print(
            "[RAG RESPONSE CACHE HIT] The normal chat endpoint would return this "
            "without retrieval. Diagnostic run will continue past it."
        )
        print(f"  cached_answer={_excerpt(cached_response.get('answer', ''), 250)!r}")
    else:
        print("[RAG RESPONSE CACHE MISS]")

    plan, decisions, searches = rag.agent.run(question, filters)
    print(
        f"\n[PLAN] type={plan.query_type} subqueries={len(plan.subqueries)} "
        f"actions={plan.actions} decisions={len(decisions)}"
    )
    for decision in decisions:
        print(
            f"  action={decision.action} attempt={decision.attempt} "
            f"results={decision.result_count} sufficient={decision.sufficient} "
            f"query={decision.query!r}"
        )

    raw_candidates = _print_searches(searches, args.chunks_per_search)
    candidates = _unique_candidates(raw_candidates)
    candidate_documents = Counter(
        (chunk.metadata or {}).get("document_id", "?") for chunk in candidates
    )
    print(f"\n[UNION] raw={len(raw_candidates)} unique={len(candidates)}")
    print(f"  candidates_by_document={dict(candidate_documents)}")

    if not candidates:
        print("[RESULT] No chunks reached generation; expected grounded refusal.")
        return

    reranked = rag.pipeline.reranker.rerank(question, candidates)
    top_chunks = reranked[:5]
    top_documents = Counter(
        (chunk.metadata or {}).get("document_id", "?") for chunk in top_chunks
    )
    print(f"\n[RERANK] input={len(candidates)} selected={len(top_chunks)}")
    print(f"  selected_by_document={dict(top_documents)}")
    for rank, chunk in enumerate(top_chunks, start=1):
        print(_source_line(chunk, rank))

    context, citation_map = rag.pipeline.context_builder.build(top_chunks)
    router = rag.pipeline.generator.llm_router
    expected_provider = "api" if router.primary else "local"
    prompt = rag.pipeline.prompt_builder.build_prompt(
        question,
        context,
        expected_provider,
    )
    print(f"\n[PROMPT] expected_provider={expected_provider}; chars={len(prompt)}")
    if args.show_context:
        print("----- EXACT RETRIEVED CONTEXT SENT TO GENERATOR -----")
        print(context or "(empty context)")
        print("----- END CONTEXT -----")
    else:
        print(
            "  Use --show-context to print the complete exact context. "
            "The five chunk excerpts above show what was selected."
        )

    print("\n[GENERATION] streaming tokens:", flush=True)
    answer_parts = []
    used_model = None
    provider = None
    for text, used_model, provider in rag.pipeline.generator.stream(prompt):
        answer_parts.append(text)
        print(text, end="", flush=True)
    print("\n[GENERATION COMPLETE]")

    answer = "".join(answer_parts)
    validation = rag.pipeline.validator.validate(answer, citation_map, top_chunks)
    print(f"  provider={provider}; model={used_model}")
    print(f"  fallback_triggered={expected_provider == 'api' and provider == 'local'}")
    print(f"  answer={answer!r}")
    print(f"  citations={validation['citations']}")
    print(f"  confidence={validation['confidence']}; valid={validation['valid']}")
    print(f"  validation_reason={validation['reason']}")
    print(f"  prompt_level_cache_totals={llm_cache_totals}")


def main() -> int:
    args = parse_args()
    if args.chunks_per_search < 1 or args.chunks_per_search > 20:
        print("--chunks-per-search must be between 1 and 20.", file=sys.stderr)
        return 2

    try:
        uploads = [_resolve_upload(path) for path in args.upload]
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.api.dependencies import get_rag_service
    from backend.app.core.config import settings

    print("RAG flow debug run")
    print(f"working_directory={Path.cwd()}")
    print(f"qdrant_collection={settings.QDRANT_COLLECTION}")
    print(f"bm25_index={Path(settings.BM25_INDEX_PATH).resolve()}")
    print(f"upload_count={len(uploads)}")
    print(f"llm_cache_mode={'BYPASS' if args.bypass_llm_cache else 'RESPECT'}")
    print("Stop the separately running Uvicorn process before this run.", flush=True)

    try:
        with TestClient(app) as client:
            health_response = client.get("/api/health")
            health_response.raise_for_status()
            health = health_response.json().get("data", {})
            print(f"health={health}")
            if not health.get("services", {}).get("qdrant"):
                print("Qdrant is unavailable; stopping diagnostic run.", file=sys.stderr)
                return 1

            for upload in uploads:
                print(f"\n[UPLOAD] {upload.name}", flush=True)
                mime_type = mimetypes.guess_type(upload.name)[0] or "application/octet-stream"
                with upload.open("rb") as stream:
                    response = client.post(
                        "/api/upload",
                        files={"file": (upload.name, stream, mime_type)},
                    )
                response.raise_for_status()
                payload = response.json()
                if payload.get("success") is not True:
                    raise RuntimeError(f"Upload failed: {payload}")
                data = payload["data"]
                print(
                    "  upload_succeeded="
                    f"document_id={data['document']['document_id']} "
                    f"chunks_indexed={data['pipeline'].get('chunks_indexed')}"
                )

            rag = get_rag_service()
            router = rag.pipeline.generator.llm_router
            cache_totals = _instrument_llm_cache(router, args.bypass_llm_cache)

            embed_cache = ROOT / "data/embeddings/embedding_cache.json"
            evidence_cache = ROOT / "data/processed/evidence_cache"
            bm25 = Path(settings.BM25_INDEX_PATH)
            print("\n[CACHE LOCATIONS]")
            print(f"  response_cache={rag.response_cache.cache_dir}")
            print(f"    files={_count_files(rag.response_cache.cache_dir)}")
            print(f"  llm_cache={router.cache.cache_dir}")
            print(f"    files={_count_files(router.cache.cache_dir)}")
            print(f"  embedding_cache={embed_cache} exists={embed_cache.is_file()}")
            print(
                f"  evidence_cache={evidence_cache} files={_count_files(evidence_cache)}"
            )
            print(
                f"  bm25_index={bm25} exists={bm25.is_file()} "
                f"in_memory_documents={len(rag.retriever.keyword.documents)}"
            )
            print(
                "  This runner reports cache state but does not clear the response, "
                "embedding, evidence, or BM25 caches."
            )

            for question in args.questions:
                _run_question(rag, question, args, cache_totals)

            print("\nDiagnostic run finished.")
            print(f"final_prompt_level_cache_totals={cache_totals}")
        return 0
    except Exception as exc:
        print(f"Diagnostic run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
