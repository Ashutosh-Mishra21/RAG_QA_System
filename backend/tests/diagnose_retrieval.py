"""Interactive retrieval diagnostics; this is a script, not a pytest test.

Run from the repository root, with Uvicorn stopped:

    python -m backend.tests.diagnose_retrieval --question "..." --show-context

Use repeatable --expected-term values to inspect whether known answer concepts
survive each retrieval stage. This script reads cache state but never clears it.
It does not upload, index, or delete documents.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", action="append", help="Question to diagnose; repeatable")
    parser.add_argument(
        "--case-file", type=Path,
        help="JSON list of {question, expected_terms, document_id?} cases",
    )
    parser.add_argument(
        "--expected-term",
        action="append",
        default=[],
        help="Answer-bearing concept to check at each stage; repeatable",
    )
    parser.add_argument("--document-id", help="Limit search to one indexed document")
    parser.add_argument("--top-k", type=int, default=10, help="Per modality; 1-50")
    parser.add_argument(
        "--chunks-per-search", type=int, default=10,
        help="How many agent search hits to print (1-50)",
    )
    parser.add_argument(
        "--show-context", action="store_true",
        help="Print the exact context assembled for the answer model",
    )
    parser.add_argument(
        "--skip-agent", action="store_true",
        help="Skip LLM query planning and inspect only direct dense/sparse retrieval",
    )
    parser.add_argument(
        "--log-file", type=Path,
        help="Save complete stdout/stderr and application logs here (default: timestamped backend/logs file)",
    )
    return parser.parse_args()


def excerpt(text: str, limit: int = 420) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[:limit].rstrip() + " …"


def terms(text: str) -> set[str]:
    return {term.casefold() for term in re.findall(r"\b[\w-]{2,}\b", text)}


def meta(chunk: Any) -> dict[str, Any]:
    return chunk.metadata or {}


def key(chunk: Any) -> str:
    return str(getattr(chunk, "id", "")) or repr(
        (meta(chunk).get("document_id"), meta(chunk).get("page_id"), chunk.content)
    )


def print_hits(label: str, chunks: list[Any], expected: list[str], limit: int) -> None:
    print(f"\n[{label}] count={len(chunks)}")
    for rank, chunk in enumerate(chunks[:limit], 1):
        md = meta(chunk)
        content_terms = terms(chunk.content)
        matched = [word for word in expected if terms(word) & content_terms]
        print(
            f"  #{rank} score={chunk.score!r} id={key(chunk)} "
            f"doc={md.get('document_id', '?')} page={md.get('page_number', '?')} "
            f"section={md.get('section') or md.get('heading') or '?'} "
            f"expected_terms={matched or '-'}"
        )
        print(f"      {excerpt(chunk.content)}")


def doc_counts(chunks: list[Any]) -> dict[str, int]:
    return dict(Counter(str(meta(c).get("document_id", "?")) for c in chunks))


def explain_sufficiency(query: str, chunks: list[Any], scorer: Any) -> None:
    evidence = [scorer.evidence_from_chunk(chunk) for chunk in chunks]
    result = scorer.sufficiency.score(query, evidence)
    query_terms = scorer.sufficiency._terms(query)
    content_terms = scorer.sufficiency._terms(
        " ".join(item.text_content() for item in evidence if item.text_content().strip())
    )
    covered = sorted(query_terms & content_terms)
    valid = [item for item in evidence if item.text_content().strip()]
    provenance = (
        sum(bool(i.document_id) and bool(i.page_id) and i.provenance.source_page >= 1 for i in valid)
        / len(valid) if valid else 0.0
    )
    pages = len({item.page_id for item in valid})
    count_score = min(len(valid) / 3.0, 1.0)
    diversity = min(pages / 3.0, 1.0)
    coverage = len(covered) / len(query_terms) if query_terms else 1.0
    print("\n[SUFFICIENCY BREAKDOWN]")
    print(f"  sufficient={result.sufficient} score={result.score:.4f} threshold={scorer.sufficiency.threshold}")
    print(f"  reasons={result.reasons}")
    print(f"  query_terms={sorted(query_terms)}")
    print(f"  covered_terms={covered} ({len(covered)}/{len(query_terms)}) coverage={coverage:.3f}")
    print(f"  evidence_count={len(valid)} provenance={provenance:.3f} distinct_pages={pages}")
    print(f"  formula: .35*coverage({coverage:.3f}) + .30*provenance({provenance:.3f}) + "
          f".20*page_diversity({diversity:.3f}) + .15*count({count_score:.3f})")
    print("  Note: this score measures term overlap/provenance/page count, not whether the answer is present.")


def traced_hybrid_retrieve(
    retriever: Any,
    query: str,
    expected: list[str],
    top_k: int,
    filters: dict[str, Any],
) -> list[Any]:
    """Trace internal retrieval stages for one call, restoring methods afterward."""
    originals = {
        "dense": retriever.dense.retrieve,
        "sparse": retriever.keyword.retrieve,
        "weights": retriever._dynamic_weights,
        "fusion": retriever._weighted_query_rrf,
        "mmr": retriever._mmr,
    }
    query_rankings: list[tuple[str, str, list[Any]]] = []
    current_weights = (0.6, 0.4)

    def traced_dense(*args, **kwargs):
        result = originals["dense"](*args, **kwargs)
        searched_query = kwargs.get("query", args[0] if args else query)
        query_rankings.append(("DENSE", str(searched_query), result))
        print_hits(f"HYBRID INTERNAL / DENSE query={searched_query!r}", result, expected, len(result))
        return result

    def traced_sparse(*args, **kwargs):
        result = originals["sparse"](*args, **kwargs)
        searched_query = kwargs.get("query", args[0] if args else query)
        query_rankings.append(("BM25", str(searched_query), result))
        print_hits(f"HYBRID INTERNAL / BM25 query={searched_query!r}", result, expected, len(result))
        return result

    def traced_weights(*args, **kwargs):
        nonlocal current_weights
        result = originals["weights"](*args, **kwargs)
        current_weights = result
        print(f"\n[HYBRID WEIGHTS] query={args[0] if args else query!r} dense={result[0]:.3f} bm25={result[1]:.3f}")
        return result

    def traced_fusion(*args, **kwargs):
        dense_rankings = kwargs.get("dense_rankings", args[0] if args else [])
        sparse_rankings = kwargs.get("keyword_rankings", args[1] if len(args) > 1 else [])
        result = originals["fusion"](*args, **kwargs)
        dense_weight, sparse_weight = current_weights
        contributions: dict[str, list[str]] = {}
        raw_totals: dict[str, float] = {}
        dense_queries = [q for modality, q, _ in query_rankings if modality == "DENSE"]
        sparse_queries = [q for modality, q, _ in query_rankings if modality == "BM25"]
        for modality, query_rankings_for_mode, modality_weight, query_names in (
            ("DENSE", dense_rankings, dense_weight, dense_queries),
            ("BM25", sparse_rankings, sparse_weight, sparse_queries),
        ):
            for query_index, (query_weight, ranked_chunks) in enumerate(query_rankings_for_mode):
                query_name = query_names[query_index] if query_index < len(query_names) else f"variant-{query_index + 1}"
                for rank, chunk in enumerate(ranked_chunks, 1):
                    contribution = modality_weight * query_weight / (retriever.rrf_k + rank)
                    contributions.setdefault(key(chunk), []).append(
                        f"{modality}[{query_name!r}] rank={rank} "
                        f"modality_w={modality_weight:.3f} query_w={query_weight:.3f} "
                        f"rrf={contribution:.7f}"
                    )
                    raw_totals[key(chunk)] = raw_totals.get(key(chunk), 0.0) + contribution

        print("\n[WEIGHTED RRF OUTPUT / BEFORE TOP-30 TRUNCATION]")
        print(f"  dense_weight={dense_weight:.3f} bm25_weight={sparse_weight:.3f} "
              f"rrf_k={retriever.rrf_k} fused_unique={len(result)}")
        for rank, chunk in enumerate(result, 1):
            factor = retriever._importance_weight(meta(chunk))
            print(f"  FUSED rank={rank} id={key(chunk)} raw_rrf={raw_totals.get(key(chunk), 0.0):.7f} "
                  f"importance={factor:.3f} final_rrf={chunk.score:.7f} "
                  f"expected_terms={[term for term in expected if terms(term) & terms(chunk.content)] or '-'}")
            for detail in contributions.get(key(chunk), []):
                print(f"      + {detail}")
            print(f"      {excerpt(chunk.content, 240)}")
        return result

    def traced_mmr(*args, **kwargs):
        candidates = kwargs.get("chunks", args[0] if args else [])
        requested_k = kwargs.get("k", args[1] if len(args) > 1 else 5)
        lambda_param = kwargs.get("lambda_param")
        lambda_param = retriever.mmr_lambda if lambda_param is None else lambda_param
        lambda_param = min(1.0, max(0.0, lambda_param))
        actual_pool_size = top_k * 3
        print("\n[TOP-30 TRUNCATION / ACTUAL MMR INPUT]")
        print(f"  fused_candidates={len(candidates)} top_k={top_k} pool_cap={actual_pool_size} "
              f"candidates_after_truncation={len(candidates)} (MMR sees this list)")
        print_hits("CANDIDATES AFTER FUSED[:top_k*3]", candidates, expected, len(candidates))

        # Mirror the production MMR scoring loop to expose every candidate's
        # relevance, maximum cosine similarity to selected chunks, and score.
        ordered = sorted(candidates, key=lambda chunk: (-(chunk.score or 0.0), chunk.id))
        last_rank = max(len(ordered) - 1, 1)
        relevance_by_id = {
            chunk.id: 1.0 - rank / last_rank
            for rank, chunk in enumerate(ordered)
        }
        rrf_rank_by_id = {chunk.id: rank for rank, chunk in enumerate(ordered, 1)}
        selected_trace = [ordered[0]] if ordered else []
        remaining = ordered[1:]
        if selected_trace:
            seed = selected_trace[0]
            print(f"\n[MMR DECISION 1] seed={key(seed)} rrf_rank=1 rrf_score={seed.score or 0.0:.7f} "
                  f"normalized_relevance={relevance_by_id[seed.id]:.7f} max_cosine_similarity=None mmr_score=None selected=True")
        decision_number = 2
        while remaining and len(selected_trace) < requested_k:
            rows = []
            for chunk in remaining:
                relevance = relevance_by_id[chunk.id]
                diversity = max((retriever._similarity(chunk, chosen) for chosen in selected_trace), default=0.0)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                rows.append((mmr_score, chunk.id, chunk, relevance, diversity))
            # Production breaks exact score ties by choosing the lexicographically smallest id.
            best_score = max(row[0] for row in rows)
            winner = min((row for row in rows if row[0] == best_score), key=lambda row: row[1])
            print(f"\n[MMR DECISION {decision_number}] lambda={lambda_param:.3f} "
                  f"formula=lambda*relevance-(1-lambda)*max_cosine_similarity")
            for mmr_score, _, chunk, relevance, diversity in sorted(rows, key=lambda row: (-row[0], row[1])):
                marker = " SELECTED" if chunk.id == winner[2].id else ""
                print(f"  id={key(chunk)} rrf_rank={rrf_rank_by_id[chunk.id]} "
                      f"rrf_score={chunk.score or 0.0:.7f} normalized_relevance={relevance:.7f} "
                      f"max_cosine_similarity={diversity:.7f} mmr_score={mmr_score:.7f} "
                      f"selected={chunk.id == winner[2].id}{marker} expected_terms="
                      f"{[term for term in expected if terms(term) & terms(chunk.content)] or '-'}")
            selected_trace.append(winner[2])
            remaining = [chunk for chunk in remaining if chunk.id != winner[2].id]
            decision_number += 1

        selected = originals["mmr"](*args, **kwargs)
        print("\n[MMR TRACE VALIDATION]")
        print(f"  simulated_ids={[key(c) for c in selected_trace]}")
        print(f"  production_ids={[key(c) for c in selected]}")
        print(f"  trace_matches_production={[key(c) for c in selected_trace] == [key(c) for c in selected]}")
        selected_terms = {
            term for term in expected
            if any(terms(term) & terms(item.content) for item in selected)
        }
        lost = {
            term for term in expected
            if any(terms(term) & terms(item.content) for item in candidates)
            and term not in selected_terms
        }
        print("\n[MMR SELECTION / FINAL SELECTED CANDIDATES]")
        print(f"  candidates_in={len(candidates)} selected={len(selected)} selected_expected_terms={sorted(selected_terms)}")
        if lost:
            print(f"  MMR_DROPPED_EXPECTED_TERMS={sorted(lost)}")
        print_hits("MMR SELECTED", selected, expected, len(selected))
        print("  selected_candidate_ids=" + repr([key(item) for item in selected]))
        return selected

    try:
        retriever.dense.retrieve = traced_dense
        retriever.keyword.retrieve = traced_sparse
        retriever._dynamic_weights = traced_weights
        retriever._weighted_query_rrf = traced_fusion
        retriever._mmr = traced_mmr
        result = retriever.retrieve(
            query,
            top_k=top_k,
            metadata_filters=filters,
        )
        print("\n[HYBRID TRACE SUMMARY]")
        print(f"  modality_calls={len(query_rankings)}")
        for modality, searched_query, chunks in query_rankings:
            found = sorted({
                term for term in expected
                if any(terms(term) & terms(chunk.content) for chunk in chunks)
            })
            print(f"  {modality} query={searched_query!r} result_count={len(chunks)} expected_terms={found}")
        return result
    finally:
        retriever.dense.retrieve = originals["dense"]
        retriever.keyword.retrieve = originals["sparse"]
        retriever._dynamic_weights = originals["weights"]
        retriever._weighted_query_rrf = originals["fusion"]
        retriever._mmr = originals["mmr"]


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if args.case_file:
        path = args.case_file if args.case_file.is_absolute() else ROOT / args.case_file
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("case file must contain a JSON list")
        cases.extend(payload)
    for question in args.question or []:
        cases.append({"question": question, "expected_terms": args.expected_term})
    if not cases:
        raise ValueError("provide --question or --case-file")
    for i, case in enumerate(cases, 1):
        if not isinstance(case, dict) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"case {i} requires a non-empty question")
        if not isinstance(case.get("expected_terms", []), list):
            raise ValueError(f"case {i} expected_terms must be a list")
    return cases


def diagnose_case(rag: Any, case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    question = case["question"]
    expected = [str(term) for term in case.get("expected_terms", args.expected_term)]
    doc_id = case.get("document_id") or args.document_id
    filters = {"tenant_id": "default", "workspace_id": "default"}
    if doc_id:
        filters["document_id"] = doc_id
    print(
        "\n" + "=" * 96
        + f"\nCASE: {question}\nexpected_terms={expected} "
        + f"expected_answerable={case.get('expected_answerable', 'unspecified')} "
        + f"filters={filters}\n"
        + "=" * 96
    )

    cached = rag.response_cache.get(question, doc_id, filters)
    print("\n[RESPONSE CACHE] " + ("HIT" if cached else "MISS"))
    if cached:
        print(f"  cached_answer={excerpt(str(cached.get('answer', '')), 250)!r}; diagnostic bypasses it")

    retriever = rag.retriever
    dense = retriever.dense.retrieve(question, top_k=args.top_k, metadata_filters=filters)
    sparse = retriever.keyword.retrieve(question, top_k=args.top_k, metadata_filters=filters)
    print_hits("DENSE (Qdrant)", dense, expected, args.chunks_per_search)
    print_hits("SPARSE (BM25)", sparse, expected, args.chunks_per_search)
    dense_ids, sparse_ids = {key(c) for c in dense}, {key(c) for c in sparse}
    print(f"\n[MODALITY OVERLAP] dense_only={len(dense_ids-sparse_ids)} sparse_only={len(sparse_ids-dense_ids)} both={len(dense_ids&sparse_ids)}")

    present_dense = {term for term in expected if any(terms(term) & terms(c.content) for c in dense)}
    present_sparse = {term for term in expected if any(terms(term) & terms(c.content) for c in sparse)}
    summary = {"question": question, "expected": expected, "dense": present_dense, "sparse": present_sparse}
    print("\n" + "-" * 88 + "\n[INTERNAL HYBRID RETRIEVER TRACE]")
    hybrid_result = traced_hybrid_retrieve(
        retriever, question, expected, args.top_k, filters
    )
    present_internal_hybrid = {
        term for term in expected
        if any(terms(term) & terms(chunk.content) for chunk in hybrid_result)
    }
    print(f"\n[HYBRID RETURN VALUE] count={len(hybrid_result)} expected_terms={sorted(present_internal_hybrid)}")
    summary["hybrid_retriever"] = present_internal_hybrid
    if args.skip_agent:
        print("\n[AGENT] skipped")
        return summary

    plan, decisions, searches = rag.agent.run(question, filters)
    print(f"\n[AGENT PLAN] type={plan.query_type} actions={plan.actions} subqueries={plan.subqueries}")
    for d in decisions:
        print(f"  {d.action} attempt={d.attempt} count={d.result_count} sufficient={d.sufficient} query={d.query!r} reason={d.reason}")
    all_chunks = []
    for n, search in enumerate(searches, 1):
        chunks = search["chunks"]
        print_hits(f"AGENT/HYBRID SEARCH {n}: {search['query']!r}", chunks, expected, args.chunks_per_search)
        explain_sufficiency(search["query"], chunks, rag.evidence_tools)
        all_chunks.extend(chunks)
    unique, seen = [], set()
    for chunk in all_chunks:
        if key(chunk) not in seen:
            seen.add(key(chunk)); unique.append(chunk)
    print(f"\n[AGENT UNION] raw={len(all_chunks)} unique={len(unique)} by_document={doc_counts(unique)}")
    print_hits("ALL AGENT-UNION CANDIDATES", unique, expected, len(unique))
    present_hybrid = {term for term in expected if any(terms(term) & terms(c.content) for c in unique)}
    summary["hybrid"] = present_hybrid
    summary["reranked"] = set()
    summary["final"] = set()
    if unique:
        print_hits("CROSS-ENCODER INPUT (agent-union order)", unique, expected, len(unique))
        reranked = rag.pipeline.reranker.rerank(question, unique)
        print_hits("CROSS-ENCODER OUTPUT / RERANKED ORDER (raw scores, not probabilities)", reranked, expected, len(reranked))
        final_chunks = reranked[:5]
        present_reranked = {term for term in expected if any(terms(term) & terms(c.content) for c in reranked)}
        present_final = {term for term in expected if any(terms(term) & terms(c.content) for c in final_chunks)}
        summary["reranked"], summary["final"] = present_reranked, present_final
        context, citation_map = rag.pipeline.context_builder.build(final_chunks)
        print(f"\n[FINAL CONTEXT] chunks={len(final_chunks)} expected_terms_present={sorted(present_final)}")
        for n, chunk in citation_map.items():
            md = meta(chunk)
            print(f"  [{n}] doc={md.get('document_id','?')} page={md.get('page_number','?')} chunk={key(chunk)}")
        if args.show_context:
            print("----- EXACT CONTEXT -----\n" + (context or "(empty)") + "\n----- END CONTEXT -----")

    print("\n[CASE DIAGNOSIS]")
    print(f"  expected terms found: dense={sorted(present_dense)} BM25={sorted(present_sparse)} internal_hybrid={sorted(present_internal_hybrid)} agent_hybrid={sorted(present_hybrid)} reranked={sorted(summary['reranked'])} final5={sorted(summary['final'])}")
    if present_sparse - present_hybrid:
        print(f"  LOSS AT HYBRID: BM25-only expected terms dropped: {sorted(present_sparse-present_hybrid)}")
    if present_hybrid - summary["final"]:
        print(f"  LOSS AT RERANK/TOP5: terms dropped: {sorted(present_hybrid-summary['final'])}")
    return summary


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    if not 1 <= args.top_k <= 50 or not 1 <= args.chunks_per_search <= 50:
        print("--top-k and --chunks-per-search must be from 1 to 50", file=sys.stderr)
        return 2
    try:
        cases = load_cases(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid diagnostic cases: {exc}", file=sys.stderr)
        return 2
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.api.dependencies import get_rag_service
    print(f"Read-only retrieval diagnostics: {len(cases)} case(s); no cache/index mutation. Stop Uvicorn first.")
    try:
        with TestClient(app) as client:
            health = client.get("/api/health"); health.raise_for_status()
            print(f"[HEALTH] {health.json().get('data', {})}")
            rag = get_rag_service()
            logging.getLogger("backend.app.retrieval.hybrid_retriever").setLevel(logging.DEBUG)
            summaries = [diagnose_case(rag, case, args) for case in cases]
            print("\n" + "=" * 96 + "\n[BATCH SUMMARY]")
            for result in summaries:
                print(f"  {result['question']!r}: dense={sorted(result['dense'])} BM25={sorted(result['sparse'])} internal_hybrid={sorted(result.get('hybrid_retriever', set()))} agent_hybrid={sorted(result.get('hybrid', set()))} final5={sorted(result.get('final', set()))}")
        return 0
    except Exception as exc:
        print(f"Diagnostic failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    cli_args = parse_args()
    # PDF text may contain symbols outside the active Windows code page. Keep
    # diagnostics complete instead of aborting while printing a chunk excerpt.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    log_path = cli_args.log_file
    if log_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = ROOT / "backend" / "logs" / f"retrieval_diagnostics_{stamp}.log"
    elif not log_path.is_absolute():
        log_path = ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, terminal, log_stream):
            self.terminal = terminal
            self.log_stream = log_stream

        def write(self, value):
            self.terminal.write(value)
            self.log_stream.write(value)
            return len(value)

        def flush(self):
            self.terminal.flush()
            self.log_stream.flush()

        def isatty(self):
            return self.terminal.isatty()

        def __getattr__(self, name):
            return getattr(self.terminal, name)

        @property
        def encoding(self):
            return getattr(self.terminal, "encoding", "utf-8")

    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as log_stream:
        tee_out = _Tee(original_stdout, log_stream)
        tee_err = _Tee(original_stderr, log_stream)
        sys.stdout, sys.stderr = tee_out, tee_err
        file_handler = logging.StreamHandler(log_stream)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        print(f"[DIAGNOSTIC LOG] {log_path}", flush=True)
        try:
            exit_code = main(cli_args)
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()
            sys.stdout, sys.stderr = original_stdout, original_stderr
    raise SystemExit(exit_code)
