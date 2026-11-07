"""Run the lightweight rubric benchmark against a running RAG API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(__file__).with_name("agentic_eval_set.json")


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _score_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    envelope = result.get("data")
    if result.get("success") is not True or not isinstance(envelope, dict):
        raise ValueError(f"API returned an invalid response envelope: {result!r}")

    answer = str(envelope.get("answer") or "")
    normalized = answer.casefold()
    concepts = case["expected_concepts"]
    hits = [any(str(term).casefold() in normalized for term in group) for group in concepts]
    passed = sum(hits) >= int(case.get("min_concept_hits", len(concepts)))
    citations = envelope.get("citations") or []
    sources = envelope.get("sources") or []
    return {
        "id": case["id"],
        "category": case.get("category", "unspecified"),
        "passed": passed,
        "concept_hits": sum(hits),
        "concept_total": len(hits),
        "min_concept_hits": int(case.get("min_concept_hits", len(concepts))),
        "confidence": envelope.get("confidence"),
        "citation_count": len(citations) if isinstance(citations, list) else 1,
        "source_count": len(sources) if isinstance(sources, list) else 1,
        "answer": answer,
    }


def run(api_url: str, dataset_path: Path, delay: float) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must contain a non-empty 'cases' list")

    results = []
    for index, case in enumerate(cases):
        print(f"[{index + 1}/{len(cases)}] {case['id']} ...", flush=True)
        response = _post_json(f"{api_url.rstrip('/')}/api/chat", {"message": case["query"]})
        results.append(_score_case(case, response))
        if delay and index + 1 < len(cases):
            time.sleep(delay)

    concept_total = sum(item["concept_total"] for item in results)
    concept_hits = sum(item["concept_hits"] for item in results)
    summary = {
        "dataset": dataset.get("name", dataset_path.name),
        "case_count": len(results),
        "passed": sum(item["passed"] for item in results),
        "pass_rate": round(sum(item["passed"] for item in results) / len(results), 4),
        "concept_coverage": round(concept_hits / concept_total, 4) if concept_total else 0.0,
        "citation_rate": round(sum(item["citation_count"] > 0 for item in results) / len(results), 4),
        "source_rate": round(sum(item["source_count"] > 0 for item in results) / len(results), 4),
        "results": results,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between model calls")
    args = parser.parse_args()
    dataset_path = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset

    try:
        report = run(args.api, dataset_path, max(0.0, args.delay))
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Could not reach the RAG API at {args.api}: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2

    for item in report["results"]:
        state = "PASS" if item["passed"] else "FAIL"
        print(
            f"{state} {item['id']}: concepts {item['concept_hits']}/{item['concept_total']}, "
            f"citations {item['citation_count']}, sources {item['source_count']}, "
            f"confidence {item['confidence']}"
        )
    print(
        f"\nSummary: {report['passed']}/{report['case_count']} passed; "
        f"concept coverage {report['concept_coverage']:.1%}; "
        f"citation rate {report['citation_rate']:.1%}; source rate {report['source_rate']:.1%}"
    )
    if args.output:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report written to {output_path}")
    return 0 if report["passed"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
