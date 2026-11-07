"""Exercise document upload and chat through the FastAPI routes, without pytest."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".docx", ".xlsx"}
logger = logging.getLogger("rag_e2e")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path, help="Path to a supported document to upload")
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Question to ask after ingestion. Repeat this option to ask more than one.",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Enable LangSmith tracing for this run; requires a valid key and project access.",
    )
    parser.add_argument(
        "--collection",
        default="rag_qa_e2e",
        help="Isolated Qdrant collection used by default (default: rag_qa_e2e).",
    )
    parser.add_argument(
        "--bm25-index",
        default="data/index/bm25_e2e.json",
        help="Isolated BM25 index file used by default.",
    )
    parser.add_argument(
        "--use-configured-index",
        action="store_true",
        help="Use QDRANT_COLLECTION and BM25_INDEX_PATH from the environment/.env.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = args.document if args.document.is_absolute() else ROOT / args.document
    document = document.resolve()
    if not document.is_file():
        print(f"Document not found: {document}", file=sys.stderr)
        return 2
    if document.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(
            f"Unsupported file type: {document.suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            file=sys.stderr,
        )
        return 2

    # Set these before importing backend.app, whose package bootstrap reads .env.
    # This keeps a manual run isolated from the application's normal indexes.
    os.environ["LANGSMITH_TRACING"] = "true" if args.langsmith else "false"
    if not args.use_configured_index:
        os.environ["QDRANT_COLLECTION"] = args.collection
        os.environ["BM25_INDEX_PATH"] = args.bm25_index

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from backend.app.core.config import settings

    if args.langsmith and not (
        settings.LANGSMITH_API_KEY or os.getenv("LANGCHAIN_API_KEY")
    ):
        print(
            "LangSmith tracing was requested, but LANGSMITH_API_KEY or "
            "LANGCHAIN_API_KEY is not configured.",
            file=sys.stderr,
        )
        return 2

    from fastapi.testclient import TestClient
    from backend.app.main import app

    questions = args.questions or [
        "Summarize the main points in this document and cite the supporting sources."
    ]

    print("=" * 72)
    print("RAG end-to-end run")
    print(f"Document: {document}")
    print(f"Qdrant collection: {settings.QDRANT_COLLECTION}")
    print(f"BM25 index: {settings.BM25_INDEX_PATH}")
    print(f"LangSmith tracing: {'enabled' if args.langsmith else 'disabled'}")
    print("Application INFO logs and API route logs follow.")
    print("=" * 72, flush=True)

    try:
        responses = []
        with TestClient(app) as client:
            print("\n[1/3] Checking backend and Qdrant health...", flush=True)
            health_response = client.get("/api/health")
            health_response.raise_for_status()
            health = health_response.json().get("data", {})
            print(json.dumps(health, indent=2), flush=True)
            if not health.get("services", {}).get("qdrant"):
                print(
                    "Qdrant is unavailable. Check QDRANT_URL/API key or start local Qdrant.",
                    file=sys.stderr,
                )
                return 2

            print(f"\n[2/3] Uploading and indexing {document.name}...", flush=True)
            mime_type = mimetypes.guess_type(document.name)[0] or "application/octet-stream"
            with document.open("rb") as source:
                upload_response = client.post(
                    "/api/upload",
                    files={"file": (document.name, source, mime_type)},
                )
            if upload_response.status_code >= 400:
                print(upload_response.text, file=sys.stderr)
                return 1
            upload_payload = upload_response.json()
            if upload_payload.get("success") is not True:
                print(json.dumps(upload_payload, indent=2), file=sys.stderr)
                return 1

            upload_data = upload_payload["data"]
            pipeline = upload_data.get("pipeline", {})
            manifest_path = pipeline.get("manifest_file")
            page_count = None
            if manifest_path and Path(manifest_path).is_file():
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                page_count = len(manifest.get("pages", []))
            print(
                "Upload complete: "
                f"document_id={upload_data.get('document', {}).get('document_id')}, "
                f"pages={page_count if page_count is not None else 'n/a'}, "
                f"chunks_indexed={pipeline.get('chunks_indexed')}",
                flush=True,
            )

            print("\n[3/3] Asking questions through POST /api/chat...", flush=True)
            for number, question in enumerate(questions, start=1):
                print(f"\nQuestion {number}: {question}", flush=True)
                response = client.post("/api/chat", json={"message": question})
                if response.status_code >= 400:
                    print(response.text, file=sys.stderr)
                    return 1
                payload = response.json()
                if payload.get("success") is not True:
                    print(json.dumps(payload, indent=2), file=sys.stderr)
                    return 1
                result = payload["data"]
                print(f"Answer: {result.get('answer', '')}")
                print(f"Confidence: {result.get('confidence')}")
                print(f"Citations: {json.dumps(result.get('citations', []), ensure_ascii=False)}")
                sources = result.get("sources", [])
                citations = result.get("citations", [])
                responses.append(
                    {
                        "source_count": len(sources) if isinstance(sources, list) else 0,
                        "citation_count": len(citations) if isinstance(citations, list) else 0,
                    }
                )
                print(f"Sources returned: {len(sources)}")
                for source in sources[:5]:
                    print(
                        "  - "
                        f"document={source.get('document_id')} "
                        f"page={source.get('page_number', source.get('page_id'))} "
                        f"type={source.get('evidence_type')} "
                        f"evidence={source.get('evidence_id')}"
                    )

        print("\nEnd-to-end API run completed.")
        grounded = sum(item["source_count"] > 0 for item in responses)
        cited = sum(item["citation_count"] > 0 for item in responses)
        print(
            f"Checks: {len(responses)} response(s), {grounded} with source metadata, "
            f"{cited} with citations. Inspect answer text and source pages for grounding."
        )
        if grounded < len(responses):
            print(
                "WARNING: some responses had no source metadata. Check retrieval logs, "
                "filters, and whether the question is answerable from this document."
            )
        if args.langsmith:
            print(
                "LangSmith tracing was enabled. Check the configured project for "
                "the rag.answer, agent, and generation runs."
            )
        return 0
    except Exception as exc:
        logger.exception("End-to-end run failed")
        print(f"End-to-end run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
