import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from backend.app.api.responses import success_response
from backend.app.services import IngestionService

router = APIRouter(tags=["upload"])
logger = logging.getLogger(__name__)
DATA_DIR = Path("data")
_ingestion_service = None


def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService(DATA_DIR)
    return _ingestion_service


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        logger.info("Upload received: %s", file.filename)
        file_bytes = await file.read()
        service = get_ingestion_service()
        saved = service.save_file(filename=file.filename, file_bytes=file_bytes)
        pipeline_result = service.ingest_and_index(saved["file_path"])
        logger.info(
            "Upload indexed successfully: filename=%s document_id=%s chunks=%s",
            file.filename,
            saved["document_id"],
            pipeline_result.get("chunks_indexed"),
        )
        return success_response(
            {
                "message": "File uploaded and indexed successfully",
                "document": saved,
                "pipeline": pipeline_result,
            }
        )
    except Exception as exc:
        logger.exception("Upload failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail="Upload failed") from exc
