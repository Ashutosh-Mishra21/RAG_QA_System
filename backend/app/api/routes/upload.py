from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from backend.app.services import IngestionService

router = APIRouter(tags=["upload"])
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
        file_bytes = await file.read()
        service = get_ingestion_service()
        saved = service.save_file(filename=file.filename, file_bytes=file_bytes)
        pipeline_result = service.ingest_and_index(saved["file_path"])
        return {
            "message": "File uploaded and indexed successfully",
            "document": saved,
            "pipeline": pipeline_result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"upload_failed: {exc}") from exc
