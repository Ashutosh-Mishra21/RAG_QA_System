from fastapi import APIRouter, UploadFile, File
from pathlib import Path
from backend.app.services.ingestion_service import IngestionService

router = APIRouter()

# Define data directory relative to project root
DATA_DIR = Path("data")
ingestion_service = IngestionService(DATA_DIR)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_bytes = await file.read()

    result = ingestion_service.save_file(filename=file.filename, file_bytes=file_bytes)

    return {"message": "File uploaded successfully", "document": result}
