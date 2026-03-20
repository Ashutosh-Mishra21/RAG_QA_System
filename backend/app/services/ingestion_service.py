import uuid
from pathlib import Path
from datetime import datetime


class IngestionService:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.raw_dir = self.storage_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, filename: str, file_bytes: bytes):
        # Generate unique document ID
        document_id = str(uuid.uuid4())

        # Create timestamp
        timestamp = datetime.utcnow().isoformat()

        # Preserve file extension
        extension = Path(filename).suffix
        new_filename = f"{document_id}{extension}"

        file_path = self.raw_dir / new_filename

        # Save file
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        return {
            "document_id": document_id,
            "filename": filename,
            "stored_as": new_filename,
            "file_path": str(file_path),
            "timestamp": timestamp,
        }
