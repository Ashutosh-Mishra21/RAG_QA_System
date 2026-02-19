from pathlib import Path
from backend.app.ingestion.parser import DocumentParser

class IngestionOrchestrator:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_dir = data_dir/"raw"
        self.processed_dir =  data_dir/"processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self, document_id:str, stored_filename:str):
        file_path = self.raw_dir / stored_filename
        parser = DocumentParser()
        structured_data = parser.parse(file_path)
        
        # Save structured data to processed directory
        output_path = self.processed_dir / f"{document_id}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(structured_data)
            
        return {
            "document_id": document_id,
            "structured_output": str(output_path)
        }