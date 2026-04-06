from backend.app.services import IngestionService, RagService


class RAGOrchestrator:
    def __init__(self, storage_dir):
        self.ingestion_service = IngestionService(storage_dir)
        self.rag_service = RagService()

    # 🔹 Phase 1–2 (already done)
    def ingest_document(self, file_path: str):
        return self.ingestion_service.ingest_and_index(file_path)

    # 🔹 Phase 2 (already done)
    def answer_query(self, query: str, metadata_filters=None):
        return self.rag_service.answer(query, metadata_filters)

    # 🔥 Phase 3 (NEW)
    def run_pipeline(self, query: str):
        result = self.answer_query(query)

        # IMPORTANT: extract context for evaluation
        context = ""
        for src in result.get("sources", []):
            # ideally use actual chunk content
            context += str(src) + "\n"
