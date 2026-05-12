import logging
from pathlib import Path

from backend.app.services.rag_service import RagService
from backend.evaluation.generation_evaluator import GenerationEvaluator
from backend.evaluation.evaluator import RetrievalEvaluator
from backend.app.core.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def run_full_evaluation():

    rag = RagService(storage_dir=Path("data"))
    for file in Path("data/raw").glob("*"):
        logger.info("Ingesting evaluation file: %s", file)
        rag.ingest(str(file))

    llm = ModelRegistry.instance().get_llm_router()

    retrieval_eval = RetrievalEvaluator("backend/evaluation/test_queries.json")
    generation_eval = GenerationEvaluator("backend/evaluation/dataset.json")

    retrieval_scores = retrieval_eval.evaluate_retriever(rag.retriever)
    generation_scores = generation_eval.evaluate(
        pipeline=rag.run_pipeline,
        llm=llm,
        data_dir=None,
    )

    logger.info("Retrieval metrics: %s", retrieval_scores)

    logger.info("Generation metrics: %s", generation_scores)


if __name__ == "__main__":
    run_full_evaluation()
