import json
from pathlib import Path
from backend.evaluation.generation_metrics import faithfulness_score, relevance_score


class GenerationEvaluator:
    def __init__(self, test_file: str):
        self.test_file = test_file
        self.test_queries = self._load()

    def _load(self):
        file_path = Path(self.test_file)
        if not file_path.is_absolute() and not file_path.exists():
            file_path = Path(__file__).resolve().parents[2] / self.test_file
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, pipeline, llm, data_dir):

        results = []

        for item in self.test_queries:
            query = item["query"]

            output = pipeline(query) if data_dir is None else pipeline(query, data_dir)

            answer = output["answer"]
            context = output["context"]

            faith = faithfulness_score(answer, context, llm)
            rel = relevance_score(answer, query, llm)

            results.append(
                {
                    "query": query,
                    "faithfulness": faith,
                    "relevance": rel,
                }
            )

        return self._aggregate(results)

    def _aggregate(self, results):
        faith = sum(r["faithfulness"] for r in results) / len(results)
        rel = sum(r["relevance"] for r in results) / len(results)

        return {
            "Faithfulness": round(faith, 4),
            "Answer Relevance": round(rel, 4),
        }