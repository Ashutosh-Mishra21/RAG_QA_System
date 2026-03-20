import json
from typing import Dict
from backend.app.retrieval.semantic_retriever import SemanticRetriever
from backend.evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k


class RetrievalEvaluator:

    def __init__(self, test_file: str, top_k: int = 5):
        self.test_file = test_file
        self.top_k = top_k
        self.test_queries = self._load_test_queries()

    def _load_test_queries(self):
        import json

        with open(self.test_file, "r") as f:
            return json.load(f)

    def evaluate_retriever(self, retriever, reranker=None):

        all_metrics = []

        for item in self.test_queries:
            query = item["query"]
            relevant_ids = set(item["relevant_ids"])

            results = retriever.retrieve(query, top_k=self.top_k)

            if reranker:
                results = reranker.rerank(query, results)

            retrieved_ids = list(
                dict.fromkeys(r["metadata"]["document_id"] for r in results)
            )

            metrics = self._compute_metrics(relevant_ids, retrieved_ids)
            all_metrics.append(metrics)

        return self._aggregate_metrics(all_metrics)

    def evaluate_agentic(self, agentic_retriever):

        all_metrics = []

        for item in self.test_queries:
            query = item["query"]
            relevant_ids = set(item["relevant_ids"])

            output = agentic_retriever.retrieve(query)
            results = output["results"]

            retrieved_ids = list(
                dict.fromkeys(r["metadata"]["document_id"] for r in results)
            )

            metrics = self._compute_metrics(relevant_ids, retrieved_ids)
            all_metrics.append(metrics)

        return self._aggregate_metrics(all_metrics)

    def _compute_metrics(self, relevant_ids, retrieved_ids):

        recall = recall_at_k(retrieved_ids, list(relevant_ids), self.top_k)
        mrr = mrr_at_k(retrieved_ids, list(relevant_ids), self.top_k)
        ndcg = ndcg_at_k(retrieved_ids, list(relevant_ids), self.top_k)

        return {"recall": recall, "mrr": mrr, "ndcg": ndcg}

    def _aggregate_metrics(self, all_metrics):

        recall = sum(m["recall"] for m in all_metrics) / len(all_metrics)
        mrr = sum(m["mrr"] for m in all_metrics) / len(all_metrics)
        ndcg = sum(m["ndcg"] for m in all_metrics) / len(all_metrics)

        return {
            "Recall@K": round(recall, 4),
            "MRR@K": round(mrr, 4),
            "nDCG@K": round(ndcg, 4),
        }
