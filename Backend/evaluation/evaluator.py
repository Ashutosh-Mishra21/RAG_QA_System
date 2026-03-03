import json
from typing import Dict
from backend.app.retrieval.semantic_retriever import SemanticRetriever
from backend.evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k


class RetrievalEvaluator:
    def __init__(self, test_file: str, top_k: int = 5):
        self.test_file = test_file
        self.top_k = top_k
        self.retriever = SemanticRetriever(top_k=top_k)

    def evaluate(self, rerank: bool = True) -> Dict[str, float]:
        with open(self.test_file, "r") as f:
            test_queries = json.load(f)

        recall_scores = []
        mrr_scores = []
        ndcg_scores = []

        for item in test_queries:
            query = item["query"]
            relevant_ids = item["relevant_document_ids"]

            results = self.retriever.retrieve(query=query, rerank=rerank)

            seen = set()
            retrieved_ids = []

            for r in results:
                doc_id = r["metadata"]["document_id"]
                if doc_id not in seen:
                    retrieved_ids.append(doc_id)
                    seen.add(doc_id)

            recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, self.top_k))

            mrr_scores.append(mrr_at_k(retrieved_ids, relevant_ids, self.top_k))

            ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, self.top_k))
            print("Query:", query)
            print("Retrieved IDs:", retrieved_ids)
            print("Relevant IDs:", relevant_ids)
            print("-" * 40)

        return {
            "Recall@K": sum(recall_scores) / len(recall_scores),
            "MRR@K": sum(mrr_scores) / len(mrr_scores),
            "nDCG@K": sum(ndcg_scores) / len(ndcg_scores),
        }
