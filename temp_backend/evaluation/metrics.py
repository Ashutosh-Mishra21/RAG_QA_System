import math
from typing import List


def recall_at_k(retrieved_id: List[str], relevant_id: List[str], k: int) -> float:
    retrieved_at_k = retrieved_id[:k]
    hits = len(set(retrieved_at_k) & set(relevant_id))
    return hits / len(relevant_id) if relevant_id else 0.0


def mrr_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    ideal_dcg = dcg_at_k(relevant_ids, relevant_ids, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
