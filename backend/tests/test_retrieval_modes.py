from backend.app.retrieval.semantic_retriever import SemanticRetriever
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.evaluation.evaluator import evaluate
from backend.app.api.routes.upload import upload_document

dense = SemanticRetriever()
reranker = CrossEncoderReranker()

documents = upload_document()  # same docs used for BM25
hybrid = HybridRetriever(documents)


query = "What is cosine similarity?"


print("Dense Only")
dense_results = dense.retrieve(query)
evaluate(query, dense_results)


print("\nDense + Rerank")
reranked = reranker.rerank(query, dense_results)
evaluate(query, reranked)


print("\nHybrid")
hybrid_results = hybrid.retrieve(query)
evaluate(query, hybrid_results)


print("\nHybrid + Rerank")
hybrid_reranked = reranker.rerank(query, hybrid_results)
evaluate(query, hybrid_reranked)
