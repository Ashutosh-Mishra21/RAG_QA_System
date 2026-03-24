from backend.app.generation import (
    AnswerValidator,
    ContextBuilder,
    Generator,
    GenerationPipeline,
    PromptBuilder,
)
from backend.app.models import Chunk


class FakeRetriever:
    def retrieve(self, query: str, top_k=40, metadata_filters=None):
        return [
            Chunk(
                id="c1",
                content="Cosine similarity compares vector angle.",
                metadata={
                    "document_id": "doc1",
                    "section": "Embeddings",
                    "page_number": 1,
                },
                score=0.9,
            )
        ]


class FakeReranker:
    def rerank(self, query, chunks):
        return chunks


class FakeLLM:
    def generate(self, prompt):
        return "Cosine similarity compares the angle between vectors. [1]"


def test_generation_pipeline_end_to_end():
    pipeline = GenerationPipeline(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        context_builder=ContextBuilder(max_chunks=5),
        prompt_builder=PromptBuilder(),
        generator=Generator(FakeLLM()),
        validator=AnswerValidator(),
    )

    result = pipeline.run("What is cosine similarity?")
    assert "answer" in result
    assert isinstance(result["sources"], list)
