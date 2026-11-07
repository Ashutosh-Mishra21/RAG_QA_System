from backend.app.generation.pipeline import GenerationPipeline
from backend.app.models import Chunk


class InsufficientAssessment:
    sufficient = False
    score = 0.2
    reasons = ["weak query-term coverage", "missing query terms: relation"]


class RecordingGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        raise AssertionError("Generation must not run with insufficient final context")

    def stream(self, prompt):
        self.calls += 1
        raise AssertionError("Generation must not stream with insufficient final context")


class IdentityReranker:
    def rerank(self, query, chunks):
        return list(chunks)


class StaticContextBuilder:
    def build(self, chunks):
        return "unsupported context", {}


def _pipeline():
    generator = RecordingGenerator()
    pipeline = GenerationPipeline(
        retriever=None,
        reranker=IdentityReranker(),
        context_builder=StaticContextBuilder(),
        prompt_builder=None,
        generator=generator,
        validator=None,
        sufficiency_checker=lambda query, chunks: InsufficientAssessment(),
    )
    return pipeline, generator


def test_generation_is_skipped_when_final_context_is_insufficient():
    pipeline, generator = _pipeline()
    retrieved = [
        Chunk(
            id="c1",
            content="A short passage that misses the requested relation.",
            metadata={"document_id": "doc1"},
            score=0.8,
        )
    ]

    result = pipeline.run("Explain the missing relation", retrieved=retrieved)

    assert result["answer"] == "I don't know based on the available evidence."
    assert "missing query terms: relation" in result["reason"]
    assert generator.calls == 0


def test_streaming_generation_is_skipped_when_final_context_is_insufficient():
    pipeline, generator = _pipeline()
    retrieved = [
        Chunk(
            id="c1",
            content="A short passage that misses the requested relation.",
            metadata={"document_id": "doc1"},
            score=0.8,
        )
    ]

    events = list(pipeline.stream("Explain the missing relation", retrieved=retrieved))

    assert [event["event"] for event in events] == ["chunk", "final"]
    assert events[-1]["data"]["answer"] == "I don't know based on the available evidence."
    assert generator.calls == 0
