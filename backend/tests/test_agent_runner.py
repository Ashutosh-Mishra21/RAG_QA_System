from backend.app.agent import AgenticQueryRunner
from backend.app.models import Chunk, EvidenceItem, EvidenceProvenance
from backend.app.retrieval import EvidenceSearchResult, EvidenceSufficiency


def _chunk(content: str = "retrieval evidence") -> Chunk:
    return Chunk(
        id="e1",
        content=content,
        metadata={
            "document_id": "doc",
            "page_id": "doc_p1",
            "page_number": 1,
            "tenant_id": "tenant",
            "workspace_id": "workspace",
        },
        score=1.0,
    )


class FakeTools:
    def __init__(self):
        self.calls = []

    def search_hybrid(self, request):
        self.calls.append(request)
        sufficient = len(self.calls) > 1
        evidence = [self.evidence_from_chunk(_chunk())]
        return EvidenceSearchResult(
            chunks=[_chunk()],
            evidence=evidence,
            sufficiency=EvidenceSufficiency(
                score=0.8 if sufficient else 0.2,
                sufficient=sufficient,
                evidence_count=1,
                covered_query_terms=1,
                query_terms=1,
            ),
        )

    @staticmethod
    def evidence_from_chunk(chunk):
        return EvidenceItem(
            evidence_id=chunk.id,
            document_id="doc",
            page_id="doc_p1",
            evidence_type="text",
            content=chunk.content,
            parser_name="test",
            confidence=1.0,
            provenance=EvidenceProvenance(source_page=1, page_id="doc_p1"),
        )


class FakeRewriter:
    def rewrite(self, query):
        return f"rewritten: {query}"


class FakeDecomposer:
    def decompose(self, query):
        return [query]


def test_agent_retries_once_and_returns_final_evidence():
    tools = FakeTools()
    runner = AgenticQueryRunner(tools, FakeRewriter(), FakeDecomposer())

    plan, decisions, results = runner.run("find retrieval evidence", {})

    assert plan.subqueries == ["find retrieval evidence"]
    assert len(tools.calls) == 2
    assert [decision.action for decision in decisions] == ["search", "rewrite", "search"]
    assert results[0]["sufficiency"].sufficient is True
