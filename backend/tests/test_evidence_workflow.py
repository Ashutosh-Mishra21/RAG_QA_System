from types import SimpleNamespace

from backend.app.indexing import KeywordIndex
from backend.app.ingestion import (
    Canonicalizer,
    EvidenceIndexer,
    PageActionExecutor,
    PageRoutingPolicy,
)
from backend.app.models import (
    CanonicalDocument,
    Chunk,
    DocumentManifest,
    EvidenceItem,
    EvidenceProvenance,
    PageManifest,
    PageSignals,
)
from backend.app.retrieval import (
    EvidenceSufficiencyScorer,
    EvidenceTools,
    GetPageInput,
    GetRegionInput,
    ParseEvidenceInput,
    SearchHybridInput,
)


def _evidence(
    evidence_id="e1",
    content="Hybrid retrieval combines dense and keyword evidence.",
    page_id="doc_p1",
    bbox=None,
    tenant_id="tenant-a",
):
    return EvidenceItem(
        evidence_id=evidence_id,
        document_id="doc",
        page_id=page_id,
        tenant_id=tenant_id,
        workspace_id="workspace-a",
        document_version="hash-1",
        evidence_type="text",
        content=content,
        bbox=bbox,
        parser_name="pymupdf",
        confidence=0.9,
        provenance=EvidenceProvenance(
            source_page=1,
            source_bbox=bbox,
            page_id=page_id,
        ),
    )


def test_evidence_indexer_propagates_provenance_and_scope_metadata():
    evidence = _evidence(bbox=[1.0, 2.0, 10.0, 12.0])
    document = CanonicalDocument(
        document_id="doc",
        source_file="doc.pdf",
        document_type="pdf",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        document_version="hash-1",
        pages=[PageManifest(document_id="doc", page_id="doc_p1", page_number=1)],
        evidence=[evidence],
    )

    chunk = EvidenceIndexer().build_chunks(document)[0]

    assert chunk.id == evidence.evidence_id
    assert chunk.metadata["evidence_id"] == "e1"
    assert chunk.metadata["page_id"] == "doc_p1"
    assert chunk.metadata["bbox"] == [1.0, 2.0, 10.0, 12.0]
    assert chunk.metadata["evidence_type"] == "text"
    assert chunk.metadata["parser_name"] == "pymupdf"
    assert chunk.metadata["confidence"] == 0.9
    assert chunk.metadata["tenant_id"] == "tenant-a"
    assert chunk.metadata["workspace_id"] == "workspace-a"
    assert chunk.metadata["document_version"] == "hash-1"


def test_evidence_page_and_region_tools_use_scoped_cached_canonical_evidence(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    first = _evidence("first", page_id="doc_p1", bbox=[0, 0, 20, 20])
    second = _evidence("second", page_id="doc_p1", bbox=[30, 30, 40, 40])
    document = CanonicalDocument(
        document_id="doc",
        source_file="doc.pdf",
        document_type="pdf",
        pages=[PageManifest(document_id="doc", page_id="doc_p1", page_number=1)],
        evidence=[first, second],
    )
    (processed / "doc_canonical.json").write_text(
        document.model_dump_json(), encoding="utf-8"
    )
    tools = EvidenceTools(retriever=None, storage_dir=tmp_path)

    page = tools.get_page(
        GetPageInput(
            document_id="doc",
            page_id="doc_p1",
            tenant_id="tenant-a",
            workspace_id="workspace-a",
        )
    )
    region = tools.get_region(
        GetRegionInput(
            document_id="doc",
            page_id="doc_p1",
            bbox=[0, 0, 25, 25],
            tenant_id="tenant-a",
            workspace_id="workspace-a",
        )
    )

    assert [item.evidence_id for item in page] == ["first", "second"]
    assert [item.evidence_id for item in region] == ["first"]


def test_search_hybrid_returns_evidence_and_honors_metadata_filters():
    class FakeRetriever:
        def retrieve(self, query, top_k, metadata_filters):
            assert query == "hybrid retrieval"
            assert top_k == 3
            assert metadata_filters == {"tenant_id": "tenant-a"}
            return [
                Chunk(
                    id="e1",
                    content="Hybrid retrieval combines dense and keyword evidence.",
                    metadata={
                        "document_id": "doc",
                        "page_id": "doc_p1",
                        "page_number": 1,
                        "evidence_id": "e1",
                        "evidence_type": "text",
                        "parser_name": "pymupdf",
                        "confidence": 1.0,
                        "tenant_id": "tenant-a",
                        "workspace_id": "workspace-a",
                        "document_version": "hash-1",
                    },
                    score=1.0,
                )
            ]

    result = EvidenceTools(FakeRetriever()).search_hybrid(
        SearchHybridInput(
            query="hybrid retrieval",
            top_k=3,
            metadata_filters={"tenant_id": "tenant-a"},
        )
    )

    assert result.evidence[0].evidence_id == "e1"
    assert result.evidence[0].provenance.page_id == "doc_p1"
    assert result.sufficiency.sufficient is True


def test_sufficiency_gate_rejects_empty_or_weak_evidence():
    scorer = EvidenceSufficiencyScorer()

    empty = scorer.score("hybrid retrieval", [])
    weak = scorer.score("hybrid retrieval", [_evidence(content="unrelated note")])

    assert empty.sufficient is False
    assert "no retrievable evidence" in empty.reasons
    assert weak.sufficient is False
    assert "weak query-term coverage" in weak.reasons


def test_targeted_parse_tools_invoke_the_bound_page_action():
    calls = []

    def parser(request, action):
        calls.append((request.document_id, action))
        return [_evidence(evidence_id="table-result")]

    tools = EvidenceTools(retriever=None, page_parser=parser)
    result = tools.parse_table(ParseEvidenceInput(document_id="doc", page_id="doc_p1"))

    assert [item.evidence_id for item in result] == ["table-result"]
    assert calls == [("doc", "parse_table")]


def test_bm25_filters_evidence_scope_metadata(tmp_path):
    document = CanonicalDocument(
        document_id="doc",
        source_file="doc.pdf",
        document_type="pdf",
        pages=[PageManifest(document_id="doc", page_id="doc_p1", page_number=1)],
        evidence=[
            _evidence("a", tenant_id="tenant-a"),
            _evidence("b", tenant_id="tenant-b"),
        ],
    )
    index = KeywordIndex(tmp_path / "bm25.json")
    index.add(EvidenceIndexer().build_chunks(document))

    results = index.retrieve(
        "hybrid retrieval", metadata_filters={"tenant_id": "tenant-b"}
    )

    assert [chunk.id for chunk in results] == ["b"]


def test_page_action_executor_replaces_selected_page_with_docling_evidence(tmp_path):
    manifest = DocumentManifest(
        document_id="doc",
        source_file="doc.pdf",
        document_type="pdf",
        page_count=1,
        pages=[
            PageManifest(
                document_id="doc",
                page_id="doc_p1",
                page_number=1,
                text="native table text",
                signals=PageSignals(text=True, table=True),
            )
        ],
    )
    PageRoutingPolicy().decide(manifest)
    canonical = Canonicalizer().from_manifest(manifest)

    class FakeDocument:
        def iterate_items(self):
            element = SimpleNamespace(
                text="Revenue | 2024",
                page_number=1,
                source_bbox={"bbox": [0, 0, 10, 10]},
                label=SimpleNamespace(name="TABLE"),
            )
            yield element, None

    class FakeParser:
        def parse_pages(self, file_path, page_numbers):
            assert page_numbers == {1}
            return FakeDocument()

    result = PageActionExecutor(parser=FakeParser()).execute(
        tmp_path / "doc.pdf", manifest, canonical
    )

    assert [item.evidence_type for item in result.evidence] == ["table"]
    assert result.evidence[0].parser_name == "docling"
