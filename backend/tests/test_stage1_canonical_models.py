import fitz
import pytest

from backend.app.ingestion import Canonicalizer
from backend.app.ingestion import PageScanner
from backend.app.ingestion import PageRoutingPolicy
from backend.app.ingestion import DoclingParser
from backend.app.ingestion.page_routing import detect_page_signals
from backend.app.models import (
    DocumentManifest,
    EvidenceItem,
    PageManifest,
    PageSignals,
    PageSource,
)


def test_manifest_canonicalization_creates_provenanced_text_evidence():
    manifest = DocumentManifest(
        document_id="doc-1",
        source_file="report.pdf",
        document_type="pdf",
        page_count=1,
        pages=[
            PageManifest(
                document_id="doc-1",
                page_id="doc-1_p1",
                page_number=1,
                text="Revenue was $61M.",
                signals=PageSignals(text=True),
                source=PageSource(
                    page_bbox=[0.0, 0.0, 612.0, 792.0],
                    storage_path="data/raw/report.pdf",
                ),
            )
        ],
    )

    document = Canonicalizer().from_manifest(manifest)

    assert len(document.evidence) == 1
    evidence = document.evidence[0]
    assert isinstance(evidence, EvidenceItem)
    assert evidence.page_id == "doc-1_p1"
    assert evidence.provenance.source_page == 1
    assert evidence.provenance.source_bbox == [0.0, 0.0, 612.0, 792.0]


def test_empty_manifest_has_no_evidence():
    manifest = DocumentManifest(
        document_id="empty",
        source_file="empty.pdf",
        document_type="pdf",
        page_count=0,
    )

    assert Canonicalizer().from_manifest(manifest).evidence == []


def test_text_scanner_creates_logical_section_pages(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("Intro\n# Methods\nMethod details", encoding="utf-8")

    manifest = PageScanner().scan(source, document_id="notes")

    assert manifest.document_type == "md"
    assert manifest.page_count == 2
    assert manifest.pages[1].section == "Methods"
    assert manifest.pages[1].blocks[0].line_start == 2


def test_routing_policy_keeps_normal_text_on_cheap_path():
    manifest = DocumentManifest(
        document_id="doc",
        source_file="doc.pdf",
        document_type="pdf",
        page_count=1,
        pages=[PageManifest(document_id="doc", page_id="doc_p1", page_number=1)],
    )

    decision = PageRoutingPolicy().decide(manifest)[0]

    assert decision.actions == []
    assert decision.priority == 0


def test_signal_detection_identifies_table_like_text():
    table, chart, formula, numeric_count, density = detect_page_signals(
        "Table 1\nRegion 2023 2024\nEurope 52 61\nAsia 40 47\nTotal 92 108",
        image_count=0,
        block_count=5,
        page_area=500_000,
    )

    assert table is True
    assert chart is False
    assert formula is False
    assert numeric_count >= 6
    assert density >= 0.0


def test_manifest_is_reused_until_source_changes(tmp_path):
    source = tmp_path / "notes.txt"
    manifest_path = tmp_path / "notes_manifest.json"
    source.write_text("first version", encoding="utf-8")
    scanner = PageScanner()

    first = scanner.load_or_scan(source, manifest_path, document_id="notes")
    second = scanner.load_or_scan(source, manifest_path, document_id="notes")
    source.write_text("second version", encoding="utf-8")
    third = scanner.load_or_scan(source, manifest_path, document_id="notes")

    assert first.document_hash == second.document_hash
    assert second.pages[0].text == "first version"
    assert third.document_hash != first.document_hash
    assert third.pages[0].text == "second version"


def test_routing_policy_escalates_scanned_and_table_pages():
    manifest = DocumentManifest(
        document_id="doc",
        source_file="doc.pdf",
        document_type="pdf",
        page_count=2,
        pages=[
            PageManifest(
                document_id="doc",
                page_id="doc_p1",
                page_number=1,
                signals=PageSignals(scanned=True, image=True),
            ),
            PageManifest(
                document_id="doc",
                page_id="doc_p2",
                page_number=2,
                signals=PageSignals(text=True, table=True),
            ),
        ],
    )

    decisions = PageRoutingPolicy().decide(manifest)

    assert "parse_ocr" in decisions[0].actions
    assert "parse_table" in decisions[1].actions


def test_scanner_rejects_page_count_limit(tmp_path):
    source = tmp_path / "document.pdf"
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(source)
    pdf.close()
    scanner = PageScanner(max_pages=0)

    with pytest.raises(ValueError, match="page-count limit"):
        scanner.scan(source, document_id="document")


def test_manifest_document_can_be_combined_with_selective_document():
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
                text="native page",
            )
        ],
    )
    parser = DoclingParser()
    native = parser.from_manifest(manifest)
    combined = parser.combine(native, native)

    assert len(list(combined.iterate_items())) == 2
