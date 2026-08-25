from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Tuple

from backend.app.models import DocumentManifest


@dataclass
class _SimpleElement:
    text: str
    label_name: str
    page_number: int | None = None
    source_bbox: dict | None = None

    @property
    def label(self) -> SimpleNamespace:
        return SimpleNamespace(name=self.label_name)


class _SimpleDocument:
    """Lightweight document with a Docling-compatible iterate_items API."""

    def __init__(self, elements: list[_SimpleElement]):
        self._elements = elements

    def iterate_items(self) -> Iterator[Tuple[_SimpleElement, None]]:
        for element in self._elements:
            yield element, None


class _CombinedDocument:
    """Docling-compatible view over native and selectively parsed elements."""

    def __init__(self, documents):
        self.documents = documents

    def iterate_items(self):
        for document in self.documents:
            yield from document.iterate_items()


class DoclingParser:
    def __init__(self) -> None:
        self.converter = None

    def parse(self, file_path: Path):
        path = Path(file_path)
        suffix = path.suffix.lower()

        # ✅ Explicit safe formats (NO native deps)
        if suffix in {".md", ".markdown", ".txt"}:
            return self._parse_markdown(path)

        docling_supported = {
            ".pdf",
            ".docx",
            ".xlsx",
        }
        # ❗ Optional: block unsupported formats early (prevents DLL crash)
        if suffix not in docling_supported:
            raise ValueError(f"Unsupported file type: {suffix}")

        try:
            # ✅ Only now attempt docling
            converter = self._get_docling_converter()
            result = converter.convert(str(path))
            return result.document
        except Exception as e:
            # Re-wrap or log the error so you know WHICH file failed
            raise RuntimeError(f"Docling failed to parse {path.name}: {e}")

    def from_manifest(
        self,
        manifest: DocumentManifest,
        excluded_page_numbers: set[int] | None = None,
    ) -> _SimpleDocument:
        """Build a lightweight document for pages that need no expensive parser."""
        excluded_page_numbers = excluded_page_numbers or set()
        elements: list[_SimpleElement] = []
        for page in manifest.pages:
            if page.page_number in excluded_page_numbers:
                continue
            if page.section:
                elements.append(
                    _SimpleElement(
                        text=page.section,
                        label_name="SECTION_HEADER",
                        page_number=page.page_number,
                        source_bbox={"bbox": page.source.page_bbox},
                    )
                )
            blocks = page.blocks
            if not blocks and page.text:
                blocks = [
                    type(
                        "ManifestBlock",
                        (),
                        {"text": page.text, "bbox": page.source.page_bbox},
                    )()
                ]
            for block in blocks:
                elements.append(
                    _SimpleElement(
                        text=block.text,
                        label_name="TEXT",
                        page_number=page.page_number,
                        source_bbox={"bbox": block.bbox},
                    )
                )
        return _SimpleDocument(elements)

    def parse_pages(
        self,
        file_path: Path,
        page_numbers: set[int],
    ) -> _CombinedDocument:
        """Parse selected PDF pages using Docling's 1-indexed page ranges."""
        if not page_numbers:
            return _CombinedDocument([])
        if any(page < 1 for page in page_numbers):
            raise ValueError("Page numbers must be positive")

        path = Path(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError("Page-selective parsing currently supports PDF files only")

        ranges = []
        sorted_pages = sorted(page_numbers)
        start = previous = sorted_pages[0]
        for page in sorted_pages[1:]:
            if page != previous + 1:
                ranges.append((start, previous))
                start = page
            previous = page
        ranges.append((start, previous))

        documents = []
        for start_page, end_page in ranges:
            try:
                converter = self._get_docling_converter(
                    page_range=(start_page, end_page)
                )
                documents.append(converter.convert(str(path)).document)
            except Exception as exc:
                raise RuntimeError(
                    f"Docling failed to parse pages {start_page}-{end_page} of {path.name}"
                ) from exc
        return _CombinedDocument(documents)

    @staticmethod
    def combine(*documents) -> _CombinedDocument:
        return _CombinedDocument([document for document in documents if document])

    def _parse_markdown(self, file_path: Path) -> _SimpleDocument:
        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        elements: list[_SimpleElement] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                label_name = "TITLE" if not elements else "SECTION_HEADER"
                elements.append(_SimpleElement(text=line, label_name=label_name))
            elif line.startswith(("- ", "* ")):
                elements.append(
                    _SimpleElement(text=line[2:].strip(), label_name="LIST_ITEM")
                )
            else:
                elements.append(_SimpleElement(text=line, label_name="TEXT"))

        return _SimpleDocument(elements)

    def _get_docling_converter(self, page_range: tuple[int, int] | None = None):
        if page_range is None and self.converter is not None:
            return self.converter

        try:
            from docling.document_converter import (
                DocumentConverter,
                PdfFormatOption,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

        except Exception as exc:
            raise RuntimeError(
                "Docling is required for PDF/DOCX parsing but failed to load."
            ) from exc

        # PDF pipeline config
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        if page_range is not None:
            pipeline_options.page_range = page_range

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        if page_range is None:
            self.converter = converter
        return converter
