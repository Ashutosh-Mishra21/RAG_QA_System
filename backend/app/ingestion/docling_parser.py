# from pathlib import Path
# from docling.document_converter import DocumentConverter


# class DoclingParser:

#     def __init__(self):
#         self.converter = DocumentConverter()

#     def parse(self, file_path: Path):
#         result = self.converter.convert(str(file_path))
#         document = result.document

#         return document

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Tuple


@dataclass
class _SimpleElement:
    text: str
    label_name: str

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

    def _get_docling_converter(self):
        if self.converter is not None:
            return self.converter

        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except Exception as exc:
            raise RuntimeError(
                "Docling is required for PDF/DOCX parsing but failed to load. "
                "This is likely due to missing native dependencies (DLLs) on your system."
            ) from exc

        # Disable OCR
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False

        self.converter = DocumentConverter(pdf_pipeline_options=pipeline_options)

        return self.converter
