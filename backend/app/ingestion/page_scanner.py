from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import fitz
from filelock import FileLock

from backend.app.models import (
    DocumentManifest,
    PageBlock,
    PageManifest,
    PageSignals,
    PageSource,
    PageStatistics,
)
from .page_routing import detect_page_signals

logger = logging.getLogger(__name__)


class PageScanner:
    """Perform a bounded, cheap first pass over supported documents."""

    schema_version = 2

    def __init__(
        self,
        max_bytes: int = 25 * 1024 * 1024,
        max_pages: int = 500,
        max_page_text_chars: int = 2_000_000,
        max_page_blocks: int = 10_000,
    ):
        self.max_bytes = max_bytes
        self.max_pages = max_pages
        self.max_page_text_chars = max_page_text_chars
        self.max_page_blocks = max_page_blocks

    def scan(
        self, file_path: str | Path, document_id: str | None = None
    ) -> DocumentManifest:
        path = Path(file_path).resolve()
        self._validate_path(path)
        resolved_id = document_id or path.stem
        document_hash = self._hash_file(path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            manifest = self._scan_pdf(path, resolved_id)
        elif suffix in {".md", ".markdown", ".txt"}:
            manifest = self._scan_text(path, resolved_id)
        else:
            raise ValueError(f"Page scanning does not support file type: {suffix}")

        manifest.document_hash = document_hash
        manifest.schema_version = self.schema_version
        manifest.scanner_name = "pymupdf" if suffix == ".pdf" else "text-scanner"
        manifest.scanner_version = self._scanner_version(manifest.scanner_name)
        return manifest

    def load_or_scan(
        self,
        file_path: str | Path,
        manifest_path: str | Path,
        document_id: str | None = None,
    ) -> DocumentManifest:
        path = Path(file_path).resolve()
        current_hash = self._hash_file(path)
        try:
            data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            manifest = DocumentManifest.model_validate(data)
            if (
                manifest.schema_version == self.schema_version
                and manifest.document_hash == current_hash
                and manifest.document_id == (document_id or path.stem)
            ):
                return manifest
        except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
            pass

        manifest = self.scan(path, document_id=document_id)
        self.persist(manifest, manifest_path)
        return manifest

    def persist(self, manifest: DocumentManifest, manifest_path: str | Path) -> None:
        target = Path(manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(target) + ".lock"):
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as temporary:
                    json.dump(manifest.model_dump(mode="json"), temporary, indent=2)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

    def _validate_path(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError("Document was not found")
        if path.stat().st_size > self.max_bytes:
            raise ValueError("Document exceeds the scan size limit")

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _scan_pdf(self, path: Path, document_id: str) -> DocumentManifest:
        pages: list[PageManifest] = []
        with fitz.open(path) as pdf:
            if pdf.is_encrypted:
                raise ValueError(
                    "Encrypted PDFs are not supported by the cheap scanner"
                )
            if len(pdf) > self.max_pages:
                raise ValueError("PDF exceeds the page-count limit")
            for index, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                raw_blocks = page.get_text("blocks")
                if len(raw_blocks) > self.max_page_blocks:
                    raise ValueError("PDF page exceeds the text-block limit")
                if len(text) > self.max_page_text_chars:
                    raise ValueError("PDF page exceeds the text limit")
                blocks = []
                for block_index, raw_block in enumerate(raw_blocks, start=1):
                    block_text = str(raw_block[4]).strip() if len(raw_block) > 4 else ""
                    if not block_text:
                        continue
                    blocks.append(
                        PageBlock(
                            block_id=f"{document_id}_p{index}_b{block_index}",
                            text=block_text,
                            bbox=[float(value) for value in raw_block[:4]],
                            block_type="text",
                        )
                    )
                image_count = len(page.get_images(full=True))
                rect = page.rect
                table, chart, formula, numeric_tokens, density = detect_page_signals(
                    text,
                    image_count,
                    len(blocks),
                    float(rect.width * rect.height),
                )
                pages.append(
                    PageManifest(
                        document_id=document_id,
                        page_id=f"{document_id}_p{index}",
                        page_number=index,
                        text=text,
                        blocks=blocks,
                        signals=PageSignals(
                            text=bool(text),
                            image=image_count > 0,
                            table=table,
                            chart=chart,
                            formula=formula,
                            scanned=not bool(text) and image_count > 0,
                        ),
                        statistics=PageStatistics(
                            text_chars=len(text),
                            text_blocks=len(blocks),
                            image_count=image_count,
                            word_count=len(text.split()),
                            numeric_token_count=numeric_tokens,
                            text_density=density,
                            heading_count=sum(
                                1
                                for line in text.splitlines()
                                if line.strip().endswith(":")
                            ),
                            list_item_count=sum(
                                1
                                for line in text.splitlines()
                                if line.strip().startswith(("- ", "* "))
                            ),
                        ),
                        source=PageSource(
                            page_bbox=[rect.x0, rect.y0, rect.x1, rect.y1],
                            storage_path=str(path),
                            page_rotation=page.rotation,
                        ),
                    )
                )
        return DocumentManifest(
            document_id=document_id,
            source_file=path.name,
            document_type="pdf",
            page_count=len(pages),
            pages=pages,
        )

    def _scan_text(self, path: Path, document_id: str) -> DocumentManifest:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        sections: list[tuple[str, list[str], int]] = []
        current_title = None
        current_lines: list[str] = []
        current_start = 1
        for line_number, line in enumerate(lines, start=1):
            if path.suffix.lower() in {".md", ".markdown"} and line.lstrip().startswith(
                "#"
            ):
                if current_lines:
                    sections.append(
                        (current_title or "Introduction", current_lines, current_start)
                    )
                current_title = line.lstrip("#").strip() or "Introduction"
                current_lines = []
                current_start = line_number
            else:
                current_lines.append(line)
        if current_lines or not sections:
            sections.append(
                (current_title or "Introduction", current_lines, current_start)
            )

        pages = []
        for index, (title, section_lines, start_line) in enumerate(sections, start=1):
            section_text = "\n".join(section_lines).strip()
            pages.append(
                PageManifest(
                    document_id=document_id,
                    page_id=f"{document_id}_s{index}",
                    page_number=index,
                    text=section_text,
                    blocks=(
                        [
                            PageBlock(
                                block_id=f"{document_id}_s{index}_b1",
                                text=section_text,
                                line_start=start_line,
                                line_end=start_line + max(len(section_lines) - 1, 0),
                            )
                        ]
                        if section_text
                        else []
                    ),
                    section=title,
                    signals=PageSignals(text=bool(section_text)),
                    statistics=PageStatistics(
                        text_chars=len(section_text),
                        text_blocks=1 if section_text else 0,
                        word_count=len(section_text.split()),
                        numeric_token_count=len(
                            re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)", section_text)
                        ),
                        text_density=min(1.0, len(section_text) / 10_000),
                        heading_count=1 if title != "Introduction" else 0,
                        list_item_count=sum(
                            1
                            for line in section_lines
                            if line.strip().startswith(("- ", "* ", "• "))
                        ),
                    ),
                    source=PageSource(storage_path=str(path)),
                )
            )
        return DocumentManifest(
            document_id=document_id,
            source_file=path.name,
            document_type=path.suffix.lstrip(".").lower(),
            page_count=len(pages),
            pages=pages,
        )

    @staticmethod
    def _scanner_version(scanner_name: str) -> str | None:
        try:
            return version("PyMuPDF") if scanner_name == "pymupdf" else "1"
        except PackageNotFoundError:
            return None
