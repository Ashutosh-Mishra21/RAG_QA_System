from __future__ import annotations

from pathlib import Path

from backend.app.ingestion.canonicalizer import Canonicalizer
from backend.app.ingestion.docling_parser import DoclingParser
from backend.app.models import CanonicalDocument, DocumentManifest


class PageActionExecutor:
    """Execute routed PDF-page actions once per selected page set."""

    def __init__(self, parser: DoclingParser | None = None) -> None:
        self.parser = parser or DoclingParser()
        self.canonicalizer = Canonicalizer()

    def execute(
        self,
        file_path: str | Path,
        manifest: DocumentManifest,
        canonical: CanonicalDocument,
        page_ids: set[str] | None = None,
    ) -> CanonicalDocument:
        path = Path(file_path)
        page_numbers = {
            page.page_number
            for page in manifest.pages
            for decision in manifest.routing
            if decision.page_id == page.page_id
            and (page_ids is None or page.page_id in page_ids)
            and any(action != "use_native_text" for action in decision.actions)
        }
        if path.suffix.lower() != ".pdf" or not page_numbers:
            return canonical

        parsed = self.parser.parse_pages(path, page_numbers)
        replacement = self.canonicalizer.from_docling(
            parsed,
            document_id=canonical.document_id,
            source_file=canonical.source_file,
            document_type=canonical.document_type,
            tenant_id=canonical.tenant_id,
            workspace_id=canonical.workspace_id,
            document_version=canonical.document_version,
        )
        target_page_ids = {
            page.page_id for page in manifest.pages if page.page_number in page_numbers
        }
        replacement_by_page = {
            page_id
            for page_id in target_page_ids
            if any(item.page_id == page_id for item in replacement.evidence)
        }
        if not replacement_by_page:
            return canonical
        evidence = [
            item for item in canonical.evidence if item.page_id not in replacement_by_page
        ]
        evidence.extend(
            item for item in replacement.evidence if item.page_id in replacement_by_page
        )
        return canonical.model_copy(update={"evidence": evidence})
