from pathlib import Path
from docling.document_converter import DocumentConverter


class DoclingParser:

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_path: Path):
        result = self.converter.convert(str(file_path))
        document = result.document

        return document
