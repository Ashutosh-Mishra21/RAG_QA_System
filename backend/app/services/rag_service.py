from typing import Dict, List

from backend.app.models.response import Response


class RagService:
    def answer(self, query: str, context: List[Dict[str, object]]) -> Response:
        return Response(answer="Not implemented yet.")
