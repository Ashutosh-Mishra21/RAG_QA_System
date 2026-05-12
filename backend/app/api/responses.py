from typing import Any, Literal

from pydantic import BaseModel


def success_response(data: Any) -> dict[str, Any]:
    return {
        "success": True,
        "data": data,
        "error": None,
    }


def error_response(error: str) -> dict[str, Any]:
    return {
        "success": False,
        "data": None,
        "error": error,
    }


class ErrorApiResponse(BaseModel):
    success: Literal[False] = False
    data: None = None
    error: str
