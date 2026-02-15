from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class Settings:
    google_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.google_api_key is None:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if self.gemini_api_key is None:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
