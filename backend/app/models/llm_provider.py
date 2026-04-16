import os
import requests
import logging
from typing import Optional, Tuple

from backend.app.core.llm_cache import LLMCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenRouterLLM:
    def __init__(self, model: str):
        self.model = model

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required for OpenRouterLLM") from exc

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "My App",
            },
        )

        logger.info(
            "[OPENROUTER] Client initialized (model=%s, api_key_present=%s)",
            self.model,
            bool(api_key),
        )

    def generate(self, prompt: str) -> str:
        try:
            logger.info(
                "[OPENROUTER] Sending completion request (model=%s)", self.model
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=60,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenRouter")

            logger.info("[OPENROUTER] Completion request succeeded")
            return content.strip()

        except Exception as e:
            logger.exception("[OPENROUTER] Request failed: %r", e)
            raise


class OllamaLLM:
    def __init__(self, model: str = "llama3", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        logger.info("Using Ollama model: %s", self.model)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )

        if response.status_code != 200:
            raise Exception(f"Ollama request failed: {response.text}")

        data = response.json()
        if "response" not in data:
            raise ValueError("Invalid response from Ollama")

        return data["response"].strip()


class LLMRouter:
    def __init__(
        self,
        primary: Optional[OpenRouterLLM] = None,
        fallback: Optional[OllamaLLM] = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.cache = LLMCache()

    def generate(self, prompt: str) -> Tuple[str, str, str]:
        cache_key_model = self.primary.model if self.primary else "fallback"

        cached = self.cache.get(prompt, model=cache_key_model)
        if cached:
            logger.info(
                "[ROUTER] Cache hit for prompt (provider=%s)",
                cached.get("provider", "unknown"),
            )
            return (
                cached["response"],
                cached.get("model", "cached"),
                cached.get("provider", "unknown"),
            )

        # Try primary
        if self.primary:
            try:
                logger.info(
                    "[ROUTER] Trying primary provider: OpenRouter (%s)",
                    self.primary.model,
                )

                primary_resp = self.primary.generate(prompt)

                logger.info(
                    "[ROUTER] Primary provider success: OpenRouter (%s)",
                    self.primary.model,
                )

                self.cache.set(
                    prompt,
                    primary_resp,
                    provider="api",
                    model=self.primary.model,
                )

                return primary_resp, self.primary.model, "api"

            except Exception as primary_error:
                logger.exception("[ROUTER] Primary provider failed: %s", primary_error)
                logger.info("[ROUTER] Switching to fallback provider")

        # Try fallback
        if self.fallback:
            try:
                logger.info(
                    "[ROUTER] Trying fallback provider: Ollama (%s)",
                    self.fallback.model,
                )

                resp = self.fallback.generate(prompt)

                logger.info(
                    "[ROUTER] Fallback provider success: Ollama (%s)",
                    self.fallback.model,
                )

                self.cache.set(
                    prompt,
                    resp,
                    provider="local",
                    model=self.fallback.model,
                )

                return resp, self.fallback.model, "local"

            except Exception as fallback_error:
                logger.exception(
                    "[ROUTER] Fallback provider failed: %s", fallback_error
                )
                raise RuntimeError(
                    f"Both providers failed. Fallback error: {fallback_error}"
                ) from fallback_error

        raise RuntimeError("No LLM providers available")
