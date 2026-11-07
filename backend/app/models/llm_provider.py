import os
import time
import requests
import logging
import json
from typing import Iterator, Optional, Tuple

from openai import OpenAI, InternalServerError, RateLimitError, APIConnectionError
from backend.app.core.llm_cache import LLMCache

logger = logging.getLogger(__name__)


class GoogleAIStudioLLM:
    """Gemini API primary provider using Google's OpenAI-compatible endpoint."""

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not found in environment"
            )

        # Allow the client 2 automatic retries for transient HTTP blips
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries=2,
        )

        logger.info(
            "[GEMINI] Client initialized (model=%s, api_key_present=%s)",
            self.model,
            bool(api_key),
        )

    def generate(self, prompt: str, max_retries: int = 2) -> str:
        """Generates completion with explicit exponential backoff for 503/429 spikes."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    "[GEMINI] Sending completion request (model=%s, attempt=%d/%d)",
                    self.model,
                    attempt + 1,
                    max_retries + 1,
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    timeout=60,
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from Gemini API")

                logger.info("[GEMINI] Completion request succeeded")
                return content.strip()

            except (InternalServerError, RateLimitError, APIConnectionError) as e:
                # 503 Service Unavailable / 429 Rate Limit
                last_exception = e
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 2  # 2s, 4s delay
                    logger.warning(
                        "[GEMINI] Transient error (%s). Retrying in %ds...",
                        e.__class__.__name__,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "[GEMINI] All retry attempts exhausted for Gemini API."
                    )
            except Exception as e:
                # Do not retry fatal errors (e.g. 401 Invalid Auth, 400 Bad Request)
                logger.exception("[GEMINI] Non-retryable error: %r", e)
                raise

        if last_exception:
            raise last_exception

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield Gemini response text as it arrives from the API."""
        logger.info("[GEMINI] Starting streaming completion (model=%s)", self.model)
        stream_client = self.client.with_options(max_retries=0)
        response_stream = stream_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=60,
            stream=True,
        )
        try:
            for event in response_stream:
                if not event.choices:
                    continue
                text = event.choices[0].delta.content
                if text:
                    yield text
            logger.info("[GEMINI] Streaming completion finished")
        finally:
            close = getattr(response_stream, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    logger.debug("[GEMINI] Error closing completed response stream", exc_info=True)


class OllamaLLM:
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
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

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield Ollama response text while keeping the HTTP connection cancellable."""
        logger.info("Using Ollama streaming model: %s", self.model)
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True},
            timeout=(10, 60),
            stream=True,
        )
        try:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                payload = json.loads(line)
                text = payload.get("response")
                if text:
                    yield text
                if payload.get("error"):
                    raise RuntimeError(payload["error"])
                if payload.get("done"):
                    break
        finally:
            response.close()


class LLMRouter:
    def __init__(
        self,
        primary: Optional[GoogleAIStudioLLM] = None,
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
                cached.get("model", cache_key_model),
                cached.get("provider", "unknown"),
            )

        # Try primary (Gemini)
        if self.primary:
            try:
                logger.info(
                    "[ROUTER] Trying primary provider: Google AI Studio (%s)",
                    self.primary.model,
                )

                primary_resp = self.primary.generate(prompt)

                logger.info(
                    "[ROUTER] Primary provider success: Google AI Studio (%s)",
                    self.primary.model,
                )

                self.cache.set(
                    prompt,
                    primary_resp,
                    provider="api",
                    model=self.primary.model,
                    cache_model=cache_key_model,
                )

                return primary_resp, self.primary.model, "api"

            except Exception as primary_error:
                logger.exception("[ROUTER] Primary provider failed: %s", primary_error)
                logger.info("[ROUTER] Switching to fallback provider")

        # Try fallback (Ollama)
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
                    cache_model=cache_key_model,
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

    def stream(self, prompt: str) -> Iterator[Tuple[str, str, str]]:
        """Yield (text, model, provider), falling back only before the first token."""
        cache_key_model = self.primary.model if self.primary else "fallback"
        cached = self.cache.get(prompt, model=cache_key_model)
        if cached:
            provider = cached.get("provider", "unknown")
            model = cached.get("model", cache_key_model)
            logger.info("[ROUTER] Streaming cache hit (provider=%s)", provider)
            text = cached.get("response", "")
            if text:
                yield text, model, provider
            return

        providers = []
        if self.primary:
            providers.append((self.primary, "api"))
        if self.fallback:
            providers.append((self.fallback, "local"))
        if not providers:
            raise RuntimeError("No LLM providers available")

        for index, (provider_client, provider_name) in enumerate(providers):
            model = provider_client.model
            emitted = False
            answer_parts = []
            try:
                if index > 0:
                    logger.warning(
                        "[ROUTER] Primary stream failed before output; trying configured fallback (%s)",
                        model,
                    )
                for text in provider_client.stream(prompt):
                    if not text:
                        continue
                    emitted = True
                    answer_parts.append(text)
                    yield text, model, provider_name
                if not emitted:
                    raise ValueError(f"{provider_name} returned an empty stream")

                try:
                    self.cache.set(
                        prompt,
                        "".join(answer_parts),
                        provider=provider_name,
                        model=model,
                        cache_model=cache_key_model,
                    )
                except Exception:
                    logger.exception("[ROUTER] Unable to cache completed stream")
                return
            except Exception:
                logger.exception(
                    "[ROUTER] %s streaming provider failed (emitted_text=%s)",
                    provider_name,
                    emitted,
                )
                if emitted or index == len(providers) - 1:
                    raise
