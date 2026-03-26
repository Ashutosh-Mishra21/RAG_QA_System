from dotenv import load_dotenv, find_dotenv
import os
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

load_dotenv(find_dotenv())


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

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=60,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print("\n🔥 OPENROUTER ERROR:", repr(e), "\n")
            raise


class OllamaLLM:
    def __init__(self, model: str = "llama3", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        print(f"🚀 Ollama using model: {self.model}")
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
            raise Exception("Ollama request failed")

        return response.json()["response"].strip()


class LLMRouter:
    def __init__(self, primary=None, fallback=None):
        self.primary = primary
        self.fallback = fallback

    def generate(self, prompt: str):
        # Try primary
        if self.primary:
            for attempt in range(2):
                try:
                    logging.info(f"Using OpenRouter model: {self.primary.model}")
                    return self.primary.generate(prompt), self.primary.model
                except Exception as e:
                    logging.warning(f"Primary failed (attempt {attempt+1}): {e}")

        # Fallback
        if self.fallback:
            logging.info("Switching to fallback model")
            logging.info(f"Using Ollama model: {self.fallback.model}")
            try:
                return self.fallback.generate(prompt), self.fallback.model
            except Exception as e:
                raise RuntimeError(f"Both providers failed: {e}")

        raise RuntimeError("No LLM providers available")
