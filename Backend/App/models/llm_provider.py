from dotenv import load_dotenv
import os
from openai import OpenAI
from pathlib import Path
import requests
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()


class OpenRouterLLM:
    def __init__(self, model: str):
        self.model = model

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "My App",
            },
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


class OllamaLLM:
    def __init__(self, model: str = "llama3", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
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

    def generate(self, prompt: str) -> str:

        # Try primary (OpenRouter)
        if self.primary:
            try:
                return self.primary.generate(prompt)
            except Exception as e:
                logging.warning(f"Primary LLM failed: {e}")
                print("🔁 Switching to fallback...")

        # Try fallback (Ollama)
        if self.fallback:
            try:
                return self.fallback.generate(prompt)
            except Exception as e:
                raise RuntimeError(f"Both providers failed: {e}")

        raise RuntimeError("No LLM providers available")
