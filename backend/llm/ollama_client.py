"""
Ollama HTTP client — wraps the Ollama local LLM API.

What is Ollama?
  Ollama is a tool that runs open-source language models (like Qwen, Llama,
  Mistral) locally on your machine via a simple REST API. No cloud, no API
  keys. You run `ollama serve` and it listens on http://localhost:11434.

Why not use a Python SDK?
  Ollama's Python library is a thin wrapper over the same HTTP API. Using
  httpx directly keeps the dependency footprint small and makes the request
  format explicit — you can see exactly what's being sent to the model.

Streaming:
  LLMs generate one token at a time. Streaming returns each token to the
  client as it's generated, so the user sees words appearing in real time
  instead of waiting for the full response. We use httpx's async streaming
  to yield tokens as they arrive.

Model configuration:
  The model name comes from Settings.ollama_model — it is never hardcoded
  here. To switch models, change OLLAMA_MODEL in your .env file.
"""

import json
from collections.abc import AsyncGenerator

import httpx

from backend.logging_config import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Async client for Ollama's /api/chat endpoint."""

    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def is_available(self) -> bool:
        """
        Check synchronously whether Ollama is running.

        Called at startup and before queries so we can return a clear
        503 error instead of a cryptic connection refused message.
        """
        try:
            response = httpx.get(
                f"{self._base_url}/api/tags", timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False

    async def chat(self, messages: list[dict]) -> str:
        """
        Send messages and wait for the complete response (non-streaming).

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.

        Returns:
            The model's full reply as a string.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": messages, "stream": False},
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

    async def stream_chat(
        self, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        """
        Stream the model's response token by token.

        Ollama returns newline-delimited JSON (NDJSON). Each line is:
            {"message": {"content": "token"}, "done": false}
        The final line has "done": true.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.

        Yields:
            Individual text tokens as the model generates them.
        """
        timeout = httpx.Timeout(
            connect=10.0,
            read=float(self._timeout),
            write=10.0,
            pool=5.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        break
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
