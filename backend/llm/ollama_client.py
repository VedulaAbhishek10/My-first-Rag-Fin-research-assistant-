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

Observability:
  Every request is logged with message count, response length, and latency
  so performance regressions and failed calls are visible in production.
"""

import json
import time
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
        start = time.perf_counter()
        logger.info(
            "LLM request (non-streaming) | model=%s | messages=%d",
            self._model,
            len(messages),
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": messages, "stream": False},
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"]

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "LLM response | model=%s | length=%d | latency=%.0fms",
            self._model,
            len(answer),
            elapsed,
        )
        return answer

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
        start = time.perf_counter()
        token_count = 0
        logger.info(
            "LLM stream request | model=%s | messages=%d",
            self._model,
            len(messages),
        )

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
                        token_count += 1
                        yield token

        elapsed = (time.perf_counter() - start) * 1000
        tps = round(token_count / (elapsed / 1000), 1) if elapsed > 0 else 0
        logger.info(
            "LLM stream done | model=%s | tokens=%d | latency=%.0fms | tps=%.1f",
            self._model,
            token_count,
            elapsed,
            tps,
        )
