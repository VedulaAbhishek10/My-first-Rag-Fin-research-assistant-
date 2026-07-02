"""
Chat API routes — the question-answering interface.

Endpoints:
    POST /api/chat/query   — full answer at once (non-streaming)
    POST /api/chat/stream  — token-by-token SSE stream
    DELETE /api/chat/sessions/{session_id} — clear conversation history

Why two endpoints?
  /query  is simple to test with curl and useful for scripts.
  /stream is what the React UI uses — it shows tokens as they arrive,
          making the app feel instant instead of waiting 5-10 seconds.

Server-Sent Events (SSE) format:
  Each chunk is sent as:
      data: {"token": "word", "citations": null, "done": false}\n\n
  The final chunk:
      data: {"token": null, "citations": [...], "done": true}\n\n
"""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_chat_service
from backend.logging_config import get_logger
from backend.models.query import QueryRequest, QueryResponse
from backend.services.chat_service import ChatService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question (non-streaming)",
)
async def query(
    request: QueryRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> QueryResponse:
    """
    Run the full RAG pipeline and return the complete answer with citations.

    Use this endpoint for scripts, testing, or clients that don't support SSE.
    The response arrives only after the LLM finishes generating (may take 5-30s).
    """
    _check_ollama(chat_service)
    return await chat_service.query(
        question=request.question,
        session_id=request.session_id or str(uuid.uuid4()),
        top_k=request.top_k,
        filters=request.filters,
    )


@router.post(
    "/stream",
    summary="Ask a question (streaming SSE)",
    response_description="Server-Sent Events stream of tokens + final citations",
)
async def stream_query(
    request: QueryRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Stream the RAG answer token by token using Server-Sent Events.

    The client receives individual tokens as they are generated, then a final
    event containing all citations once generation is complete.

    Connect with:
        const es = new EventSource('/api/chat/stream');   (GET-only)
        // For POST+SSE use fetch() with a ReadableStream reader in the UI.
    """
    _check_ollama(chat_service)
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            async for chunk in chat_service.stream_query(
                question=request.question,
                session_id=session_id,
                top_k=request.top_k,
                filters=request.filters,
            ):
                yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as exc:
            logger.exception("Streaming error: %s", exc)
            error_payload = json.dumps({"error": str(exc), "done": True})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind proxy
        },
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear conversation history for a session",
)
async def clear_session(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service),
) -> None:
    """Wipe the conversation history for the given session ID."""
    chat_service._memory.clear(session_id)
    logger.info("Cleared session: %s", session_id[:8])


# ── Helper ────────────────────────────────────────────────────────────────────


def _check_ollama(chat_service: ChatService) -> None:
    """Raise 503 with a clear message if Ollama is not reachable."""
    if not chat_service._llm.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Ollama is not running. Start it with: ollama serve\n"
                "Then pull the model: ollama pull qwen2.5-coder:3b-instruct"
            ),
        )
