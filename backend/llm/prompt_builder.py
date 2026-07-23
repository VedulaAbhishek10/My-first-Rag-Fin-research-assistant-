"""
Prompt builder — constructs the messages list sent to Ollama.

Why does prompt design matter?
  The LLM's answer quality depends heavily on the instructions we give it.
  A vague prompt produces vague answers. Our prompt tells the model:
    1. Its role (financial research assistant).
    2. The golden rule (only use the provided context — no hallucination).
    3. What to do when the context is insufficient.
    4. The context itself (retrieved document chunks with source labels).
    5. The user's question.

  We also inject conversation history so the model can handle follow-up
  questions like "What about 2022?" after asking about 2024.

Message format (Ollama / OpenAI style):
    [
        {"role": "system",    "content": "...instructions..."},
        {"role": "user",      "content": "...previous question..."},   # history
        {"role": "assistant", "content": "...previous answer..."},     # history
        {"role": "user",      "content": "...context + question..."},  # current
    ]

Grounding & Faithfulness:
  The system prompt now includes explicit instructions to:
  - Cite sources inline using [Source: filename, Page: N] notation
  - Refuse to answer when context is insufficient
  - Flag low-confidence answers with a [LOW CONFIDENCE] prefix
  - Never invent numbers, dates, or company names
"""

from backend.vectorstore.chroma_store import SearchResult

_SYSTEM_PROMPT = """You are a Financial Research Assistant. Your job is to answer \
questions about companies, SEC filings, earnings reports, and financial statements.

IMPORTANT RULES:
1. Answer ONLY from the document excerpts provided below — do not use outside knowledge.
2. If the context does not contain enough information, respond with: \
"The provided documents do not contain enough information to answer this question."
3. Always be precise with numbers, dates, and company names \
as they appear in the documents.
4. Do not invent, guess, or extrapolate figures that are not explicitly stated.
5. Keep your answer concise and well-structured.
6. When citing specific facts, include the source in brackets: \
[Source: filename, Page: N].
7. If you are unsure about any claim, prefix your answer with [LOW CONFIDENCE] \
and explain why.
8. Never make up financial figures, dates, or company names — if they are not in \
the context, say so.
9. If the question asks for a comparison but only one company's data is available, \
state that clearly.
10. If the question asks about a time period not covered by the documents, state \
the available time range."""


def _format_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block."""
    if not results:
        return "No relevant document excerpts were found."

    lines: list[str] = []
    for i, result in enumerate(results, start=1):
        source = result.metadata.get("source", "Unknown document")
        page = result.metadata.get("page_number") or "?"
        score = result.score
        lines.append(
            f"[{i}] Source: {source} | Page: {page} | Relevance: {score:.2f}"
        )
        lines.append(f"    {result.text.strip()}")
        lines.append("")
    return "\n".join(lines)


def build_messages(
    question: str,
    results: list[SearchResult],
    history: list[dict],
) -> list[dict]:
    """
    Build the full messages list to send to Ollama.

    Args:
        question: The user's current question.
        results:  Retrieved document chunks from ChromaDB.
        history:  Previous turns from ConversationMemory (may be empty).

    Returns:
        A list of message dicts ready for OllamaClient.chat() or .stream_chat().
    """
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Inject previous conversation turns so the model has context for follow-ups
    messages.extend(history)

    # Current turn: context + question
    context_block = _format_context(results)
    user_content = (
        f"Document excerpts:\n\n{context_block}\n"
        f"Question: {question}"
    )
    messages.append({"role": "user", "content": user_content})

    return messages
