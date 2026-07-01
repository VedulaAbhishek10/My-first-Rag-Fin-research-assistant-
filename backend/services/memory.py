"""
In-memory conversation history store.

What is conversation memory?
  Without memory, every question is treated as a fresh conversation. The user
  can't ask "What about Q2?" as a follow-up because the model has forgotten
  what they were talking about. Memory stores the recent Q&A exchanges for
  each session so the LLM gets context for follow-up questions.

Phase 1 implementation: plain Python dict (in-memory, lost on restart).
Phase 2 plan: persist to SQLite so history survives server restarts.

Why cap at max_turns?
  Conversation history is included in every prompt. If history grows too large
  it will exceed the model's context window. Capping at 10 turns (~2000 tokens)
  keeps prompts well within the limit for a 3B parameter model.
"""

from collections import defaultdict

from backend.logging_config import get_logger

logger = get_logger(__name__)

# Default number of past Q&A exchanges to retain per session.
_DEFAULT_MAX_TURNS = 10


class ConversationMemory:
    """Stores recent conversation history for each session."""

    def __init__(self, max_turns: int = _DEFAULT_MAX_TURNS) -> None:
        # Each session maps to a list of {"role": ..., "content": ...} dicts
        self._history: dict[str, list[dict]] = defaultdict(list)
        self._max_turns = max_turns  # one turn = one user + one assistant message

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """
        Record one complete Q&A exchange.

        Args:
            session_id: Identifies the conversation.
            question:   What the user asked.
            answer:     What the assistant replied.
        """
        history = self._history[session_id]
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        # Trim to the last max_turns exchanges (2 messages per turn)
        max_messages = self._max_turns * 2
        if len(history) > max_messages:
            self._history[session_id] = history[-max_messages:]

        logger.debug(
            "Memory updated: session=%s turns=%d",
            session_id[:8],
            len(self._history[session_id]) // 2,
        )

    def get_history(self, session_id: str) -> list[dict]:
        """Return the stored message history for a session (may be empty)."""
        return list(self._history.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Wipe the history for a session (e.g. user clicks 'New Chat')."""
        self._history.pop(session_id, None)
        logger.debug("Memory cleared for session=%s", session_id[:8])

    def session_count(self) -> int:
        """Number of active sessions (useful for monitoring)."""
        return len(self._history)
