// useChat — manages the chat session: messages list, streaming state, errors.
//
// Why useRef for sessionId?
//   We want sessionId to persist across renders without causing a re-render
//   when it changes (e.g. after "New Session"). useRef is the right tool for
//   mutable values that don't drive rendering.

import { useState, useCallback, useRef } from 'react';
import { streamQuery, clearSession } from '../api/chat';
import type { Message, SearchFilters } from '../types';

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Each browser tab gets its own session with the backend conversation memory.
  const sessionId = useRef<string>(crypto.randomUUID());

  const sendMessage = useCallback(
    async (question: string, filters?: SearchFilters) => {
      if (isStreaming || !question.trim()) return;

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: 'user',
        content: question,
      };

      // Placeholder for the assistant's streaming response.
      const assistantId = crypto.randomUUID();
      const assistantMsg: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        isStreaming: true,
      };

      setMessages(prev => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);
      setError(null);

      try {
        for await (const chunk of streamQuery(question, sessionId.current, filters)) {
          if (chunk.done) {
            // Final event — attach citations and mark streaming complete.
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, isStreaming: false, citations: chunk.citations ?? [] }
                  : m,
              ),
            );
          } else if (chunk.token) {
            // Append each arriving token to the assistant message.
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, content: m.content + chunk.token }
                  : m,
              ),
            );
          }
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Streaming failed';
        setError(message);
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, isStreaming: false, content: `Error: ${message}` }
              : m,
          ),
        );
      } finally {
        setIsStreaming(false);
      }
    },
    [isStreaming],
  );

  const retryLastMessage = useCallback(async () => {
    if (isStreaming) return;

    // Find the last user message
    const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
    if (!lastUserMsg) return;

    // Remove the last assistant message (which contains the error)
    setMessages(prev => {
      const newMessages = [...prev];
      if (newMessages.length > 0 && newMessages[newMessages.length - 1].role === 'assistant') {
        newMessages.pop();
      }
      return newMessages;
    });

    // Resend the question
    await sendMessage(lastUserMsg.content);
  }, [isStreaming, messages, sendMessage]);

  const clearHistory = useCallback(async () => {
    await clearSession(sessionId.current).catch(() => {});
    sessionId.current = crypto.randomUUID();
    setMessages([]);
    setError(null);
  }, []);

  return { messages, isStreaming, error, sendMessage, clearHistory, retryLastMessage };
}
