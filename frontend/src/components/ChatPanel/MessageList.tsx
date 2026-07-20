import { useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import type { Message } from '../../types';

interface Props {
  messages: Message[];
  onRetry: () => void;
}

export function MessageList({ messages, onRetry }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the latest message whenever the list changes.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="message-list empty">
        <p>
          Upload a financial document using the panel on the left,
          then ask a question here.
        </p>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map(m => (
        <div key={m.id}>
          <MessageBubble message={m} />
          {m.role === 'assistant' && m.content.startsWith('Error:') && (
            <button className="btn-secondary" onClick={onRetry}>
              Retry
            </button>
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
