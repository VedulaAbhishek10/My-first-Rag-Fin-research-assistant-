import { useState } from 'react';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { FilterBar } from './FilterBar';
import { useChat } from '../../hooks/useChat';
import type { DocumentRecord, SearchFilters } from '../../types';

interface Props {
  // Passed from App so the filter dropdowns reflect the current document set.
  documents: DocumentRecord[];
}

export function ChatPanel({ documents }: Props) {
  const { messages, isStreaming, error, sendMessage, clearHistory } = useChat();
  const [filters, setFilters] = useState<SearchFilters>({});

  return (
    <div className="chat-panel">
      <div className="chat-panel-header">
        <h2>Chat</h2>
        <button
          className="btn-secondary"
          onClick={clearHistory}
          disabled={isStreaming}
          title="Start a new conversation (clears history)"
        >
          New Session
        </button>
      </div>

      <FilterBar
        documents={documents}
        filters={filters}
        onChange={setFilters}
        disabled={isStreaming}
      />

      {error && <div className="error-banner">{error}</div>}

      <MessageList messages={messages} />
      <ChatInput onSend={question => sendMessage(question, filters)} disabled={isStreaming} />
    </div>
  );
}
