import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { useChat } from '../../hooks/useChat';

export function ChatPanel() {
  const { messages, isStreaming, error, sendMessage, clearHistory } = useChat();

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

      {error && <div className="error-banner">{error}</div>}

      <MessageList messages={messages} />
      <ChatInput onSend={sendMessage} disabled={isStreaming} />
    </div>
  );
}
