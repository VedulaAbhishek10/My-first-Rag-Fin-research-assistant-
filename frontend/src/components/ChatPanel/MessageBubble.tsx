import { CitationCard } from '../Citations/CitationCard';
import type { Message } from '../../types';

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user';

  return (
    <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
      <div className="bubble-role">{isUser ? 'You' : 'Assistant'}</div>

      <div className="bubble-content">
        {message.content}
        {/* Blinking cursor while the assistant is still streaming */}
        {message.isStreaming && <span className="cursor">▋</span>}
      </div>

      {message.citations && message.citations.length > 0 && (
        <div className="bubble-citations">
          <div className="citations-label">Sources</div>
          {message.citations.map((c, i) => (
            <CitationCard key={`${c.document_id}-${i}`} citation={c} index={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
