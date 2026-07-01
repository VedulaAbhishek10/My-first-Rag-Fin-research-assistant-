import { useState, type KeyboardEvent } from 'react';

interface Props {
  onSend: (question: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState('');

  function submit() {
    const q = value.trim();
    if (q && !disabled) {
      onSend(q);
      setValue('');
    }
  }

  // Send on Enter (plain), allow Shift+Enter for newlines.
  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="chat-input">
      <textarea
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about your financial documents… (Enter to send, Shift+Enter for newline)"
        disabled={disabled}
        rows={3}
      />
      <button onClick={submit} disabled={disabled || !value.trim()}>
        {disabled ? 'Thinking…' : 'Send'}
      </button>
    </div>
  );
}
