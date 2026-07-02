// Chat API — wraps the backend's /api/chat/* endpoints.
//
// Why fetch() instead of EventSource for SSE?
//   EventSource only supports GET requests. Our /api/chat/stream endpoint is POST
//   (it needs a request body with the question). So we use fetch() with a
//   ReadableStream reader and manually parse the "data: ...\n\n" SSE lines.

import type { SearchFilters, StreamChunk } from '../types';

const BASE = '/api/chat';

// Drop undefined/empty fields so we only send active filters to the backend.
// An all-empty object becomes undefined, which the backend reads as "no filter".
function cleanFilters(filters?: SearchFilters): SearchFilters | undefined {
  if (!filters) return undefined;
  const active = Object.fromEntries(
    Object.entries(filters).filter(([, v]) => v !== undefined && v !== ''),
  );
  return Object.keys(active).length > 0 ? active : undefined;
}

// Streams tokens from POST /api/chat/stream.
// Yields one StreamChunk per SSE event. The last chunk has done=true and citations.
export async function* streamQuery(
  question: string,
  sessionId: string,
  filters?: SearchFilters,
  topK = 5,
): AsyncGenerator<StreamChunk> {
  const response = await fetch(`${BASE}/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      top_k: topK,
      filters: cleanFilters(filters),
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail ?? `HTTP ${response.status}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // Decode the incoming bytes and accumulate in buffer.
    // { stream: true } tells the decoder not to flush incomplete multi-byte chars.
    buffer += decoder.decode(value, { stream: true });

    // SSE lines are split by "\n". We keep the last (possibly incomplete) line
    // in the buffer and process everything before it.
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const json = line.slice(6).trim();
        if (json) {
          yield JSON.parse(json) as StreamChunk;
        }
      }
    }
  }
}

export async function clearSession(sessionId: string): Promise<void> {
  await fetch(`${BASE}/sessions/${sessionId}`, { method: 'DELETE' });
}
