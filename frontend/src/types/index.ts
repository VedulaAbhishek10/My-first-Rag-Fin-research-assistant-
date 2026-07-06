// TypeScript interfaces that mirror the backend's Pydantic models.
// When the backend shape changes, update here to match.

export interface Citation {
  document_name: string;
  document_id: string;
  page_number: number | null;
  chunk_text: string;
  similarity_score: number;
  company?: string | null;
  ticker?: string | null;
  year?: number | null;
  quarter?: string | null;
  doc_type?: string | null;
}

// One chunk arriving over SSE from POST /api/chat/stream.
// Either a token (content) or the final done=true event (which carries citations).
export interface StreamChunk {
  token: string | null;
  citations: Citation[] | null;
  done: boolean;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  session_id: string;
  processing_time_ms: number;
}

export interface DocumentMetadata {
  company: string | null;
  ticker: string | null;
  year: number | null;
  quarter: string | null;
  doc_type: string;
  source: string | null;
}

export type DocumentStatus = 'PROCESSING' | 'READY' | 'ERROR';

export interface DocumentRecord {
  id: string;
  filename: string;
  file_path: string;
  metadata: DocumentMetadata;
  status: DocumentStatus;
  chunk_count: number;
  created_at: string;
  error_message: string | null;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  status: string;
  message: string;
}

// Optional metadata filters sent with a query (M5).
// Mirrors backend SearchFilters. A field left undefined means "don't filter".
export interface SearchFilters {
  company?: string;
  ticker?: string;
  year?: number;
  quarter?: string;
  doc_type?: string;
}

// A single message in the chat UI. Not from the backend — built locally.
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  isStreaming?: boolean;
}
