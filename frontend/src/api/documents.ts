// Document API — wraps /api/documents/* endpoints.

import type { DocumentRecord, UploadResponse } from '../types';

const BASE = '/api/documents';

export async function listDocuments(): Promise<DocumentRecord[]> {
  const response = await fetch(`${BASE}/`);
  if (!response.ok) throw new Error('Failed to fetch documents');
  return response.json();
}

// Uploads a file via multipart/form-data to POST /api/documents/upload.
// The backend parses, chunks, embeds, and stores it in ChromaDB + SQLite.
export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);

  const response = await fetch(`${BASE}/upload`, {
    method: 'POST',
    body: form,
    // Do NOT set Content-Type — the browser sets it automatically with the
    // correct multipart boundary when using FormData.
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail ?? `HTTP ${response.status}`);
  }

  return response.json();
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`${BASE}/${documentId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Delete failed' }));
    throw new Error(err.detail ?? `HTTP ${response.status}`);
  }
}
