// useDocuments — manages the document list and file upload state.

import { useState, useEffect, useCallback } from 'react';
import { listDocuments, uploadDocument, deleteDocument } from '../api/documents';
import type { DocumentRecord } from '../types';

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch {
      // Silently ignore list errors — backend may be starting up.
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Fetch the document list once on mount.
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const upload = useCallback(
    async (file: File) => {
      setIsUploading(true);
      setUploadError(null);
      try {
        await uploadDocument(file);
        await fetchDocuments();
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : 'Upload failed');
      } finally {
        setIsUploading(false);
      }
    },
    [fetchDocuments],
  );

  const removeDocument = useCallback(
    async (documentId: string) => {
      try {
        await deleteDocument(documentId);
        setDocuments(prev => prev.filter(d => d.id !== documentId));
      } catch (err) {
        // Optionally handle error state here
        console.error('Failed to delete document:', err);
      }
    },
    [],
  );

  return { documents, isLoading, isUploading, uploadError, upload, fetchDocuments, removeDocument };
}
