import type { DocumentRecord, DocumentStatus } from '../../types';

interface Props {
  documents: DocumentRecord[];
  isLoading: boolean;
}

const STATUS_CLASS: Record<DocumentStatus, string> = {
  READY: 'status-ready',
  PROCESSING: 'status-processing',
  ERROR: 'status-error',
};

export function DocumentList({ documents, isLoading }: Props) {
  if (isLoading) return <p className="doc-list-info">Loading…</p>;
  if (documents.length === 0) {
    return <p className="doc-list-info">No documents yet.</p>;
  }

  return (
    <ul className="document-list">
      {documents.map(doc => (
        <li key={doc.id} className="document-item">
          <span className="doc-name" title={doc.filename}>
            {doc.filename}
          </span>
          <div className="doc-meta">
            <span className={`doc-status ${STATUS_CLASS[doc.status] ?? ''}`}>
              {doc.status}
            </span>
            {doc.chunk_count > 0 && (
              <span className="doc-chunks">{doc.chunk_count} chunks</span>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
