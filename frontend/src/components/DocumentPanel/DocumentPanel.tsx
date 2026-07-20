import { UploadZone } from './UploadZone';
import { DocumentList } from './DocumentList';
import type { useDocuments } from '../../hooks/useDocuments';

interface Props {
  // The shared document state, lifted to App so the ChatPanel filters stay in
  // sync with uploads. Typed as the useDocuments hook's return value.
  docs: ReturnType<typeof useDocuments>;
}

export function DocumentPanel({ docs }: Props) {
  const {
  documents,
  isLoading,
  isUploading,
  uploadError,
  upload,
  fetchDocuments,
  removeDocument,
} = docs;

  return (
    <div className="document-panel">
      <div className="document-panel-header">
        <h2>Documents</h2>
        <button
          className="btn-icon"
          onClick={fetchDocuments}
          title="Refresh document list"
        >
          ↻
        </button>
      </div>

      <UploadZone onUpload={upload} isUploading={isUploading} />

      {uploadError && <p className="upload-error">{uploadError}</p>}

      <DocumentList
  documents={documents}
  isLoading={isLoading}
  onDelete={removeDocument}
/>
    </div>
  );
}
