import { UploadZone } from './UploadZone';
import { DocumentList } from './DocumentList';
import { useDocuments } from '../../hooks/useDocuments';

export function DocumentPanel() {
  const { documents, isLoading, isUploading, uploadError, upload, fetchDocuments } =
    useDocuments();

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

      <DocumentList documents={documents} isLoading={isLoading} />
    </div>
  );
}
