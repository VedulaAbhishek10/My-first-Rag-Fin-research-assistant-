import { useRef, type DragEvent, type ChangeEvent } from 'react';

interface Props {
  onUpload: (file: File) => void;
  isUploading: boolean;
}

export function UploadZone({ onUpload, isUploading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && !isUploading) onUpload(file);
  }

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
    // Reset so the same file can be re-uploaded after an error.
    e.target.value = '';
  }

  return (
    <div
      className={`upload-zone ${isUploading ? 'uploading' : ''}`}
      onDrop={handleDrop}
      onDragOver={e => e.preventDefault()}
      onClick={() => !isUploading && inputRef.current?.click()}
      role="button"
      aria-label="Upload document"
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,.md,.html,.htm"
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      {isUploading ? (
        <p>Ingesting document…</p>
      ) : (
        <>
          <p>Drop a file here or click to upload</p>
          <small>PDF · TXT · MD · HTML</small>
        </>
      )}
    </div>
  );
}
