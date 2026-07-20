"""
Document management API routes.

Endpoints:
    POST /api/documents/upload   — upload and ingest a document
    GET  /api/documents/         — list all documents
    GET  /api/documents/{id}     — get one document's status
    DELETE /api/documents/{id}   — delete a document
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.api.dependencies import get_database, get_ingestion_pipeline, get_vector_store
from backend.config import get_settings
from backend.database.sqlite_db import SQLiteDatabase
from backend.ingestion.pipeline import IngestionPipeline
from backend.logging_config import get_logger
from backend.models.document import DocumentRecord, DocumentStatus, UploadResponse
from backend.vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])

ALLOWED_EXTENSIONS: frozenset[str] = frozenset([".pdf", ".txt", ".html", ".htm", ".md"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a financial document",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, or HTML document to ingest"),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    database: SQLiteDatabase = Depends(get_database),
) -> UploadResponse:
    """
    Upload a document, run the full ingestion pipeline, and return the result.

    The pipeline: parse → extract metadata → chunk → embed → store in ChromaDB + SQLite.
    This runs synchronously — the response arrives once ingestion is complete.
    """
    settings = get_settings()
    filename = file.filename or "upload"

    # Validate extension
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File type '{suffix}' is not supported. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    # Read and validate size
    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_size_mb} MB limit.",
        )

    # Save to upload directory
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / filename
    dest_path.write_bytes(content)
    logger.info("Saved uploaded file: %s (%d bytes)", dest_path, len(content))

    # Run pipeline
    record = pipeline.ingest(file_path=dest_path, filename=filename)

    message = (
        f"Successfully ingested {record.chunk_count} chunks."
        if record.status == DocumentStatus.READY
        else f"Ingestion failed: {record.error_message}"
    )
    return UploadResponse(
        document_id=record.id,
        filename=record.filename,
        status=record.status,
        message=message,
    )


@router.get(
    "/",
    response_model=list[DocumentRecord],
    summary="List all ingested documents",
)
async def list_documents(
    database: SQLiteDatabase = Depends(get_database),
) -> list[DocumentRecord]:
    """Return all documents in the registry, newest first."""
    return database.list_documents()


@router.get(
    "/{document_id}",
    response_model=DocumentRecord,
    summary="Get a specific document's status and metadata",
)
async def get_document(
    document_id: str,
    database: SQLiteDatabase = Depends(get_database),
) -> DocumentRecord:
    """Fetch one document by its UUID."""
    record = database.get_document(document_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return record


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
async def delete_document(
    document_id: str,
    database: SQLiteDatabase = Depends(get_database),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
) -> None:
    """Delete a document from the registry and vector store."""
    deleted = database.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    
    # Also remove from ChromaDB
    try:
        vector_store.delete_document(document_id)
    except Exception as e:
        logger.error("Failed to delete document %s from vector store: %s", document_id, e)
        # We don't raise an error here because the document is already deleted from SQLite
