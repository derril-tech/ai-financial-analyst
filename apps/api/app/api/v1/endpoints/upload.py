"""Upload endpoints."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.observability import trace_function
from app.models.document import Document
from app.schemas.upload import (
    DocumentMetadata,
    UploadResponse,
    PresignedUrlRequest,
    PresignedUrlResponse,
)
from app.services.upload import UploadService

router = APIRouter()


@router.post("/", response_model=UploadResponse)
@trace_function("upload_endpoint.upload_file")
async def upload_file(
    file: UploadFile = File(...),
    org_id: str = Form(...),
    title: str | None = Form(None),
    ticker: str | None = Form(None),
    fiscal_year: int | None = Form(None),
    fiscal_period: str | None = Form(None),
    language: str | None = Form("en"),
    uploader: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """Upload a file."""
    # Create metadata
    metadata = DocumentMetadata(
        title=title,
        ticker=ticker,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        language=language,
        uploader=uploader,
    )
    
    # Upload file
    upload_service = UploadService()
    upload_response = await upload_service.upload_file(file, org_id, metadata)
    
    # Create document record
    document = Document(
        id=upload_response.document_id,
        org_id=upload_response.org_id,
        kind=upload_response.kind,
        title=upload_response.title,
        ticker=upload_response.ticker,
        fiscal_year=upload_response.fiscal_year,
        fiscal_period=upload_response.fiscal_period,
        language=upload_response.language,
        checksum=upload_response.checksum,
        path_s3=upload_response.path_s3,
        meta=upload_response.meta,
    )
    
    db.add(document)
    await db.commit()
    
    return upload_response


@router.post("/presigned", response_model=PresignedUrlResponse)
@trace_function("upload_endpoint.get_presigned_url")
async def get_presigned_url(
    request: PresignedUrlRequest,
    org_id: str,
) -> PresignedUrlResponse:
    """Get presigned URL for direct upload."""
    upload_service = UploadService()
    url = upload_service.get_presigned_url(
        org_id=org_id,
        filename=request.filename,
        expires_in=request.expires_in,
    )
    
    # Extract document ID from URL (simplified)
    import uuid
    document_id = str(uuid.uuid4())
    
    return PresignedUrlResponse(
        url=url,
        document_id=document_id,
        expires_in=request.expires_in,
    )


@router.get("/{document_id}/download")
@trace_function("upload_endpoint.download_file")
async def download_file(
    document_id: str,
    org_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Get download URL for a document."""
    # Get document
    result = await db.execute(
        "SELECT path_s3 FROM documents WHERE id = :id AND org_id = :org_id",
        {"id": document_id, "org_id": org_id}
    )
    document = result.fetchone()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get download URL
    upload_service = UploadService()
    url = upload_service.get_download_url(document.path_s3)
    
    return {"download_url": url}
