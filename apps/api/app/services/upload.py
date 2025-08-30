"""Upload service for handling file uploads."""

import hashlib
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile, HTTPException
from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.core.observability import trace_function
from app.models.document import Document
from app.schemas.upload import UploadResponse, DocumentMetadata


class UploadService:
    """Service for handling file uploads."""
    
    def __init__(self) -> None:
        """Initialize upload service."""
        self.minio_client = Minio(
            settings.S3_ENDPOINT.replace("http://", "").replace("https://", ""),
            access_key=settings.S3_ACCESS_KEY,
            secret_key=settings.S3_SECRET_KEY,
            secure=settings.S3_ENDPOINT.startswith("https://"),
        )
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """Ensure the S3 bucket exists."""
        try:
            if not self.minio_client.bucket_exists(settings.S3_BUCKET):
                self.minio_client.make_bucket(settings.S3_BUCKET)
        except S3Error as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create bucket: {e}"
            )
    
    @trace_function("upload_service.upload_file")
    async def upload_file(
        self,
        file: UploadFile,
        org_id: str,
        metadata: DocumentMetadata,
    ) -> UploadResponse:
        """Upload file to S3 and create document record."""
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()
        
        # Generate document ID and S3 path
        doc_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        s3_path = f"{org_id}/raw/{doc_id}{file_extension}"
        
        try:
            # Upload to S3
            self.minio_client.put_object(
                bucket_name=settings.S3_BUCKET,
                object_name=s3_path,
                data=content,
                length=len(content),
                content_type=file.content_type or "application/octet-stream",
            )
            
            # Determine document kind from file extension
            kind = self._get_document_kind(file_extension)
            
            # Create document metadata
            document_meta = {
                "original_filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "uploader": metadata.uploader if metadata.uploader else "system",
            }
            
            return UploadResponse(
                document_id=doc_id,
                org_id=org_id,
                kind=kind,
                title=metadata.title or file.filename,
                ticker=metadata.ticker,
                fiscal_year=metadata.fiscal_year,
                fiscal_period=metadata.fiscal_period,
                language=metadata.language or "en",
                checksum=checksum,
                path_s3=s3_path,
                meta=document_meta,
            )
            
        except S3Error as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload file: {e}"
            )
    
    def _get_document_kind(self, file_extension: str) -> str:
        """Determine document kind from file extension."""
        extension_map = {
            ".pdf": "pdf",
            ".mp3": "audio",
            ".mp4": "video",
            ".wav": "audio",
            ".m4a": "audio",
            ".xlsx": "spreadsheet",
            ".xls": "spreadsheet",
            ".csv": "spreadsheet",
            ".pptx": "presentation",
            ".ppt": "presentation",
            ".xml": "xbrl",
            ".xbrl": "xbrl",
            ".htm": "html",
            ".html": "html",
        }
        return extension_map.get(file_extension.lower(), "unknown")
    
    @trace_function("upload_service.get_presigned_url")
    def get_presigned_url(
        self,
        org_id: str,
        filename: str,
        expires_in: int = 3600,
    ) -> str:
        """Get presigned URL for direct upload."""
        doc_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix
        s3_path = f"{org_id}/raw/{doc_id}{file_extension}"
        
        try:
            url = self.minio_client.presigned_put_object(
                bucket_name=settings.S3_BUCKET,
                object_name=s3_path,
                expires=expires_in,
            )
            return url
        except S3Error as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate presigned URL: {e}"
            )
    
    @trace_function("upload_service.get_download_url")
    def get_download_url(
        self,
        s3_path: str,
        expires_in: int = 3600,
    ) -> str:
        """Get presigned URL for download."""
        try:
            url = self.minio_client.presigned_get_object(
                bucket_name=settings.S3_BUCKET,
                object_name=s3_path,
                expires=expires_in,
            )
            return url
        except S3Error as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate download URL: {e}"
            )
