"""Upload schemas."""

from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Document metadata for upload."""
    
    title: str | None = None
    ticker: str | None = Field(None, max_length=10)
    fiscal_year: int | None = Field(None, ge=1900, le=2100)
    fiscal_period: str | None = Field(None, regex=r"^(Q1|Q2|Q3|Q4|FY)$")
    language: str | None = Field("en", max_length=10)
    uploader: str | None = None


class UploadResponse(BaseModel):
    """Upload response."""
    
    document_id: str
    org_id: str
    kind: str
    title: str
    ticker: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    language: str
    checksum: str
    path_s3: str
    meta: dict[str, Any]


class PresignedUrlRequest(BaseModel):
    """Presigned URL request."""
    
    filename: str
    content_type: str | None = None
    expires_in: int = Field(3600, ge=60, le=86400)


class PresignedUrlResponse(BaseModel):
    """Presigned URL response."""
    
    url: str
    document_id: str
    expires_in: int
