"""Export schemas."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ExportRequest(BaseModel):
    """Export request schema."""
    
    format: str = Field(..., regex="^(excel|xlsx|pptx|pdf)$")
    data: Dict[str, Any]
    options: Optional[Dict[str, Any]] = None


class ExportResponse(BaseModel):
    """Export response schema."""
    
    export_id: str
    filename: str
    format: str
    status: str  # pending, processing, completed, failed
    download_url: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
