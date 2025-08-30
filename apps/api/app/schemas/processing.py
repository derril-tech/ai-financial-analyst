"""Processing schemas."""

from typing import Any, List, Dict

from pydantic import BaseModel


class ExtractedTable(BaseModel):
    """Extracted table data."""
    
    page_number: int
    table_index: int
    data: List[Dict[str, Any]]
    headers: List[str]
    confidence: float
    bbox: List[float] | None = None


class ExtractedImage(BaseModel):
    """Extracted image data."""
    
    page_number: int
    image_index: int
    image_data: bytes
    width: int
    height: int
    format: str


class TranscriptSegment(BaseModel):
    """Transcript segment with speaker and timing."""
    
    speaker: str | None = None
    text: str
    start_time: float
    end_time: float
    confidence: float | None = None


class ProcessingResult(BaseModel):
    """Result of document processing."""
    
    document_id: str
    status: str  # completed, failed, skipped
    message: str
    artifacts: List[Any] = []


class XBRLFact(BaseModel):
    """XBRL fact data."""
    
    taxonomy: str
    tag: str
    value: float | str | None
    unit: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    decimals: int | None = None
    context: str | None = None
