"""Transcript model."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import String, DateTime, ForeignKey, JSON, Float, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.document import Document


class Transcript(Base):
    """Transcript segment model."""
    
    __tablename__ = "transcripts"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("organizations.id"), 
        nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("documents.id"), 
        nullable=False
    )
    speaker: Mapped[str | None] = mapped_column(String(100))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
