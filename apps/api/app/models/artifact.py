"""Artifact model (separate from document.py to avoid circular imports)."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import String, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.document import Document


class Artifact(Base):
    """Artifact model for processed document outputs."""
    
    __tablename__ = "artifacts"
    
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
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # table, image, text, etc.
    path_s3: Mapped[str] = mapped_column(String(500), nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
