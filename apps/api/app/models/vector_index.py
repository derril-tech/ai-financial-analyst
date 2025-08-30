"""Vector index model."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import String, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.document import Document


class VectorIndex(Base):
    """Vector index model for embeddings."""
    
    __tablename__ = "vector_index"
    
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
    collection: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))  # OpenAI embedding dimension
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
