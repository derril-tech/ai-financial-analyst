"""Document and artifact models."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import String, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.organization import Organization


class Document(Base):
    """Document model for uploaded files."""
    
    __tablename__ = "documents"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("organizations.id"), 
        nullable=False
    )
    kind: Mapped[str] = mapped_column(String(50), nullable=False)  # pdf, audio, xbrl, etc.
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    ticker: Mapped[str | None] = mapped_column(String(10))
    fiscal_year: Mapped[int | None] = mapped_column()
    fiscal_period: Mapped[str | None] = mapped_column(String(10))  # Q1, Q2, Q3, Q4, FY
    language: Mapped[str] = mapped_column(String(10), default="en")
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    path_s3: Mapped[str] = mapped_column(String(500), nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    organization: Mapped["Organization"] = relationship(
        "Organization", 
        back_populates="documents"
    )
    artifacts: Mapped[list["Artifact"]] = relationship(
        "Artifact", 
        back_populates="document"
    )


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
    
    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", 
        back_populates="artifacts"
    )
