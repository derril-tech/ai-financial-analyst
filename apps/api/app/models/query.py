"""Query and answer models."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import String, DateTime, ForeignKey, JSON, Text, Float, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.user import User


class Query(Base):
    """Query model."""
    
    __tablename__ = "queries"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("organizations.id"), 
        nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("users.id"), 
        nullable=False
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    plan: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    answers: Mapped[list["Answer"]] = relationship(
        "Answer", 
        back_populates="query"
    )


class Answer(Base):
    """Answer model."""
    
    __tablename__ = "answers"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    org_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("organizations.id"), 
        nullable=False
    )
    query_id: Mapped[str] = mapped_column(
        String, 
        ForeignKey("queries.id"), 
        nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    citations: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    exports: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    query: Mapped["Query"] = relationship(
        "Query", 
        back_populates="answers"
    )
