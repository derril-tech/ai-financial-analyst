"""XBRL fact model."""

from datetime import datetime, date
from typing import TYPE_CHECKING, Any

from sqlalchemy import String, DateTime, Date, ForeignKey, JSON, Float, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.document import Document


class FactXBRL(Base):
    """XBRL fact model."""
    
    __tablename__ = "facts_xbrl"
    
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
    taxonomy: Mapped[str] = mapped_column(String(100), nullable=False)
    tag: Mapped[str] = mapped_column(String(200), nullable=False)
    value_num: Mapped[float | None] = mapped_column(Float)
    value_text: Mapped[str | None] = mapped_column(String(1000))
    unit: Mapped[str | None] = mapped_column(String(50))
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date | None] = mapped_column(Date)
    decimals: Mapped[int | None] = mapped_column(Integer)
    restated_from_id: Mapped[str | None] = mapped_column(
        String, 
        ForeignKey("facts_xbrl.id")
    )
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
