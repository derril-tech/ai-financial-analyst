"""Organization model."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.document import Document


class Organization(Base):
    """Organization model."""
    
    __tablename__ = "organizations"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    users: Mapped[list["User"]] = relationship(
        "User", 
        back_populates="organization"
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", 
        back_populates="organization"
    )
