"""Balance model for Oculus platform."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Balance(Base):
    """Balance model for tracking user token balances and transactions."""
    
    __tablename__ = "balances"
    
    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Balance info
    tokens: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "purchase", "deduction", "refund", "earning"
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Foreign key
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    
    # Indexes
    __table_args__ = (
        Index("idx_balance_user_id", "user_id"),
        Index("idx_balance_transaction_type", "transaction_type"),
    )
    
    def __repr__(self) -> str:
        return f"<Balance(uuid={self.uuid}, user_id={self.user_id}, tokens={self.tokens}, transaction_type={self.transaction_type})>"

