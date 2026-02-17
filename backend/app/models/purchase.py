"""Purchase model for Oculus platform."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Purchase(Base):
    """Purchase model for tracking product purchases."""
    
    __tablename__ = "purchases"
    
    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Purchase info
    product_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Stripe info
    stripe_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # Foreign keys
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)
    product_id: Mapped[str] = mapped_column(String(36), ForeignKey("products.uuid"), nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    purchased_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    product: Mapped["Product"] = relationship("Product", back_populates="purchases", foreign_keys=[product_id])
    
    # Indexes
    __table_args__ = (
        Index("idx_purchase_user_id", "user_id"),
        Index("idx_purchase_product_id", "product_id"),
        Index("idx_purchase_stripe_id", "stripe_id"),
    )
    
    def __repr__(self) -> str:
        return f"<Purchase(uuid={self.uuid}, user_id={self.user_id}, product_id={self.product_id})>"

