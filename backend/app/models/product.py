"""Product model for Oculus platform."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Integer, Float, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Product(Base):
    """Product model for tokens and licenses."""
    
    __tablename__ = "products"
    
    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Product info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    product_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "token_package", "license"
    
    # Token info
    token_amount: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Stripe info
    stripe_product_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stripe_price_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    purchases: Mapped[list["Purchase"]] = relationship("Purchase", back_populates="product")
    
    # Indexes
    __table_args__ = (
        Index("idx_product_type", "product_type"),
        Index("idx_product_stripe_id", "stripe_product_id"),
    )
    
    def __repr__(self) -> str:
        return f"<Product(uuid={self.uuid}, name={self.name}, product_type={self.product_type})>"

