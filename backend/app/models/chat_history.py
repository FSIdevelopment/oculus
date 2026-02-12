"""ChatHistory model for Oculus platform."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, DateTime, Text, JSON, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class ChatHistory(Base):
    """ChatHistory model for storing chat messages."""
    
    __tablename__ = "chat_history"
    
    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Message info
    message: Mapped[str] = mapped_column(Text, nullable=False)
    message_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "user", "assistant", "system"
    content: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Foreign keys
    strategy_id: Mapped[str] = mapped_column(String(36), ForeignKey("strategies.uuid"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="chat_history", foreign_keys=[strategy_id])
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    
    # Indexes
    __table_args__ = (
        Index("idx_chat_strategy_id", "strategy_id"),
        Index("idx_chat_user_id", "user_id"),
    )
    
    def __repr__(self) -> str:
        return f"<ChatHistory(uuid={self.uuid}, strategy_id={self.strategy_id}, message_type={self.message_type})>"

