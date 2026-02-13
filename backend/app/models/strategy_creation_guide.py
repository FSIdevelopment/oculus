"""StrategyCreationGuide model for storing per-strategy-type creation knowledge."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Integer, DateTime, Text, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class StrategyCreationGuide(Base):
    """Stores creation guide content per strategy type.

    One row per strategy type (e.g. "general", "Momentum", "Mean Reversion").
    The "general" row holds cross-cutting lessons applicable to all strategies.
    Strategy-specific rows hold type-specific knowledge (best indicators, pitfalls, etc.).
    No user_id — this is a global, shared guide.
    """

    __tablename__ = "strategy_creation_guides"

    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))

    # Strategy type — exact human-readable string from frontend dropdown
    # e.g. "general", "Momentum", "Mean Reversion", "Bollinger Band Breakout"
    strategy_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Full markdown content for this strategy type
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Optimistic locking version — incremented on each update to prevent concurrent overwrites
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Table constraints and indexes
    __table_args__ = (
        UniqueConstraint("strategy_type", name="uq_strategy_creation_guide_type"),
        Index("idx_guide_strategy_type", "strategy_type"),
    )

    def __repr__(self) -> str:
        return f"<StrategyCreationGuide(uuid={self.uuid}, strategy_type={self.strategy_type}, version={self.version})>"

