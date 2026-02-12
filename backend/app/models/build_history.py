"""BuildHistory and FeatureTracker models for Oculus platform."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class BuildHistory(Base):
    """BuildHistory model for tracking strategy build records and cross-build memory."""
    
    __tablename__ = "build_history"
    
    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Build metadata
    strategy_name: Mapped[str] = mapped_column(String(255), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(50), nullable=False)  # "tech", "semiconductors", "energy", etc.
    timeframe: Mapped[str] = mapped_column(String(50), nullable=False)  # "1d", "1h", etc.
    target: Mapped[float] = mapped_column(Float, nullable=False)  # Target return percentage
    
    # Symbols (stored as JSON array)
    symbols: Mapped[list] = mapped_column(JSON, nullable=False)
    
    # Build results
    success: Mapped[bool] = mapped_column(default=False)  # Whether build hit target
    build_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Backtest results (stored as JSON)
    backtest_results: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    # Model info (stored as JSON)
    model_info: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    # Risk parameters (stored as JSON)
    risk_params: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    # Features used (stored as JSON)
    features: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    # Iteration history (stored as JSON)
    iterations: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    # Foreign key
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_build_history_user_id", "user_id"),
        Index("idx_build_history_strategy_name", "strategy_name"),
        Index("idx_build_history_asset_class", "asset_class"),
        Index("idx_build_history_build_date", "build_date"),
    )
    
    def __repr__(self) -> str:
        return f"<BuildHistory(uuid={self.uuid}, strategy_name={self.strategy_name}, success={self.success})>"


class FeatureTracker(Base):
    """FeatureTracker model for aggregated feature effectiveness data."""
    
    __tablename__ = "feature_tracker"
    
    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Feature name
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    
    # Usage statistics
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance data (stored as JSON)
    # Contains: returns_when_used (list), asset_classes (list), etc.
    performance_data: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    # Entry rule statistics (stored as JSON)
    # Contains: count, thresholds, operators
    entry_rule_stats: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    # Exit rule statistics (stored as JSON)
    exit_rule_stats: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    # Last updated
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index("idx_feature_tracker_name", "feature_name"),
        Index("idx_feature_tracker_updated_at", "updated_at"),
    )
    
    def __repr__(self) -> str:
        return f"<FeatureTracker(feature_name={self.feature_name}, times_used={self.times_used})>"

