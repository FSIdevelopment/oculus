"""BuildIteration model for tracking per-iteration ML training results."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey, Index, JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class BuildIteration(Base):
    """Tracks each iteration of a strategy build with full ML training results."""

    __tablename__ = "build_iterations"

    # Primary key
    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))

    # Foreign keys
    build_id: Mapped[str] = mapped_column(String(36), ForeignKey("strategy_builds.uuid"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.uuid"), nullable=False)

    # Iteration info
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # "pending", "designing", "training", "extracting_rules", "complete", "failed"
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # LLM Context
    llm_design: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    llm_thinking: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Training Config Used
    training_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Label Configuration Results
    optimal_label_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Feature Data
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Training Results Per Model Type
    hyperparameter_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    nn_training_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    lstm_training_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_evaluations: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Best Model
    best_model: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Extracted Rules
    entry_rules: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    exit_rules: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Backtest Results
    backtest_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    build: Mapped["StrategyBuild"] = relationship("StrategyBuild", back_populates="iterations", foreign_keys=[build_id])
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])

    # Table args
    __table_args__ = (
        UniqueConstraint("build_id", "iteration_number", name="uq_build_iteration"),
        Index("idx_iteration_build_id", "build_id"),
        Index("idx_iteration_user_id", "user_id"),
        Index("idx_iteration_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<BuildIteration(uuid={self.uuid}, build_id={self.build_id}, iteration={self.iteration_number}, status={self.status})>"

