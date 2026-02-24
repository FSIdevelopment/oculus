"""Pydantic schemas for strategy endpoints."""
from typing import Optional, List, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, computed_field
from app.services.pricing import calculate_license_price


class StrategyCreate(BaseModel):
    """Schema for creating a new strategy."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    config: dict = Field(default_factory=dict)


class StrategyUpdate(BaseModel):
    """Schema for updating strategy metadata."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    config: Optional[dict] = None


class BuildResponse(BaseModel):
    """Schema for strategy build details."""
    uuid: str
    strategy_id: str
    status: str
    phase: Optional[str] = None
    tokens_consumed: float
    iteration_count: int
    max_iterations: Optional[int] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    strategy_type: Optional[str] = None
    strategy_name: Optional[str] = None
    queue_position: Optional[int] = None

    class Config:
        from_attributes = True


class MarketplaceListRequest(BaseModel):
    """Schema for listing/unlisting a strategy on the marketplace."""
    listed: bool
    price: Optional[float] = None


class StrategyResponse(BaseModel):
    """Schema for strategy detail response (user-facing, excludes Docker info)."""
    uuid: str
    name: str
    description: Optional[str] = None
    status: str
    strategy_type: Optional[str] = None
    symbols: Optional[Union[List, dict]] = None
    target_return: Optional[float] = None
    backtest_results: Optional[dict] = None
    marketplace_listed: bool = False
    marketplace_price: Optional[float] = None
    version: int
    subscriber_count: int
    rating: Optional[float] = None
    user_id: str
    created_at: datetime
    updated_at: datetime
    # Admin-specific fields (populated only in admin endpoints)
    owner_name: Optional[str] = None
    build_count: Optional[int] = None
    annual_return: Optional[float] = None
    best_return_pct: Optional[float] = None

    # calculate_license_price is a pure arithmetic function (no I/O), so calling
    # it separately for each field is safe. Both calls produce identical results.
    # Guard uses falsy check (not just `is None`) so that an empty dict {} also
    # returns None rather than the floor price.
    @computed_field  # type: ignore[misc]
    @property
    def monthly_price(self) -> int | None:
        if not self.backtest_results:
            return None
        monthly, _, _ = calculate_license_price(self.backtest_results)
        return monthly

    @computed_field  # type: ignore[misc]
    @property
    def annual_price(self) -> int | None:
        if not self.backtest_results:
            return None
        _, annual, _ = calculate_license_price(self.backtest_results)
        return annual

    class Config:
        from_attributes = True


class StrategyListResponse(BaseModel):
    """Schema for paginated strategy list response."""
    items: List[StrategyResponse]
    total: int
    skip: int
    limit: int


class StrategyInternalResponse(BaseModel):
    """Schema for internal strategy response (includes Docker info for SignalSynk)."""
    uuid: str
    name: str
    description: Optional[str] = None
    status: str
    strategy_type: Optional[str] = None
    symbols: Optional[Union[List, dict]] = None
    target_return: Optional[float] = None
    backtest_results: Optional[dict] = None
    config: Optional[dict] = None  # Strategy config.json for SignalSynk
    docker_registry: Optional[str] = None
    docker_image_url: Optional[str] = None
    version: int
    subscriber_count: int
    rating: Optional[float] = None
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BuildListItem(BuildResponse):
    """Schema for build list item with strategy name."""
    strategy_name: str


class BuildListResponse(BaseModel):
    """Schema for paginated build list response."""
    items: List[BuildListItem]
    total: int
    skip: int
    limit: int


class BuildIterationResponse(BaseModel):
    """Schema for build iteration details."""
    uuid: str
    build_id: str
    user_id: str
    iteration_number: int
    status: str
    duration_seconds: Optional[float] = None
    llm_design: Optional[dict] = None
    llm_thinking: Optional[str] = None
    training_config: Optional[dict] = None
    optimal_label_config: Optional[dict] = None
    features: Optional[Any] = None
    hyperparameter_results: Optional[Any] = None
    nn_training_results: Optional[Any] = None
    lstm_training_results: Optional[Any] = None
    model_evaluations: Optional[Any] = None
    best_model: Optional[dict] = None
    entry_rules: Optional[list] = None
    exit_rules: Optional[list] = None
    backtest_results: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

