"""Tests for build orchestrator service."""
import pytest
import json
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.user import User
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.models.build_history import BuildHistory, FeatureTracker
from app.services.build_orchestrator import BuildOrchestrator
from app.services.build_history_service import BuildHistoryService, FeatureTrackerService
from app.auth.security import hash_password

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    AsyncSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with AsyncSessionLocal() as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def test_user(test_db):
    """Create test user."""
    user = User(
        email="test@example.com",
        hashed_password=hash_password("password123"),
        status="active",
        user_role="user",
    )
    test_db.add(user)
    await test_db.flush()
    return user


@pytest.fixture
async def test_strategy(test_db, test_user):
    """Create test strategy."""
    strategy = Strategy(
        name="Test Strategy",
        description="A test strategy",
        status="draft",
        symbols=["NVDA", "AMD"],
        user_id=test_user.uuid,
    )
    test_db.add(strategy)
    await test_db.flush()
    return strategy


@pytest.fixture
async def test_build(test_db, test_user, test_strategy):
    """Create test build."""
    build = StrategyBuild(
        strategy_id=test_strategy.uuid,
        user_id=test_user.uuid,
        status="queued",
        phase="initializing",
        started_at=datetime.utcnow(),
    )
    test_db.add(build)
    await test_db.flush()
    return build


@pytest.mark.asyncio
async def test_build_history_add_build(test_db, test_user):
    """Test adding a build to history."""
    build = await BuildHistoryService.add_build(
        test_db,
        user_id=test_user.uuid,
        strategy_name="Test Strategy",
        asset_class="semiconductors",
        timeframe="1d",
        target=10.0,
        symbols=["NVDA", "AMD"],
        success=True,
        backtest_results={"total_return": 15.5, "win_rate": 0.65},
    )
    
    assert build.strategy_name == "Test Strategy"
    assert build.asset_class == "semiconductors"
    assert build.success is True
    assert build.backtest_results["total_return"] == 15.5


@pytest.mark.asyncio
async def test_build_history_get_relevant_builds(test_db, test_user):
    """Test retrieving relevant builds."""
    # Add multiple builds
    await BuildHistoryService.add_build(
        test_db,
        user_id=test_user.uuid,
        strategy_name="Strategy 1",
        asset_class="semiconductors",
        timeframe="1d",
        target=10.0,
        symbols=["NVDA", "AMD"],
        success=True,
    )
    
    await BuildHistoryService.add_build(
        test_db,
        user_id=test_user.uuid,
        strategy_name="Strategy 2",
        asset_class="tech",
        timeframe="1d",
        target=10.0,
        symbols=["AAPL", "MSFT"],
        success=False,
    )
    
    # Get relevant builds for semiconductors
    relevant = await BuildHistoryService.get_relevant_builds(
        test_db,
        user_id=test_user.uuid,
        symbols=["NVDA"],
        asset_class="semiconductors",
        max_results=5,
    )
    
    assert len(relevant) > 0
    assert relevant[0].strategy_name == "Strategy 1"


@pytest.mark.asyncio
async def test_feature_tracker_update_stats(test_db):
    """Test updating feature tracker stats."""
    tracker = await FeatureTrackerService.update_feature_stats(
        test_db,
        feature_name="RSI_14",
        times_used=5,
        performance_data={"returns": [1.5, 2.0, 1.8]},
    )
    
    assert tracker.feature_name == "RSI_14"
    assert tracker.times_used == 5
    assert tracker.performance_data["returns"] == [1.5, 2.0, 1.8]


@pytest.mark.asyncio
async def test_feature_tracker_get_stats(test_db):
    """Test retrieving feature tracker stats."""
    # Create tracker
    await FeatureTrackerService.update_feature_stats(
        test_db,
        feature_name="MACD_HIST",
        times_used=3,
    )
    
    # Retrieve tracker
    tracker = await FeatureTrackerService.get_feature_stats(test_db, "MACD_HIST")
    
    assert tracker is not None
    assert tracker.feature_name == "MACD_HIST"
    assert tracker.times_used == 3


@pytest.mark.asyncio
async def test_build_orchestrator_infer_asset_class(test_db, test_user, test_build):
    """Test asset class inference."""
    orchestrator = BuildOrchestrator(test_db, test_user, test_build)
    
    # Test semiconductor detection
    asset_class = orchestrator._infer_asset_class(["NVDA", "AMD"])
    assert asset_class == "semiconductors"
    
    # Test tech detection
    asset_class = orchestrator._infer_asset_class(["AAPL", "MSFT"])
    assert asset_class == "tech"
    
    # Test mixed
    asset_class = orchestrator._infer_asset_class(["XYZ", "ABC"])
    assert asset_class == "mixed"


@pytest.mark.asyncio
async def test_build_orchestrator_parse_design_response(test_db, test_user, test_build):
    """Test parsing Claude response."""
    orchestrator = BuildOrchestrator(test_db, test_user, test_build)
    
    # Mock response with JSON
    class MockContent:
        def __init__(self, text):
            self.text = text
    
    class MockResponse:
        def __init__(self):
            self.content = [
                MockContent('Some text before\n{"strategy_description": "Test", "priority_features": ["RSI"]}\nSome text after')
            ]
    
    design = orchestrator._parse_design_response(MockResponse())
    
    assert design["strategy_description"] == "Test"
    assert "RSI" in design["priority_features"]


@pytest.mark.asyncio
async def test_build_orchestrator_logging(test_db, test_user, test_build):
    """Test build orchestrator logging."""
    orchestrator = BuildOrchestrator(test_db, test_user, test_build)
    
    orchestrator._log("Test message")
    
    assert len(orchestrator.logs) == 1
    assert "Test message" in orchestrator.logs[0]
    assert "[" in orchestrator.logs[0]  # Has timestamp

