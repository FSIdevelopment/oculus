"""Tests for build orchestrator service."""
import pytest
import json
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.database import Base, get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.models.build_history import BuildHistory, FeatureTracker
from app.services.build_orchestrator import BuildOrchestrator
from app.services.build_history_service import BuildHistoryService, FeatureTrackerService
from app.auth.security import hash_password
from main import app

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
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("password123"),
        balance=100.0,
        status="active",
        user_role="user",
    )
    test_db.add(user)
    await test_db.flush()
    return user


@pytest.fixture
def client(test_db):
    """Create test client with DB override."""
    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


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
    """Test parsing Claude response with text and thinking blocks."""
    orchestrator = BuildOrchestrator(test_db, test_user, test_build)

    # Mock response blocks with type attribute
    class MockTextBlock:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class MockThinkingBlock:
        def __init__(self, thinking):
            self.type = "thinking"
            self.thinking = thinking

    class MockResponse:
        def __init__(self):
            self.content = [
                MockThinkingBlock("I should design a momentum strategy using RSI."),
                MockTextBlock('Some text before\n{"strategy_description": "Test", "priority_features": ["RSI"], "entry_rules": [{"feature": "RSI_14", "operator": "<=", "threshold": 30}], "exit_rules": [{"feature": "RSI_14", "operator": ">=", "threshold": 70}]}\nSome text after'),
            ]

    result = orchestrator._parse_design_response(MockResponse())

    # Result now returns {"design": ..., "thinking": ...}
    assert result["design"]["strategy_description"] == "Test"
    assert "RSI" in result["design"]["priority_features"]
    assert "momentum strategy" in result["thinking"]


@pytest.mark.asyncio
async def test_build_orchestrator_logging(test_db, test_user, test_build):
    """Test build orchestrator logging."""
    orchestrator = BuildOrchestrator(test_db, test_user, test_build)

    orchestrator._log("Test message")

    assert len(orchestrator.logs) == 1
    assert "Test message" in orchestrator.logs[0]
    assert "[" in orchestrator.logs[0]  # Has timestamp



def test_get_build_pricing(client):
    """Test GET /api/builds/pricing returns token cost per iteration."""
    response = client.get("/api/builds/pricing")
    assert response.status_code == 200
    data = response.json()
    assert "tokens_per_iteration" in data
    assert isinstance(data["tokens_per_iteration"], (int, float))
    assert data["tokens_per_iteration"] > 0


@pytest.mark.asyncio
async def test_build_stops_on_insufficient_tokens(test_db, test_user, test_strategy):
    """Test that a build can be marked as stopped when tokens are insufficient."""
    from app.services.balance import deduct_tokens
    from fastapi import HTTPException

    # Give user a small balance (less than TOKENS_PER_ITERATION)
    test_user.balance = 5.0
    await test_db.commit()
    await test_db.refresh(test_user)

    # Create a build in "running" state
    build = StrategyBuild(
        strategy_id=test_strategy.uuid,
        user_id=test_user.uuid,
        status="running",
        phase="designing",
        started_at=datetime.utcnow(),
        tokens_consumed=0.0,
        iteration_count=0,
    )
    test_db.add(build)
    await test_db.commit()
    await test_db.refresh(build)

    # Attempt to deduct TOKENS_PER_ITERATION (10.0) — should fail with 402
    from app.config import settings
    with pytest.raises(HTTPException) as exc_info:
        await deduct_tokens(
            test_db,
            test_user.uuid,
            settings.TOKENS_PER_ITERATION,
            f"Build iteration 1 for 'Test Strategy'"
        )
    assert exc_info.value.status_code == 402

    # Simulate what _run_build_loop does on 402: mark build as stopped
    build.status = "stopped"
    build.phase = "insufficient_tokens"
    build.completed_at = datetime.utcnow()
    await test_db.commit()
    await test_db.refresh(build)

    # Verify the build was stopped gracefully
    assert build.status == "stopped"
    assert build.phase == "insufficient_tokens"
    assert build.completed_at is not None

    # Verify user balance was NOT deducted (402 prevents deduction)
    await test_db.refresh(test_user)
    assert test_user.balance == 5.0