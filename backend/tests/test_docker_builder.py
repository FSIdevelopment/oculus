"""Tests for Docker builder service."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from app.services.docker_builder import DockerBuilder
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.models.user import User


@pytest.fixture
async def test_user(test_db):
    """Create test user."""
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash="hashed_password",
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
        version=1,
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
        status="training",
        phase="training",
        started_at=datetime.utcnow(),
    )
    test_db.add(build)
    await test_db.flush()
    return build


@pytest.mark.asyncio
async def test_sanitize_docker_tag():
    """Test Docker tag sanitization."""
    builder = DockerBuilder(None)
    
    # Test various inputs
    assert builder._sanitize_docker_tag("Test Strategy") == "test-strategy"
    assert builder._sanitize_docker_tag("Test_Strategy_V2") == "test-strategy-v2"
    assert builder._sanitize_docker_tag("Strategy@#$%") == "strategy"
    assert builder._sanitize_docker_tag("---test---") == "test"
    assert builder._sanitize_docker_tag("a" * 200) == "a" * 128


@pytest.mark.asyncio
async def test_build_and_push_success(test_db, test_strategy, test_build):
    """Test successful Docker build and push."""
    builder = DockerBuilder(test_db)
    
    # Mock the _run_command method
    with patch.object(builder, '_run_command', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = 0  # Success
        
        result = await builder.build_and_push(
            test_strategy,
            test_build,
            "/tmp/test_strategy"
        )
        
        assert result is True
        assert test_strategy.docker_registry == "forefrontsolutions"
        assert "forefrontsolutions/test-strategy:1" in test_strategy.docker_image_url
        assert test_build.status == "complete"
        assert test_build.completed_at is not None


@pytest.mark.asyncio
async def test_build_and_push_build_failure(test_db, test_strategy, test_build):
    """Test Docker build failure."""
    builder = DockerBuilder(test_db)
    
    # Mock the _run_command method to fail on build
    with patch.object(builder, '_run_command', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = 1  # Failure
        
        result = await builder.build_and_push(
            test_strategy,
            test_build,
            "/tmp/test_strategy"
        )
        
        assert result is False
        assert test_build.status == "failed"
        assert test_build.completed_at is not None


@pytest.mark.asyncio
async def test_build_and_push_no_pat(test_db, test_strategy, test_build):
    """Test Docker build without PAT (should still work)."""
    builder = DockerBuilder(test_db)
    builder.docker_hub_pat = ""  # No PAT
    
    with patch.object(builder, '_run_command', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = 0
        
        result = await builder.build_and_push(
            test_strategy,
            test_build,
            "/tmp/test_strategy"
        )
        
        assert result is True
        # Should still build and push (login skipped)
        assert test_build.status == "complete"

