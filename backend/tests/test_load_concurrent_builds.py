"""Load tests for concurrent build creation."""
import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from main import app
from app.database import get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.auth.security import hash_password, create_access_token


@pytest.fixture
def client(test_db):
    """Create test client with DB override."""
    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
async def test_user(test_db):
    """Create test user with sufficient balance."""
    user = User(
        name="Load Test User",
        email="loadtest@example.com",
        password_hash=hash_password("LoadTest123"),
        status="active",
        user_role="user",
        balance=100000.0  # Sufficient balance for many builds
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user


@pytest.fixture
def user_token(test_user):
    """Create JWT token for test user."""
    return create_access_token(
        data={"sub": test_user.uuid, "email": test_user.email, "role": test_user.user_role}
    )


@pytest.fixture
async def test_strategy(test_db, test_user):
    """Create test strategy."""
    strategy = Strategy(
        name="Load Test Strategy",
        description="A strategy for load testing",
        status="draft",
        symbols=["AAPL", "MSFT", "GOOGL"],
        user_id=test_user.uuid,
        version=1,
    )
    test_db.add(strategy)
    await test_db.commit()
    await test_db.refresh(strategy)
    return strategy


async def trigger_build_request(client, strategy_id: str, user_token: str):
    """Helper function to trigger a single build request."""
    start_time = time.time()
    response = client.post(
        "/api/builds/trigger",
        params={
            "strategy_id": strategy_id,
            "description": "Load test build",
            "target_return": 10.0,
            "timeframe": "1d",
            "max_iterations": 5,
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    elapsed_time = time.time() - start_time
    return response, elapsed_time


@pytest.mark.asyncio
async def test_concurrent_build_creation(client, test_db, test_strategy, user_token):
    """Test 5 concurrent build requests."""
    # Mock the background task to prevent actual ML training
    with patch("app.routers.builds.asyncio.create_task") as mock_create_task:
        mock_create_task.return_value = AsyncMock()
        
        # Trigger 5 concurrent builds
        tasks = [
            trigger_build_request(client, test_strategy.uuid, user_token)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)
        
        # Verify all responses
        build_ids = set()
        for response, elapsed_time in results:
            assert response.status_code in [200, 201, 202], f"Unexpected status: {response.status_code}"
            data = response.json()
            assert "uuid" in data
            assert data["status"] == "queued"
            assert data["uuid"] not in build_ids, "Duplicate build ID detected"
            build_ids.add(data["uuid"])
            assert elapsed_time < 2.0, f"Response time {elapsed_time}s exceeded 2s threshold"
        
        # Verify all builds have unique IDs
        assert len(build_ids) == 5, f"Expected 5 unique build IDs, got {len(build_ids)}"
        
        # Verify background task was called 5 times
        assert mock_create_task.call_count == 5


@pytest.mark.asyncio
async def test_10_concurrent_builds(client, test_db, test_strategy, user_token):
    """Test 10 concurrent build requests."""
    with patch("app.routers.builds.asyncio.create_task") as mock_create_task:
        mock_create_task.return_value = AsyncMock()
        
        # Trigger 10 concurrent builds
        tasks = [
            trigger_build_request(client, test_strategy.uuid, user_token)
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)
        
        # Verify all responses
        build_ids = set()
        response_times = []
        for response, elapsed_time in results:
            assert response.status_code in [200, 201, 202], f"Unexpected status: {response.status_code}"
            data = response.json()
            assert "uuid" in data
            assert data["status"] == "queued"
            assert data["uuid"] not in build_ids, "Duplicate build ID detected"
            build_ids.add(data["uuid"])
            response_times.append(elapsed_time)
            assert elapsed_time < 2.0, f"Response time {elapsed_time}s exceeded 2s threshold"
        
        # Verify all builds have unique IDs
        assert len(build_ids) == 10, f"Expected 10 unique build IDs, got {len(build_ids)}"
        
        # Verify background task was called 10 times
        assert mock_create_task.call_count == 10
        
        # Report timing stats
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        print(f"\n10 concurrent builds - Avg: {avg_time:.3f}s, Max: {max_time:.3f}s")


@pytest.mark.asyncio
async def test_25_concurrent_builds(client, test_db, test_strategy, user_token):
    """Test 25 concurrent build requests."""
    with patch("app.routers.builds.asyncio.create_task") as mock_create_task:
        mock_create_task.return_value = AsyncMock()

        # Trigger 25 concurrent builds
        tasks = [
            trigger_build_request(client, test_strategy.uuid, user_token)
            for _ in range(25)
        ]
        results = await asyncio.gather(*tasks)

        # Verify all responses
        build_ids = set()
        response_times = []
        for response, elapsed_time in results:
            assert response.status_code in [200, 201, 202], f"Unexpected status: {response.status_code}"
            data = response.json()
            assert "uuid" in data
            assert data["status"] == "queued"
            assert data["uuid"] not in build_ids, "Duplicate build ID detected"
            build_ids.add(data["uuid"])
            response_times.append(elapsed_time)
            assert elapsed_time < 2.0, f"Response time {elapsed_time}s exceeded 2s threshold"

        # Verify all builds have unique IDs
        assert len(build_ids) == 25, f"Expected 25 unique build IDs, got {len(build_ids)}"

        # Verify background task was called 25 times
        assert mock_create_task.call_count == 25

        # Report timing stats
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        print(f"\n25 concurrent builds - Avg: {avg_time:.3f}s, Max: {max_time:.3f}s, Min: {min_time:.3f}s")

