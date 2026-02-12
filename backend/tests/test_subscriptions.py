"""Tests for subscription endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from main import app
from app.database import Base, get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.models.subscription import Subscription
from app.auth.security import hash_password, create_access_token

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
    
    await engine.dispose()


@pytest.fixture
def client(test_db):
    """Create test client."""
    async def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.fixture
async def test_user(test_db):
    """Create a test user."""
    user = User(
        uuid="test-user-1",
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        status="active",
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user


@pytest.fixture
async def creator_user(test_db):
    """Create a test creator user."""
    user = User(
        uuid="creator-user-1",
        name="Creator User",
        email="creator@example.com",
        password_hash=hash_password("CreatorPass123"),
        status="active",
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user


@pytest.fixture
async def test_strategy(test_db, creator_user):
    """Create a test strategy."""
    strategy = Strategy(
        uuid="strategy-1",
        name="Test Strategy",
        user_id=creator_user.uuid,
        status="published",
        subscriber_count=0
    )
    test_db.add(strategy)
    await test_db.commit()
    await test_db.refresh(strategy)
    return strategy


@pytest.fixture
def user_token(test_user):
    """Create a test token for the user."""
    return create_access_token(
        data={"sub": test_user.uuid, "email": test_user.email, "role": test_user.user_role}
    )


@pytest.fixture
def creator_token(creator_user):
    """Create a test token for the creator."""
    return create_access_token(
        data={"sub": creator_user.uuid, "email": creator_user.email, "role": creator_user.user_role}
    )


def test_subscribe_to_strategy(client, user_token, test_strategy):
    """Test subscribing to a marketplace strategy."""
    response = client.post(
        f"/api/marketplace/{test_strategy.uuid}/subscribe",
        json={},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "active"
    assert data["strategy_id"] == test_strategy.uuid


def test_subscribe_duplicate(client, user_token, test_db, test_user, test_strategy):
    """Test that duplicate subscriptions are prevented."""
    # Create first subscription
    sub = Subscription(
        uuid="sub-1",
        status="active",
        user_id=test_user.uuid,
        strategy_id=test_strategy.uuid
    )
    test_db.add(sub)
    test_db.commit()
    
    # Try to subscribe again
    response = client.post(
        f"/api/marketplace/{test_strategy.uuid}/subscribe",
        json={},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    assert response.status_code == 400
    assert "Already subscribed" in response.json()["detail"]


def test_list_user_subscriptions(client, user_token, test_db, test_user, test_strategy):
    """Test listing user's subscriptions."""
    # Create a subscription
    sub = Subscription(
        uuid="sub-1",
        status="active",
        user_id=test_user.uuid,
        strategy_id=test_strategy.uuid
    )
    test_db.add(sub)
    test_db.commit()
    
    response = client.get(
        "/api/users/me/subscriptions",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["uuid"] == "sub-1"


def test_get_creator_earnings(client, creator_token, test_db, creator_user, test_strategy):
    """Test getting creator earnings dashboard."""
    # Create some subscriptions
    for i in range(3):
        sub = Subscription(
            uuid=f"sub-{i}",
            status="active",
            user_id=f"user-{i}",
            strategy_id=test_strategy.uuid
        )
        test_db.add(sub)
    test_db.commit()
    
    response = client.get(
        "/api/users/me/earnings",
        headers={"Authorization": f"Bearer {creator_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["subscriber_count"] == 3
    assert data["active_subscriptions"] == 3
    assert len(data["strategies"]) == 1
    assert data["strategies"][0]["strategy_id"] == test_strategy.uuid
    assert data["strategies"][0]["subscriber_count"] == 3


def test_cancel_subscription(client, user_token, test_db, test_user, test_strategy):
    """Test canceling a subscription."""
    # Create a subscription
    sub = Subscription(
        uuid="sub-1",
        status="active",
        user_id=test_user.uuid,
        strategy_id=test_strategy.uuid
    )
    test_db.add(sub)
    test_db.commit()
    
    response = client.post(
        "/api/subscriptions/sub-1/cancel",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"


def test_cancel_subscription_unauthorized(client, user_token, test_db):
    """Test that users can't cancel others' subscriptions."""
    # Create a subscription for a different user
    sub = Subscription(
        uuid="sub-1",
        status="active",
        user_id="other-user",
        strategy_id="strategy-1"
    )
    test_db.add(sub)
    test_db.commit()
    
    response = client.post(
        "/api/subscriptions/sub-1/cancel",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    assert response.status_code == 403
    assert "Not authorized" in response.json()["detail"]

