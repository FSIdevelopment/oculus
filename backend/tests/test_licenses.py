"""Tests for license endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from main import app
from app.database import Base, get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.models.license import License
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
        user_role="user",
        stripe_customer_id="cus_test123"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user


@pytest.fixture
async def test_strategy(test_db, test_user):
    """Create a test strategy."""
    strategy = Strategy(
        uuid="strategy-1",
        name="Test Strategy",
        user_id=test_user.uuid,
        status="published"
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


def test_purchase_license(client, user_token, test_strategy):
    """Test purchasing a license."""
    response = client.post(
        f"/api/strategies/{test_strategy.uuid}/license",
        json={"license_type": "monthly", "strategy_id": test_strategy.uuid},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "active"
    assert data["license_type"] == "monthly"
    assert data["strategy_id"] == test_strategy.uuid


def test_list_user_licenses(client, user_token, test_db, test_user, test_strategy):
    """Test listing user's licenses."""
    # Create a license
    license_obj = License(
        uuid="license-1",
        status="active",
        license_type="monthly",
        strategy_id=test_strategy.uuid,
        user_id=test_user.uuid,
        expires_at=datetime.utcnow() + timedelta(days=30)
    )
    test_db.add(license_obj)
    test_db.commit()
    
    response = client.get(
        "/api/users/me/licenses",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["uuid"] == "license-1"


def test_renew_license(client, user_token, test_db, test_user, test_strategy):
    """Test renewing a license."""
    # Create a license
    license_obj = License(
        uuid="license-1",
        status="active",
        license_type="monthly",
        strategy_id=test_strategy.uuid,
        user_id=test_user.uuid,
        expires_at=datetime.utcnow() + timedelta(days=5)
    )
    test_db.add(license_obj)
    test_db.commit()
    
    response = client.put(
        "/api/licenses/license-1/renew",
        json={"license_type": "annual"},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["license_type"] == "annual"


def test_update_webhook(client, user_token, test_db, test_user, test_strategy):
    """Test updating license webhook URL."""
    # Create a license
    license_obj = License(
        uuid="license-1",
        status="active",
        license_type="monthly",
        strategy_id=test_strategy.uuid,
        user_id=test_user.uuid,
        expires_at=datetime.utcnow() + timedelta(days=30)
    )
    test_db.add(license_obj)
    test_db.commit()
    
    response = client.put(
        "/api/licenses/license-1/webhook",
        json={"webhook_url": "https://example.com/webhook"},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["webhook_url"] == "https://example.com/webhook"


def test_validate_license(client, test_db, test_user, test_strategy):
    """Test license validation endpoint."""
    # Create an active license
    license_obj = License(
        uuid="license-1",
        status="active",
        license_type="monthly",
        strategy_id=test_strategy.uuid,
        user_id=test_user.uuid,
        expires_at=datetime.utcnow() + timedelta(days=30)
    )
    test_db.add(license_obj)
    test_db.commit()
    
    response = client.get("/api/licenses/license-1/validate")
    
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["license_id"] == "license-1"
    assert data["polygon_api_key"] is not None


def test_validate_expired_license(client, test_db, test_user, test_strategy):
    """Test validation of expired license."""
    # Create an expired license
    license_obj = License(
        uuid="license-1",
        status="active",
        license_type="monthly",
        strategy_id=test_strategy.uuid,
        user_id=test_user.uuid,
        expires_at=datetime.utcnow() - timedelta(days=1)
    )
    test_db.add(license_obj)
    test_db.commit()
    
    response = client.get("/api/licenses/license-1/validate")
    
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert data["polygon_api_key"] is None

