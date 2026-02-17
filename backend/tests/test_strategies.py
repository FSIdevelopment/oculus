"""Tests for strategy CRUD endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from main import app
from app.database import Base, get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.auth.security import hash_password, create_access_token
from datetime import datetime

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
async def admin_user(test_db):
    """Create a test admin user."""
    user = User(
        uuid="admin-user-1",
        name="Admin User",
        email="admin@example.com",
        password_hash=hash_password("AdminPass123"),
        status="active",
        user_role="admin"
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
def admin_token(admin_user):
    """Create JWT token for admin user."""
    return create_access_token(
        data={"sub": admin_user.uuid, "email": admin_user.email, "role": admin_user.user_role}
    )


@pytest.mark.asyncio
async def test_create_strategy(client, user_token):
    """Test creating a new strategy."""
    response = client.post(
        "/api/strategies",
        json={
            "name": "Test Strategy",
            "description": "A test strategy",
            "config": {"symbols": ["AAPL", "GOOGL"]}
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Strategy"
    assert data["description"] == "A test strategy"


@pytest.mark.asyncio
async def test_create_strategy_duplicate_name(client, test_db, test_user, user_token):
    """Test creating strategy with duplicate name appends UUID suffix."""
    # Create first strategy
    response1 = client.post(
        "/api/strategies",
        json={"name": "MyStrategy", "description": "First", "config": {}},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response1.status_code == 201
    
    # Create second with same name
    response2 = client.post(
        "/api/strategies",
        json={"name": "MyStrategy", "description": "Second", "config": {}},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response2.status_code == 201
    data = response2.json()
    # Should have suffix
    assert "-" in data["name"]
    assert test_user.uuid[-6:] in data["name"]


@pytest.mark.asyncio
async def test_list_strategies(client, test_db, test_user, user_token):
    """Test listing user's strategies."""
    # Create a strategy
    client.post(
        "/api/strategies",
        json={"name": "Strategy1", "description": "Desc1", "config": {}},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    # List strategies
    response = client.get(
        "/api/strategies",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1


@pytest.mark.asyncio
async def test_get_strategy(client, test_db, test_user, user_token):
    """Test getting strategy detail."""
    # Create strategy
    create_resp = client.post(
        "/api/strategies",
        json={"name": "TestStrat", "description": "Test", "config": {}},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    strategy_id = create_resp.json()["uuid"]
    
    # Get strategy
    response = client.get(
        f"/api/strategies/{strategy_id}",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert response.json()["uuid"] == strategy_id


@pytest.mark.asyncio
async def test_update_strategy(client, test_db, test_user, user_token):
    """Test updating strategy."""
    # Create strategy
    create_resp = client.post(
        "/api/strategies",
        json={"name": "Original", "description": "Orig", "config": {}},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    strategy_id = create_resp.json()["uuid"]
    
    # Update
    response = client.put(
        f"/api/strategies/{strategy_id}",
        json={"name": "Updated", "description": "New desc"},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Updated"


@pytest.mark.asyncio
async def test_delete_strategy(client, test_db, test_user, user_token):
    """Test soft-deleting strategy."""
    # Create strategy
    create_resp = client.post(
        "/api/strategies",
        json={"name": "ToDelete", "description": "Delete me", "config": {}},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    strategy_id = create_resp.json()["uuid"]
    
    # Delete
    response = client.delete(
        f"/api/strategies/{strategy_id}",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 204
    
    # Verify it's gone from list
    list_resp = client.get(
        "/api/strategies",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert list_resp.json()["total"] == 0


@pytest.mark.asyncio
async def test_admin_list_all_strategies(client, test_db, test_user, admin_user, admin_token, user_token):
    """Test admin can list all strategies."""
    # User creates strategy
    client.post(
        "/api/strategies",
        json={"name": "UserStrat", "description": "User's", "config": {}},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    
    # Admin lists all
    response = client.get(
        "/api/admin/strategies",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    assert response.json()["total"] == 1


@pytest.mark.asyncio
async def test_admin_update_strategy(client, test_db, test_user, admin_token, user_token):
    """Test admin can update any strategy."""
    # User creates strategy
    create_resp = client.post(
        "/api/strategies",
        json={"name": "UserStrat", "description": "User's", "config": {}},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    strategy_id = create_resp.json()["uuid"]
    
    # Admin updates it
    response = client.put(
        f"/api/admin/strategies/{strategy_id}",
        json={"name": "AdminUpdated"},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "AdminUpdated"


@pytest.mark.asyncio
async def test_admin_delete_strategy(client, test_db, test_user, admin_token, user_token):
    """Test admin can delete any strategy."""
    # User creates strategy
    create_resp = client.post(
        "/api/strategies",
        json={"name": "UserStrat", "description": "User's", "config": {}},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    strategy_id = create_resp.json()["uuid"]
    
    # Admin deletes it
    response = client.delete(
        f"/api/admin/strategies/{strategy_id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 204


def test_get_docker_info_success(client, test_db, test_user, user_token):
    """Test getting Docker info for a strategy with built image."""
    # Create strategy with Docker image
    strategy = Strategy(
        uuid="docker-test-1",
        name="Docker Test Strategy",
        description="Test",
        status="complete",
        user_id=test_user.uuid,
        docker_registry="forfrontsolutions",
        docker_image_url="forfrontsolutions/docker-test-strategy:1",
    )
    test_db.add(strategy)
    test_db.commit()

    response = client.get(
        f"/api/strategies/docker-test-1/docker",
        headers={"Authorization": f"Bearer {user_token}"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["image_url"] == "forfrontsolutions/docker-test-strategy:1"
    assert "docker pull" in data["pull_command"]
    assert "docker run" in data["run_command"]
    assert "LICENSE_ID" in data["environment_variables"]
    assert "terms_of_use" in data


def test_get_docker_info_not_built(client, test_db, test_user, user_token):
    """Test getting Docker info for strategy without built image."""
    # Create strategy without Docker image
    strategy = Strategy(
        uuid="no-docker-1",
        name="No Docker Strategy",
        description="Test",
        status="draft",
        user_id=test_user.uuid,
    )
    test_db.add(strategy)
    test_db.commit()

    response = client.get(
        f"/api/strategies/no-docker-1/docker",
        headers={"Authorization": f"Bearer {user_token}"}
    )

    assert response.status_code == 400
    assert "not yet built" in response.json()["detail"]


def test_get_docker_info_not_found(client, user_token):
    """Test getting Docker info for non-existent strategy."""
    response = client.get(
        "/api/strategies/nonexistent/docker",
        headers={"Authorization": f"Bearer {user_token}"}
    )

    assert response.status_code == 404

