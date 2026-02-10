"""Tests for user management endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from main import app
from app.database import Base, get_db
from app.models.user import User
from app.auth.security import hash_password, create_access_token


# Test database setup
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
        password_hash=hash_password("TestPassword123"),
        phone_number="+1234567890",
        balance=100.0,
        status="active",
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user


@pytest.fixture
async def test_admin(test_db):
    """Create a test admin user."""
    admin = User(
        uuid="test-admin-1",
        name="Admin User",
        email="admin@example.com",
        password_hash=hash_password("AdminPassword123"),
        balance=1000.0,
        status="active",
        user_role="admin"
    )
    test_db.add(admin)
    await test_db.commit()
    await test_db.refresh(admin)
    return admin


def get_auth_header(user_uuid: str) -> dict:
    """Generate auth header with JWT token."""
    token = create_access_token(data={
        "sub": user_uuid,
        "email": "test@example.com",
        "role": "user"
    })
    return {"Authorization": f"Bearer {token}"}


def test_get_user_profile(client, test_user):
    """Test getting current user profile."""
    headers = get_auth_header(test_user.uuid)
    response = client.get("/api/users/me", headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["uuid"] == test_user.uuid
    assert data["email"] == test_user.email
    assert data["name"] == test_user.name


def test_update_user_profile(client, test_user):
    """Test updating user profile."""
    headers = get_auth_header(test_user.uuid)
    update_data = {
        "name": "Updated Name",
        "phone_number": "+9876543210"
    }
    response = client.put("/api/users/me", json=update_data, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Name"
    assert data["phone_number"] == "+9876543210"


def test_get_user_balance(client, test_user):
    """Test getting user balance."""
    headers = get_auth_header(test_user.uuid)
    response = client.get("/api/users/me/balance", headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["balance"] == 100.0
    assert data["currency"] == "USD"


def test_list_users_admin(client, test_admin, test_user):
    """Test listing users as admin."""
    headers = get_auth_header(test_admin.uuid)
    response = client.get("/api/admin/users", headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["limit"] == 20


def test_list_users_pagination(client, test_admin):
    """Test user list pagination."""
    headers = get_auth_header(test_admin.uuid)
    response = client.get("/api/admin/users?skip=0&limit=10", headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["skip"] == 0
    assert data["limit"] == 10


def test_get_user_detail_admin(client, test_admin, test_user):
    """Test getting user detail as admin."""
    headers = get_auth_header(test_admin.uuid)
    response = client.get(f"/api/admin/users/{test_user.uuid}", headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["uuid"] == test_user.uuid
    assert data["email"] == test_user.email


def test_soft_delete_user(client, test_admin, test_user):
    """Test soft-deleting a user."""
    headers = get_auth_header(test_admin.uuid)
    response = client.delete(f"/api/admin/users/{test_user.uuid}", headers=headers)
    
    assert response.status_code == 204


def test_impersonate_user(client, test_admin, test_user):
    """Test admin impersonation."""
    headers = get_auth_header(test_admin.uuid)
    response = client.post(
        f"/api/admin/users/{test_user.uuid}/impersonate",
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["impersonated_user_id"] == test_user.uuid
    assert data["impersonator_id"] == test_admin.uuid

