"""Tests for authentication endpoints."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app
from app.database import get_db
from app.models.user import User
from app.auth.security import hash_password, create_access_token, create_refresh_token


@pytest.fixture
def client(test_db):
    """Create test client."""
    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.mark.asyncio
async def test_register_success(client, test_db):
    """Test successful user registration."""
    with patch("stripe.Customer.create") as mock_customer:
        mock_customer.return_value = MagicMock(id="cus_test123")

        response = client.post(
            "/api/auth/register",
            json={
                "name": "Test User",
                "email": "newuser@example.com",
                "password": "Test1234a",
                "phone_number": "+1234567890"
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    
    # Verify user was created
    result = await test_db.execute(
        __import__('sqlalchemy').select(User).where(User.email == "newuser@example.com")
    )
    user = result.scalar_one_or_none()
    assert user is not None
    assert user.name == "Test User"
    assert user.email == "newuser@example.com"


@pytest.mark.asyncio
async def test_register_duplicate_email(client, test_db):
    """Test registration fails with duplicate email."""
    # Create existing user
    user = User(
        name="Existing User",
        email="existing@example.com",
        password_hash=hash_password("Test1234a"),
        status="active",
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()

    # Try to register with same email
    response = client.post(
        "/api/auth/register",
        json={
            "name": "New User",
            "email": "existing@example.com",
            "password": "Test1234a"
        }
    )
    
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_weak_password(client, test_db):
    """Test registration fails with weak password."""
    response = client.post(
        "/api/auth/register",
        json={
            "name": "Test User",
            "email": "test@example.com",
            "password": "weakpass"  # No uppercase, no number
        }
    )
    
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_register_invalid_email(client, test_db):
    """Test registration fails with invalid email."""
    response = client.post(
        "/api/auth/register",
        json={
            "name": "Test User",
            "email": "not-an-email",
            "password": "Test1234a"
        }
    )
    
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_success(client, test_db):
    """Test successful login."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("Test1234a"),
        status="active",
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()

    # Login
    response = client.post(
        "/api/auth/login",
        json={
            "email": "test@example.com",
            "password": "Test1234a"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client, test_db):
    """Test login fails with wrong password."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("Test1234a"),
        status="active",
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()

    # Try to login with wrong password
    response = client.post(
        "/api/auth/login",
        json={
            "email": "test@example.com",
            "password": "Wrong123!"
        }
    )
    
    assert response.status_code == 401
    assert "invalid" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_login_nonexistent_user(client, test_db):
    """Test login fails with non-existent user."""
    response = client.post(
        "/api/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "Test1234a"
        }
    )
    
    assert response.status_code == 401
    assert "invalid" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_refresh_token_success(client, test_db):
    """Test successful token refresh."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("Test1234a"),
        status="active",
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Create refresh token
    refresh_token = create_refresh_token(
        data={"sub": user.uuid, "email": user.email, "role": user.user_role}
    )
    
    # Refresh token
    response = client.post(
        "/api/auth/refresh",
        json={"refresh_token": refresh_token}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_refresh_token_invalid(client, test_db):
    """Test refresh fails with invalid token."""
    response = client.post(
        "/api/auth/refresh",
        json={"refresh_token": "invalid_token"}
    )
    
    assert response.status_code == 401
    assert "invalid" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_refresh_token_missing(client, test_db):
    """Test refresh fails with missing token."""
    response = client.post(
        "/api/auth/refresh",
        json={"refresh_token": ""}
    )
    
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_refresh_token_user_deleted(client, test_db):
    """Test refresh fails if user no longer exists."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("Test1234a"),
        status="active",
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Create refresh token
    refresh_token = create_refresh_token(
        data={"sub": user.uuid, "email": user.email, "role": user.user_role}
    )
    
    # Delete user
    await test_db.delete(user)
    await test_db.commit()
    
    # Try to refresh
    response = client.post(
        "/api/auth/refresh",
        json={"refresh_token": refresh_token}
    )
    
    assert response.status_code == 401
    assert "not found" in response.json()["detail"].lower()

