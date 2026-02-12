"""Tests for admin endpoints."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app
from app.database import get_db
from app.models.user import User
from app.models.product import Product
from app.auth.security import hash_password, create_access_token


@pytest.fixture
def client(test_db):
    """Create test client."""
    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.fixture
async def admin_user(test_db):
    """Create admin user."""
    user = User(
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
async def regular_user(test_db):
    """Create regular user."""
    user = User(
        name="Regular User",
        email="user@example.com",
        password_hash=hash_password("UserPass123"),
        status="active",
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user


@pytest.mark.asyncio
async def test_list_users_admin(client, admin_user, regular_user):
    """Test admin can list users."""
    token = create_access_token(
        data={"sub": admin_user.uuid, "email": admin_user.email, "role": "admin"}
    )
    response = client.get(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 2


@pytest.mark.asyncio
async def test_list_users_non_admin(client, regular_user):
    """Test non-admin cannot list users."""
    token = create_access_token(
        data={"sub": regular_user.uuid, "email": regular_user.email, "role": "user"}
    )
    response = client.get(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_user_admin(client, test_db, admin_user):
    """Test admin can create user."""
    with patch("stripe.Customer.create") as mock_customer:
        mock_customer.return_value = MagicMock(id="cus_test123")
        
        token = create_access_token(
            data={"sub": admin_user.uuid, "email": admin_user.email, "role": "admin"}
        )
        response = client.post(
            "/api/admin/users",
            json={
                "name": "New User",
                "email": "newuser@example.com",
                "password": "Test1234a",
                "user_role": "user",
                "status": "active"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "New User"
    assert data["email"] == "newuser@example.com"
    assert data["user_role"] == "user"


@pytest.mark.asyncio
async def test_create_user_non_admin(client, regular_user):
    """Test non-admin cannot create user."""
    token = create_access_token(
        data={"sub": regular_user.uuid, "email": regular_user.email, "role": "user"}
    )
    response = client.post(
        "/api/admin/users",
        json={
            "name": "New User",
            "email": "newuser@example.com",
            "password": "Test1234a"
        },
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_user_duplicate_email(client, test_db, admin_user, regular_user):
    """Test cannot create user with duplicate email."""
    token = create_access_token(
        data={"sub": admin_user.uuid, "email": admin_user.email, "role": "admin"}
    )
    response = client.post(
        "/api/admin/users",
        json={
            "name": "Another User",
            "email": regular_user.email,  # Duplicate
            "password": "Test1234a"
        },
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 400
    assert "already in use" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_user_detail_admin(client, admin_user, regular_user):
    """Test admin can get user detail."""
    token = create_access_token(
        data={"sub": admin_user.uuid, "email": admin_user.email, "role": "admin"}
    )
    response = client.get(
        f"/api/admin/users/{regular_user.uuid}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["uuid"] == regular_user.uuid
    assert data["email"] == regular_user.email


@pytest.mark.asyncio
async def test_list_products_admin(client, test_db, admin_user):
    """Test admin can list products."""
    # Create products
    p1 = Product(
        name="Token Package",
        price=9.99,
        product_type="token_package",
        token_amount=100
    )
    p2 = Product(
        name="License",
        price=29.99,
        product_type="license"
    )
    test_db.add(p1)
    test_db.add(p2)
    await test_db.commit()

    token = create_access_token(
        data={"sub": admin_user.uuid, "email": admin_user.email, "role": "admin"}
    )
    # Use public products endpoint (no admin-specific list endpoint)
    response = client.get(
        "/api/products",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2

