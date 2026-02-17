"""Tests for Stripe Connect endpoints."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app
from app.database import get_db
from app.models.user import User
from app.auth.security import hash_password, create_access_token


@pytest.fixture
def client(test_db):
    """Create test client."""
    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.fixture
async def test_user(test_db):
    """Create test user."""
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
    return user


@pytest.mark.asyncio
async def test_connect_onboard_new_account(client, test_db, test_user):
    """Test creating new Stripe Connect account."""
    with patch("stripe.Account.create") as mock_create, \
         patch("stripe.AccountLink.create") as mock_link:

        mock_create.return_value = MagicMock(id="acct_test123")
        mock_link.return_value = MagicMock(url="https://connect.stripe.com/onboarding/test")

        token = create_access_token(
            data={"sub": test_user.uuid, "email": test_user.email, "role": "user"}
        )
        response = client.post(
            "/api/connect/onboard",
            headers={"Authorization": f"Bearer {token}"}
        )

    assert response.status_code == 200
    data = response.json()
    assert "url" in data
    assert "stripe.com" in data["url"]


@pytest.mark.asyncio
async def test_connect_onboard_existing_account(client, test_db, test_user):
    """Test onboarding with existing Connect account."""
    # Set existing account
    test_user.stripe_connect_account_id = "acct_existing123"
    await test_db.commit()

    with patch("stripe.AccountLink.create") as mock_link:
        mock_link.return_value = MagicMock(url="https://connect.stripe.com/onboarding/test")

        token = create_access_token(
            data={"sub": test_user.uuid, "email": test_user.email, "role": "user"}
        )
        response = client.post(
            "/api/connect/onboard",
            headers={"Authorization": f"Bearer {token}"}
        )

    assert response.status_code == 200
    data = response.json()
    assert "url" in data


@pytest.mark.asyncio
async def test_connect_status_not_started(client, test_user):
    """Test Connect status when no account exists."""
    token = create_access_token(
        data={"sub": test_user.uuid, "email": test_user.email, "role": "user"}
    )
    response = client.get(
        "/api/connect/status",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_started"
    assert data["account_id"] is None


@pytest.mark.asyncio
async def test_connect_status_pending(client, test_db, test_user):
    """Test Connect status when account is pending."""
    test_user.stripe_connect_account_id = "acct_pending123"
    await test_db.commit()
    
    with patch("stripe.Account.retrieve") as mock_retrieve:
        mock_retrieve.return_value = MagicMock(
            charges_enabled=False,
            details_submitted=False
        )
        
        token = create_access_token(
            data={"sub": test_user.uuid, "email": test_user.email, "role": "user"}
        )
        response = client.get(
            "/api/connect/status",
            headers={"Authorization": f"Bearer {token}"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert data["account_id"] == "acct_pending123"


@pytest.mark.asyncio
async def test_connect_status_complete(client, test_db, test_user):
    """Test Connect status when account is fully onboarded."""
    test_user.stripe_connect_account_id = "acct_complete123"
    await test_db.commit()
    
    with patch("stripe.Account.retrieve") as mock_retrieve:
        mock_retrieve.return_value = MagicMock(
            charges_enabled=True,
            details_submitted=True
        )
        
        token = create_access_token(
            data={"sub": test_user.uuid, "email": test_user.email, "role": "user"}
        )
        response = client.get(
            "/api/connect/status",
            headers={"Authorization": f"Bearer {token}"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "complete"
    assert data["account_id"] == "acct_complete123"


@pytest.mark.asyncio
async def test_connect_dashboard_link_no_account(client, test_user):
    """Test dashboard link fails without Connect account."""
    token = create_access_token(
        data={"sub": test_user.uuid, "email": test_user.email, "role": "user"}
    )
    response = client.get(
        "/api/connect/dashboard-link",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 400
    assert "no stripe connect account" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_connect_dashboard_link_success(client, test_db, test_user):
    """Test getting dashboard link for onboarded account."""
    test_user.stripe_connect_account_id = "acct_complete123"
    await test_db.commit()
    
    with patch("stripe.Account.create_login_link") as mock_link:
        mock_link.return_value = MagicMock(url="https://dashboard.stripe.com/test")
        
        token = create_access_token(
            data={"sub": test_user.uuid, "email": test_user.email, "role": "user"}
        )
        response = client.get(
            "/api/connect/dashboard-link",
            headers={"Authorization": f"Bearer {token}"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "url" in data
    assert "stripe.com" in data["url"]


@pytest.mark.asyncio
async def test_connect_unauthenticated(client):
    """Test Connect endpoints require authentication."""
    response = client.get("/api/connect/status")
    assert response.status_code == 401

    response = client.post("/api/connect/onboard")
    assert response.status_code == 401

    response = client.get("/api/connect/dashboard-link")
    assert response.status_code == 401

