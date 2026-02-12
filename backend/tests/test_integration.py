"""Integration tests for end-to-end API flows."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app
from app.database import get_db
from app.models.user import User
from app.models.product import Product
from app.models.strategy import Strategy
from app.auth.security import hash_password, create_access_token


@pytest.fixture
def client(test_db):
    """Create test client."""
    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.mark.asyncio
async def test_full_auth_flow(client, test_db):
    """Test complete auth flow: register -> login -> access protected endpoint."""
    # Register
    with patch("stripe.Customer.create") as mock_customer:
        mock_customer.return_value = MagicMock(id="cus_test123")
        
        register_response = client.post(
            "/api/auth/register",
            json={
                "name": "Test User",
                "email": "testuser@example.com",
                "password": "Test1234a"
            }
        )
    
    assert register_response.status_code == 200
    register_data = register_response.json()
    access_token = register_data["access_token"]
    
    # Access protected endpoint with token
    profile_response = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    assert profile_response.status_code == 200
    profile_data = profile_response.json()
    assert profile_data["email"] == "testuser@example.com"
    assert profile_data["name"] == "Test User"


@pytest.mark.asyncio
async def test_admin_product_creation_flow(client, test_db):
    """Test admin creating product and user listing it."""
    # Create admin user
    admin = User(
        name="Admin",
        email="admin@example.com",
        password_hash=hash_password("AdminPass123"),
        status="active",
        user_role="admin"
    )
    test_db.add(admin)
    await test_db.commit()
    await test_db.refresh(admin)
    
    admin_token = create_access_token(
        data={"sub": admin.uuid, "email": admin.email, "role": "admin"}
    )
    
    # Admin creates product
    create_response = client.post(
        "/api/admin/products",
        json={
            "name": "100 Tokens",
            "price_cents": 999,
            "product_type": "token_package",
            "token_amount": 100,
            "description": "100 trading tokens"
        },
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert create_response.status_code == 200
    product_data = create_response.json()
    product_id = product_data["uuid"]
    
    # User lists products (no auth required)
    list_response = client.get("/api/products")
    assert list_response.status_code == 200
    products = list_response.json()
    assert len(products) > 0
    assert any(p["uuid"] == product_id for p in products)


@pytest.mark.asyncio
async def test_full_token_purchase_flow(client, test_db):
    """Test complete token purchase: create product -> payment intent -> confirm."""
    # Create user
    user = User(
        name="Buyer",
        email="buyer@example.com",
        password_hash=hash_password("BuyerPass123"),
        balance=0.0,
        status="active",
        user_role="user",
        stripe_customer_id="cus_test123"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Create product
    product = Product(
        name="100 Tokens",
        price=9.99,
        product_type="token_package",
        token_amount=100
    )
    test_db.add(product)
    await test_db.commit()
    await test_db.refresh(product)
    
    user_token = create_access_token(
        data={"sub": user.uuid, "email": user.email, "role": "user"}
    )
    
    # Create payment intent
    with patch("stripe.Customer.create") as mock_customer, \
         patch("stripe.PaymentIntent.create") as mock_intent:
        
        mock_customer.return_value = MagicMock(id="cus_test123")
        mock_intent.return_value = MagicMock(
            client_secret="pi_test_secret",
            id="pi_test123"
        )
        
        intent_response = client.post(
            "/api/payments/create-intent",
            json={"product_id": product.uuid},
            headers={"Authorization": f"Bearer {user_token}"}
        )
    
    assert intent_response.status_code == 200
    intent_data = intent_response.json()
    assert intent_data["client_secret"] == "pi_test_secret"
    
    # Confirm payment
    with patch("stripe.PaymentIntent.retrieve") as mock_retrieve:
        mock_retrieve.return_value = MagicMock(
            status="succeeded",
            id="pi_test123"
        )
        
        confirm_response = client.post(
            "/api/payments/confirm",
            json={
                "payment_intent_id": "pi_test123",
                "product_id": product.uuid
            },
            headers={"Authorization": f"Bearer {user_token}"}
        )
    
    assert confirm_response.status_code == 200
    confirm_data = confirm_response.json()
    assert confirm_data["success"] is True
    assert confirm_data["tokens_added"] == 100
    assert confirm_data["new_balance"] == 100.0


@pytest.mark.asyncio
async def test_strategy_creation_flow(client, test_db):
    """Test creating strategy and verifying it appears in list."""
    # Create user
    user = User(
        name="Strategist",
        email="strategist@example.com",
        password_hash=hash_password("StratPass123"),
        status="active",
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    user_token = create_access_token(
        data={"sub": user.uuid, "email": user.email, "role": "user"}
    )
    
    # Create strategy
    create_response = client.post(
        "/api/strategies",
        json={
            "name": "Test Strategy",
            "description": "A test trading strategy",
            "config": {"symbols": ["AAPL", "MSFT"]}
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )

    assert create_response.status_code == 201
    strategy_data = create_response.json()
    strategy_id = strategy_data["uuid"]
    
    # List strategies
    list_response = client.get(
        "/api/strategies",
        headers={"Authorization": f"Bearer {user_token}"}
    )

    assert list_response.status_code == 200
    response_data = list_response.json()
    strategies = response_data["items"]
    assert any(s["uuid"] == strategy_id for s in strategies)


@pytest.mark.asyncio
async def test_admin_user_management_flow(client, test_db):
    """Test admin creating user and managing them."""
    # Create admin
    admin = User(
        name="Admin",
        email="admin@example.com",
        password_hash=hash_password("AdminPass123"),
        status="active",
        user_role="admin"
    )
    test_db.add(admin)
    await test_db.commit()
    await test_db.refresh(admin)
    
    admin_token = create_access_token(
        data={"sub": admin.uuid, "email": admin.email, "role": "admin"}
    )
    
    # Admin creates user
    with patch("stripe.Customer.create") as mock_customer:
        mock_customer.return_value = MagicMock(id="cus_test123")
        
        create_response = client.post(
            "/api/admin/users",
            json={
                "name": "New User",
                "email": "newuser@example.com",
                "password": "Test1234a",
                "user_role": "user"
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
    
    assert create_response.status_code == 200
    user_data = create_response.json()
    user_id = user_data["uuid"]
    
    # Admin lists users
    list_response = client.get(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert list_response.status_code == 200
    users_data = list_response.json()
    assert users_data["total"] >= 2
    assert any(u["uuid"] == user_id for u in users_data["items"])

