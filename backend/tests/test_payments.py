"""Tests for payment endpoints."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database import Base
from main import app
from app.database import get_db
from app.models.user import User
from app.models.product import Product
from app.models.purchase import Purchase
from app.auth.security import create_access_token, hash_password


@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as session:
        yield session


@pytest.fixture
def client(test_db):
    """Create test client."""
    async def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.mark.asyncio
async def test_create_payment_intent_success(client, test_db):
    """Test creating a payment intent successfully."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        balance=0.0
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
    
    # Mock Stripe API
    with patch("stripe.Customer.create") as mock_customer, \
         patch("stripe.PaymentIntent.create") as mock_intent:
        
        mock_customer.return_value = MagicMock(id="cus_test123")
        mock_intent.return_value = MagicMock(
            client_secret="pi_test_secret",
            id="pi_test123"
        )
        
        token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": "user"})
        response = client.post(
            "/api/payments/create-intent",
            json={"product_id": product.uuid},
            headers={"Authorization": f"Bearer {token}"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["client_secret"] == "pi_test_secret"
    assert data["product_id"] == product.uuid
    assert data["amount"] == 999  # 9.99 * 100


@pytest.mark.asyncio
async def test_create_payment_intent_product_not_found(client, test_db):
    """Test creating payment intent with non-existent product."""
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123")
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": "user"})
    response = client.post(
        "/api/payments/create-intent",
        json={"product_id": "nonexistent"},
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_payment_intent_non_token_product(client, test_db):
    """Test creating payment intent for non-token product."""
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123")
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Create license product
    product = Product(
        name="Monthly License",
        price=29.99,
        product_type="license"
    )
    test_db.add(product)
    await test_db.commit()
    await test_db.refresh(product)
    
    token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": "user"})
    response = client.post(
        "/api/payments/create-intent",
        json={"product_id": product.uuid},
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 400
    assert "token" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_confirm_payment_success(client, test_db):
    """Test confirming payment and adding tokens."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        balance=0.0,
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
    
    # Mock Stripe API
    with patch("stripe.PaymentIntent.retrieve") as mock_retrieve:
        mock_retrieve.return_value = MagicMock(
            status="succeeded",
            id="pi_test123"
        )
        
        token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": "user"})
        response = client.post(
            "/api/payments/confirm",
            json={
                "payment_intent_id": "pi_test123",
                "product_id": product.uuid
            },
            headers={"Authorization": f"Bearer {token}"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["tokens_added"] == 100
    assert data["new_balance"] == 100.0


@pytest.mark.asyncio
async def test_confirm_payment_idempotency(client, test_db):
    """Test payment confirmation is idempotent."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        balance=100.0,
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

    # Create existing purchase
    purchase = Purchase(
        user_id=user.uuid,
        product_id=product.uuid,
        product_type="token_package",
        stripe_id="pi_test123",
        purchased_at=None
    )
    test_db.add(purchase)
    await test_db.commit()
    
    # Mock Stripe API
    with patch("stripe.PaymentIntent.retrieve") as mock_retrieve:
        mock_retrieve.return_value = MagicMock(
            status="succeeded",
            id="pi_test123"
        )
        
        token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": "user"})
        response = client.post(
            "/api/payments/confirm",
            json={
                "payment_intent_id": "pi_test123",
                "product_id": product.uuid
            },
            headers={"Authorization": f"Bearer {token}"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["message"] == "Payment already processed"
    # Balance should not change
    assert data["new_balance"] == 100.0


@pytest.mark.asyncio
async def test_confirm_payment_not_succeeded(client, test_db):
    """Test confirming payment that hasn't succeeded."""
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        stripe_customer_id="cus_test123"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    product = Product(
        name="100 Tokens",
        price=9.99,
        product_type="token_package",
        token_amount=100
    )
    test_db.add(product)
    await test_db.commit()
    await test_db.refresh(product)

    with patch("stripe.PaymentIntent.retrieve") as mock_retrieve:
        mock_retrieve.return_value = MagicMock(
            status="processing",
            id="pi_test123"
        )
        
        token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": "user"})
        response = client.post(
            "/api/payments/confirm",
            json={
                "payment_intent_id": "pi_test123",
                "product_id": product.uuid
            },
            headers={"Authorization": f"Bearer {token}"}
        )
    
    assert response.status_code == 400
    assert "not succeeded" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_webhook_payment_succeeded(client, test_db):
    """Test webhook handling for payment.intent.succeeded event."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        balance=0.0
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

    # Mock Stripe webhook verification
    with patch("stripe.Webhook.construct_event") as mock_event:
        mock_event.return_value = {
            "type": "payment_intent.succeeded",
            "data": {
                "object": {
                    "id": "pi_test123",
                    "metadata": {
                        "user_id": user.uuid,
                        "product_id": product.uuid
                    }
                }
            }
        }
        
        response = client.post(
            "/api/webhooks/stripe",
            json={},
            headers={"stripe-signature": "test_signature"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processed"

