"""Tests for balance and transaction endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database import Base
from main import app
from app.database import get_db
from app.models.user import User
from app.models.balance import Balance
from app.auth.security import create_access_token, hash_password
from app.services.balance import deduct_tokens, add_tokens


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
async def test_get_balance(client, test_db):
    """Test getting current balance."""
    # Create user with balance
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        balance=100.0,
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Get balance
    token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    response = client.get(
        "/api/users/me/balance",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["balance"] == 100.0


@pytest.mark.asyncio
async def test_get_balance_history(client, test_db):
    """Test getting balance history."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        balance=100.0,
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Add transactions
    t1 = Balance(
        tokens=50.0,
        transaction_type="purchase",
        description="Token purchase",
        user_id=user.uuid,
        status="completed"
    )
    t2 = Balance(
        tokens=10.0,
        transaction_type="deduction",
        description="Strategy usage",
        user_id=user.uuid,
        status="completed"
    )
    test_db.add(t1)
    test_db.add(t2)
    await test_db.commit()
    
    # Get history
    token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    response = client.get(
        "/api/users/me/balance/history",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["transactions"]) == 2


@pytest.mark.asyncio
async def test_deduct_tokens_success(test_db):
    """Test successful token deduction."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        balance=100.0,
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Deduct tokens
    transaction = await deduct_tokens(test_db, user.uuid, 30.0, "Test deduction")
    await test_db.commit()
    
    assert transaction.tokens == 30.0
    assert transaction.transaction_type == "deduction"
    
    # Verify balance updated
    await test_db.refresh(user)
    assert user.balance == 70.0


@pytest.mark.asyncio
async def test_deduct_tokens_insufficient_balance(test_db):
    """Test deduction with insufficient balance."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        balance=10.0,
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Try to deduct more than balance
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await deduct_tokens(test_db, user.uuid, 50.0, "Test deduction")
    
    assert exc_info.value.status_code == 402


@pytest.mark.asyncio
async def test_add_tokens(test_db):
    """Test adding tokens."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        balance=100.0,
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Add tokens
    transaction = await add_tokens(test_db, user.uuid, 50.0, "Test addition")
    await test_db.commit()
    
    assert transaction.tokens == 50.0
    assert transaction.transaction_type == "purchase"
    
    # Verify balance updated
    await test_db.refresh(user)
    assert user.balance == 150.0

