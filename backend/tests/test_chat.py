"""Tests for chat history endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database import Base
from main import app
from app.database import get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.models.chat_history import ChatHistory
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
async def test_save_chat_message(client, test_db):
    """Test saving a chat message."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Create strategy
    strategy = Strategy(
        name="Test Strategy",
        user_id=user.uuid
    )
    test_db.add(strategy)
    await test_db.commit()
    await test_db.refresh(strategy)
    
    # Create token
    token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    
    # Save chat message
    response = client.post(
        f"/api/strategies/{strategy.uuid}/chat",
        json={"role": "user", "content": "Hello"},
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "user"
    assert data["content"] == "Hello"


@pytest.mark.asyncio
async def test_get_chat_history(client, test_db):
    """Test getting chat history."""
    # Create user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Create strategy
    strategy = Strategy(
        name="Test Strategy",
        user_id=user.uuid
    )
    test_db.add(strategy)
    await test_db.commit()
    await test_db.refresh(strategy)
    
    # Add chat messages
    msg1 = ChatHistory(
        message="Hello",
        message_type="user",
        strategy_id=strategy.uuid,
        user_id=user.uuid
    )
    msg2 = ChatHistory(
        message="Hi there",
        message_type="assistant",
        strategy_id=strategy.uuid,
        user_id=user.uuid
    )
    test_db.add(msg1)
    test_db.add(msg2)
    await test_db.commit()
    
    # Get chat history
    token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    response = client.get(
        f"/api/strategies/{strategy.uuid}/chat",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["role"] == "user"
    assert data[1]["role"] == "assistant"

