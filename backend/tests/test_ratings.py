"""Tests for rating endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database import Base
from main import app
from app.database import get_db
from app.models.user import User
from app.models.strategy import Strategy
from app.models.rating import Rating
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
async def test_create_rating(client, test_db):
    """Test creating a rating."""
    # Create users
    user1 = User(
        name="User 1",
        email="user1@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    user2 = User(
        name="User 2",
        email="user2@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    test_db.add(user1)
    test_db.add(user2)
    await test_db.commit()
    await test_db.refresh(user1)
    await test_db.refresh(user2)
    
    # Create strategy
    strategy = Strategy(
        name="Test Strategy",
        user_id=user1.uuid
    )
    test_db.add(strategy)
    await test_db.commit()
    await test_db.refresh(strategy)
    
    # Create rating
    token = create_access_token(data={"sub": user2.uuid, "email": user2.email, "role": user2.user_role})
    response = client.post(
        f"/api/strategies/{strategy.uuid}/ratings",
        json={"score": 5, "review": "Great strategy!"},
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["rating"] == 5
    assert data["review_text"] == "Great strategy!"


@pytest.mark.asyncio
async def test_duplicate_rating_not_allowed(client, test_db):
    """Test that users can only rate once per strategy."""
    # Create users
    user1 = User(
        name="User 1",
        email="user1@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    user2 = User(
        name="User 2",
        email="user2@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    test_db.add(user1)
    test_db.add(user2)
    await test_db.commit()
    await test_db.refresh(user1)
    await test_db.refresh(user2)
    
    # Create strategy
    strategy = Strategy(
        name="Test Strategy",
        user_id=user1.uuid
    )
    test_db.add(strategy)
    await test_db.commit()
    await test_db.refresh(strategy)
    
    # Create first rating
    token = create_access_token(data={"sub": user2.uuid, "email": user2.email, "role": user2.user_role})
    response = client.post(
        f"/api/strategies/{strategy.uuid}/ratings",
        json={"score": 5, "review": "Great!"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    
    # Try to create duplicate rating
    response = client.post(
        f"/api/strategies/{strategy.uuid}/ratings",
        json={"score": 3, "review": "Actually not great"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_strategy_ratings(client, test_db):
    """Test getting ratings for a strategy."""
    # Create users
    user1 = User(
        name="User 1",
        email="user1@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    user2 = User(
        name="User 2",
        email="user2@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    test_db.add(user1)
    test_db.add(user2)
    await test_db.commit()
    await test_db.refresh(user1)
    await test_db.refresh(user2)
    
    # Create strategy
    strategy = Strategy(
        name="Test Strategy",
        user_id=user1.uuid
    )
    test_db.add(strategy)
    await test_db.commit()
    await test_db.refresh(strategy)
    
    # Add ratings
    r1 = Rating(rating=5, user_id=user1.uuid, strategy_id=strategy.uuid)
    r2 = Rating(rating=4, user_id=user2.uuid, strategy_id=strategy.uuid)
    test_db.add(r1)
    test_db.add(r2)
    await test_db.commit()
    
    # Get ratings
    response = client.get(f"/api/strategies/{strategy.uuid}/ratings")
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["ratings"]) == 2


@pytest.mark.asyncio
async def test_update_rating(client, test_db):
    """Test updating own rating."""
    # Create users
    user1 = User(
        name="User 1",
        email="user1@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    user2 = User(
        name="User 2",
        email="user2@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    test_db.add(user1)
    test_db.add(user2)
    await test_db.commit()
    await test_db.refresh(user1)
    await test_db.refresh(user2)
    
    # Create strategy and rating
    strategy = Strategy(name="Test Strategy", user_id=user1.uuid)
    test_db.add(strategy)
    await test_db.commit()
    await test_db.refresh(strategy)
    
    rating = Rating(rating=3, user_id=user2.uuid, strategy_id=strategy.uuid)
    test_db.add(rating)
    await test_db.commit()
    await test_db.refresh(rating)
    
    # Update rating
    token = create_access_token(data={"sub": user2.uuid, "email": user2.email, "role": user2.user_role})
    response = client.put(
        f"/api/ratings/{rating.uuid}",
        json={"score": 5},
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["rating"] == 5


@pytest.mark.asyncio
async def test_delete_rating_admin(client, test_db):
    """Test admin deleting a rating."""
    # Create users
    user1 = User(
        name="User 1",
        email="user1@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    admin = User(
        name="Admin",
        email="admin@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="admin"
    )
    test_db.add(user1)
    test_db.add(admin)
    await test_db.commit()
    await test_db.refresh(user1)
    await test_db.refresh(admin)
    
    # Create strategy and rating
    strategy = Strategy(name="Test Strategy", user_id=user1.uuid)
    test_db.add(strategy)
    await test_db.commit()
    await test_db.refresh(strategy)
    
    rating = Rating(rating=3, user_id=user1.uuid, strategy_id=strategy.uuid)
    test_db.add(rating)
    await test_db.commit()
    await test_db.refresh(rating)
    
    # Delete rating as admin
    token = create_access_token(data={"sub": admin.uuid, "email": admin.email, "role": admin.user_role})
    response = client.delete(
        f"/api/admin/ratings/{rating.uuid}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 204

