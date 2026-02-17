"""Tests for product endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database import Base
from main import app
from app.database import get_db
from app.models.user import User
from app.models.product import Product
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
async def test_list_products(client, test_db):
    """Test listing products."""
    # Add products
    p1 = Product(
        name="Token Package 100",
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
    
    # List products (no auth required)
    response = client.get("/api/products")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "Token Package 100"


@pytest.mark.asyncio
async def test_create_product_admin_only(client, test_db):
    """Test creating product requires admin."""
    # Create non-admin user
    user = User(
        name="Test User",
        email="test@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="user"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Try to create product as non-admin
    token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    response = client.post(
        "/api/admin/products",
        json={
            "name": "Test Product",
            "price_cents": 999,
            "product_type": "token_package",
            "token_amount": 100
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_product_admin(client, test_db):
    """Test creating product as admin."""
    # Create admin user
    user = User(
        name="Admin User",
        email="admin@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="admin"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Create product as admin
    token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    response = client.post(
        "/api/admin/products",
        json={
            "name": "Test Product",
            "price_cents": 999,
            "product_type": "token_package",
            "token_amount": 100
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Product"
    assert data["price"] == 9.99


@pytest.mark.asyncio
async def test_update_product(client, test_db):
    """Test updating product."""
    # Create admin user
    user = User(
        name="Admin User",
        email="admin@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="admin"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Create product
    product = Product(
        name="Test Product",
        price=9.99,
        product_type="token_package",
        token_amount=100
    )
    test_db.add(product)
    await test_db.commit()
    await test_db.refresh(product)
    
    # Update product
    token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    response = client.put(
        f"/api/admin/products/{product.uuid}",
        json={"name": "Updated Product"},
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Product"


@pytest.mark.asyncio
async def test_delete_product(client, test_db):
    """Test deleting product."""
    # Create admin user
    user = User(
        name="Admin User",
        email="admin@example.com",
        password_hash=hash_password("TestPass123"),
        user_role="admin"
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    # Create product
    product = Product(
        name="Test Product",
        price=9.99,
        product_type="token_package",
        token_amount=100
    )
    test_db.add(product)
    await test_db.commit()
    await test_db.refresh(product)
    
    # Delete product
    token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    response = client.delete(
        f"/api/admin/products/{product.uuid}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 204

