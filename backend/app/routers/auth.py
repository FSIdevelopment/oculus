"""Authentication router for user registration, login, and token management."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import stripe

from app.database import get_db
from app.config import settings
from app.models.user import User
from app.schemas.auth import UserRegister, UserLogin, Token, UserResponse, RefreshTokenRequest
from app.auth.security import hash_password, verify_password, create_access_token, create_refresh_token, decode_token
from app.auth.dependencies import get_current_active_user

router = APIRouter()

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


@router.post("/register", response_model=Token)
async def register(user_data: UserRegister, db: AsyncSession = Depends(get_db)):
    """
    Register a new user.
    
    - Validates input (password strength, email format)
    - Checks email uniqueness
    - Hashes password with bcrypt
    - Creates user with UUID
    - Creates Stripe customer (non-blocking)
    - Returns JWT tokens
    """
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    new_user = User(
        name=user_data.name,
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        phone_number=user_data.phone_number,
        balance=0.0,
        status="active",
        user_role="user"
    )
    
    # Try to create Stripe customer (non-blocking)
    if settings.STRIPE_SECRET_KEY:
        try:
            customer = stripe.Customer.create(
                email=user_data.email,
                name=user_data.name
            )
            new_user.stripe_customer_id = customer.id
        except Exception:
            # Don't fail registration if Stripe is unavailable
            pass
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    # Create tokens
    access_token = create_access_token(data={"sub": new_user.uuid, "email": new_user.email, "role": new_user.user_role})
    refresh_token = create_refresh_token(data={"sub": new_user.uuid, "email": new_user.email, "role": new_user.user_role})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    """
    Login with email and password.
    
    - Validates credentials
    - Returns JWT access + refresh tokens
    """
    # Find user by email
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create tokens
    access_token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    refresh_token = create_refresh_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=Token)
async def refresh(request: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    """
    Refresh an expired access token.

    - Accepts refresh token
    - Validates and returns new access token
    """
    token = request.refresh_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh token required"
        )
    
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    # Fetch user to verify they still exist
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create new access token
    access_token = create_access_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    new_refresh_token = create_refresh_token(data={"sub": user.uuid, "email": user.email, "role": user.user_role})
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_active_user)):
    """
    Get current user profile.
    
    - Protected endpoint (requires valid JWT)
    - Returns current user profile (UserResponse schema)
    """
    return current_user

