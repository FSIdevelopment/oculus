"""Authentication schemas for user registration, login, and token management."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


class UserRegister(BaseModel):
    """Schema for user registration."""
    
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8)
    phone_number: Optional[str] = Field(None, max_length=20)
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength: min 8 chars, at least 1 uppercase, 1 lowercase, 1 number."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLogin(BaseModel):
    """Schema for user login."""
    
    email: EmailStr
    password: str


class Token(BaseModel):
    """Schema for JWT token response."""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for JWT token payload data."""

    user_id: str
    email: str
    role: str


class RefreshTokenRequest(BaseModel):
    """Schema for refresh token request."""

    refresh_token: str


class UserResponse(BaseModel):
    """Schema for user profile response (never expose password_hash)."""
    
    uuid: str
    name: str
    email: str
    phone_number: Optional[str] = None
    balance: float
    user_role: str
    created_at: datetime
    
    class Config:
        from_attributes = True

