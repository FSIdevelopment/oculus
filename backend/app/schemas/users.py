"""User management schemas for self-service and admin endpoints."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, field_validator


class UserUpdate(BaseModel):
    """Schema for user self-service profile update."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = Field(None, max_length=20)
    password: Optional[str] = Field(None, min_length=8)
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength if provided."""
        if v is None:
            return v
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class PayoutUpdate(BaseModel):
    """Schema for payout account information update."""
    
    payout_account: Optional[str] = Field(None, max_length=255)
    payout_method: Optional[str] = Field(None, max_length=50)  # "bank", "paypal", etc.


class BalanceResponse(BaseModel):
    """Schema for user balance response."""
    
    balance: float
    currency: str = "USD"


class UserAdminUpdate(BaseModel):
    """Schema for admin user update (can change role, status, etc.)."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = Field(None, max_length=20)
    user_role: Optional[str] = Field(None, max_length=50)  # "user", "admin"
    status: Optional[str] = Field(None, max_length=50)  # "active", "inactive"
    balance: Optional[float] = None


class AdminUserCreate(BaseModel):
    """Schema for admin creating a new user."""

    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8)
    phone_number: Optional[str] = Field(None, max_length=20)
    user_role: str = Field(default="user", max_length=50)  # "user", "admin"
    status: str = Field(default="active", max_length=50)  # "active", "inactive"

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


class UserDetailResponse(BaseModel):
    """Schema for detailed user response (admin view)."""

    uuid: str
    name: str
    email: str
    phone_number: Optional[str] = None
    balance: float
    user_role: str
    status: str
    stripe_customer_id: Optional[str] = None
    stripe_connect_account_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserListItem(BaseModel):
    """Schema for user list item."""
    
    uuid: str
    name: str
    email: str
    user_role: str
    status: str
    balance: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """Schema for paginated user list response."""
    
    items: List[UserListItem]
    total: int
    skip: int
    limit: int


class ImpersonationTokenResponse(BaseModel):
    """Schema for impersonation token response."""
    
    access_token: str
    token_type: str = "bearer"
    impersonated_user_id: str
    impersonator_id: str

