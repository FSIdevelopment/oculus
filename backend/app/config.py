"""Application configuration loaded from environment variables."""
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@postgres:5432/oculus_db"
    DATABASE_ECHO: bool = False

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def ensure_async_driver(cls, v: str) -> str:
        """Convert standard postgresql:// URL to postgresql+asyncpg:// for async SQLAlchemy."""
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        # asyncpg uses 'ssl' parameter, not 'sslmode' (psycopg2/libpq specific)
        if "sslmode=" in v:
            v = v.replace("sslmode=", "ssl=")
        return v
    
    # CORS - can be "*" for all origins or comma-separated list
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001,http://localhost:8000,https://app.oculusalgorithms.com,https://oculus-avvr7.ondigitalocean.app"
    
    # JWT
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Stripe
    STRIPE_SECRET_KEY: str = ""
    STRIPE_PUBLISHABLE_KEY: str = ""
    STRIPE_CONNECT_CLIENT_ID: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    FRONTEND_URL: str = "http://localhost:3000"

    # Stripe License Price IDs — no longer used for checkout (dynamic price_data is
    # used instead), but kept here in case a fixed-price fallback is ever needed.
    STRIPE_LICENSE_MONTHLY_PRICE_ID: str = ""
    STRIPE_LICENSE_ANNUAL_PRICE_ID: str = ""

    # ── Dynamic License Pricing ────────────────────────────────────────────────
    # Price bounds (whole USD).  Strategies at or below the minimum performance
    # thresholds are charged LICENSE_MONTHLY_MIN_PRICE; those at or above the
    # maximum thresholds are charged LICENSE_MONTHLY_MAX_PRICE.
    LICENSE_MONTHLY_MIN_PRICE: int = 50     # $ floor — minimum monthly price
    LICENSE_MONTHLY_MAX_PRICE: int = 1999  # $ ceiling — maximum monthly price
    # Annual price = monthly * multiplier  (10 ≈ 2 months free, ~17% discount)
    LICENSE_ANNUAL_MULTIPLIER: int = 10

    # Performance thresholds that define the min/max of the pricing curve.
    # total_return is expressed as a percentage (e.g. 10.0 = 10%).
    PERF_MIN_RETURN: float = 10.0   # % — maps to LICENSE_MONTHLY_MIN_PRICE
    PERF_MAX_RETURN: float = 350.0  # % — maps to LICENSE_MONTHLY_MAX_PRICE
    PERF_MIN_SHARPE: float = 0.5    # — maps to LICENSE_MONTHLY_MIN_PRICE
    PERF_MAX_SHARPE: float = 5.0    # — maps to LICENSE_MONTHLY_MAX_PRICE

    # Metric weights (must be > 0; normalised internally so they don't need to sum to 1).
    PERF_RETURN_WEIGHT: float = 0.85
    PERF_SHARPE_WEIGHT: float = 0.15

    # Power-curve exponent applied to the combined performance score.
    # < 1 gives an ease-in curve (steeper at the low end, rewards early improvement).
    # > 1 gives an ease-out curve (steeper near the top, rewards excellence).
    # 0.65 produces a gentle ease-in that feels natural across the range.
    PERF_CURVE_EXPONENT: float = 0.9

    # Data Provider Keys (centrally managed)
    POLYGON_API_KEY: str = ""
    ALPHAVANTAGE_API_KEY: str = ""
    ALPACA_API_KEY: str = ""
    ALPACA_API_SECRET: str = ""

    # Email
    RESEND_API_KEY: str = ""

    # Digital Ocean Container Registry
    DO_REGISTRY_URL: str = "registry.digitalocean.com/oculus-strategies"
    DO_REGISTRY_TOKEN: str = ""

    # LLM and Build Orchestration
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MAX_TOKENS: int = 10000  # Max tokens for Claude API responses (must be > thinking budget)
    TOKENS_PER_ITERATION: float = 10.0  # Tokens charged per build iteration (env var for quick pricing)
    MAX_BUILD_ITERATIONS: int = 5  # Default max iterations per build
    TUNNEL_URL: str = "http://localhost:4040"  # Ngrok tunnel to local M3 Max
    REDIS_URL: str = "redis://localhost:6379"
    TRAINING_QUEUE: str = "oculus:training_queue"

    # Admin Seed
    ADMIN_EMAIL: str = "admin@oculusalgorithms.com"
    ADMIN_PASSWORD: str = "Admin123!"
    ADMIN_NAME: str = "Admin"

    # Default User Seed
    DEFAULT_USER_EMAIL: str = "user@oculusalgorithms.com"
    DEFAULT_USER_PASSWORD: str = "User123!"
    DEFAULT_USER_NAME: str = "Default User"

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    class Config:
        env_file = (".env", "../.env")
        case_sensitive = True
        extra = "allow"


settings = Settings()

