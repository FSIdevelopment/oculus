"""
FastAPI application entry point for Oculus Strategy Platform.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.config import settings
from app.logging_config import setup_logging
from app.routers import auth, users, admin, chat, balance, products, ratings, strategies, licenses, subscriptions, payments, builds

# Configure logging
setup_logging()

app = FastAPI(
    title="Oculus Strategy API",
    description="Backend API for the Oculus Strategy Platform",
    version="0.1.0"
)

# Configure rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Rate limiting middleware for auth endpoints
@app.middleware("http")
async def rate_limit_auth_endpoints(request: Request, call_next):
    """Apply rate limiting to auth endpoints."""
    path = request.url.path

    # Apply rate limiting based on endpoint
    if path == "/api/auth/login":
        # 5 requests per minute for login
        try:
            await limiter.limit("5/minute")(request)
        except RateLimitExceeded:
            raise
    elif path == "/api/auth/register":
        # 3 requests per minute for register
        try:
            await limiter.limit("3/minute")(request)
        except RateLimitExceeded:
            raise

    return await call_next(request)

# Parse CORS origins from config
cors_origins = (
    ["*"] if settings.CORS_ORIGINS == "*"
    else [origin.strip() for origin in settings.CORS_ORIGINS.split(",")]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Register routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])
app.include_router(builds.router, tags=["builds"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(chat.router, tags=["chat"])
app.include_router(balance.router, tags=["balance"])
app.include_router(products.router, tags=["products"])
app.include_router(ratings.router, tags=["ratings"])
app.include_router(licenses.router, tags=["licenses"])
app.include_router(subscriptions.router, tags=["subscriptions"])
app.include_router(payments.router, tags=["payments"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

