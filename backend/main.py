"""
FastAPI application entry point for Oculus Strategy Platform.
"""
import logging
import asyncio
import multiprocessing
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.config import settings
from app.logging_config import setup_logging
from app.routers import auth, users, admin, chat, balance, products, ratings, strategies, licenses, subscriptions, payments, builds, connect, worker
from app.services.scheduler import start_scheduler, stop_scheduler

# Get logger for request logging
logger = logging.getLogger(__name__)

# Configure logging
setup_logging()


async def generate_missing_readmes_background():
    """Background task to generate READMEs for builds that are missing them.

    Only runs on the master process to avoid duplicate work across workers.
    Waits 30 seconds after startup to ensure the app is fully ready.
    """
    try:
        # Wait 30 seconds to ensure app is fully started and health checks pass
        await asyncio.sleep(30)

        # Import here to avoid circular dependencies
        from scripts.generate_missing_readmes import main as generate_readmes

        logger.info("Starting background README generation...")
        await generate_readmes()
        logger.info("Background README generation complete")
    except Exception as e:
        logger.error(f"Background README generation failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting up Oculus Strategy API...")

    # Only start scheduler and background tasks on SpawnProcess-1 (master worker)
    # This prevents duplicate work when running with multiple uvicorn workers
    current_process_name = multiprocessing.current_process().name
    is_master = current_process_name == "SpawnProcess-1"

    if is_master:
        start_scheduler()
        # Start README generation in background after a delay (don't await - let it run async)
        asyncio.create_task(generate_missing_readmes_background())
    else:
        logger.info(f"Skipping background tasks on {current_process_name}")

    yield
    # Shutdown
    logger.info("Shutting down Oculus Strategy API...")
    if is_master:
        stop_scheduler()


app = FastAPI(
    title="Oculus Strategy API",
    description="Backend API for the Oculus Strategy Platform",
    version="0.1.0",
    lifespan=lifespan
)

# Configure rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Parse CORS origins from config
# In development mode, allow all origins for easier local development
if settings.ENVIRONMENT == "development" or settings.DEBUG:
    cors_origins = ["*"]
else:
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

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with method, path, origin, and response status."""
    origin = request.headers.get("origin", "no-origin")
    logger.info(f"Request: {request.method} {request.url.path} | Origin: {origin}")

    response = await call_next(request)

    logger.info(f"Response: {request.method} {request.url.path} | Status: {response.status_code}")
    return response

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
app.include_router(connect.router, tags=["connect"])
app.include_router(worker.router, tags=["worker"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

