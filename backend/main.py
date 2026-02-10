"""
FastAPI application entry point for Oculus Strategy Platform.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import auth, users, admin, chat, balance, products, ratings, strategies

app = FastAPI(
    title="Oculus Strategy API",
    description="Backend API for the Oculus Strategy Platform",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(chat.router, tags=["chat"])
app.include_router(balance.router, tags=["balance"])
app.include_router(products.router, tags=["products"])
app.include_router(ratings.router, tags=["ratings"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

