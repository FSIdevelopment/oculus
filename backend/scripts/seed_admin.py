"""Admin seed script for Oculus Strategy API.

This script creates a default admin user on startup if one doesn't exist.
It is idempotent and safe to run on every container start.
"""

import asyncio
import logging
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.user import User
from app.auth.security import hash_password
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def seed_admin():
    """Create default admin user if it doesn't exist."""
    async with AsyncSessionLocal() as session:
        # Check if admin user already exists
        result = await session.execute(
            select(User).where(User.email == settings.ADMIN_EMAIL)
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            logger.info(f"Admin user already exists, skipping")
            return
        
        # Create new admin user
        admin_user = User(
            name=settings.ADMIN_NAME,
            email=settings.ADMIN_EMAIL,
            password_hash=hash_password(settings.ADMIN_PASSWORD),
            user_role="admin",
            status="active",
            balance=0.0
        )
        
        session.add(admin_user)
        await session.commit()
        
        logger.info(f"Admin user created: {settings.ADMIN_EMAIL}")


def main():
    """Entry point for the seed script."""
    asyncio.run(seed_admin())


if __name__ == "__main__":
    main()

