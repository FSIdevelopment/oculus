"""Priority management for build queue."""
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.license import License

logger = logging.getLogger(__name__)


class PriorityManager:
    """Manages build priority based on user attributes."""
    
    # Priority levels
    PRIORITY_ADMIN = 100
    PRIORITY_ANNUAL = 50
    PRIORITY_MONTHLY = 25
    PRIORITY_DEFAULT = 0
    
    @staticmethod
    async def get_user_priority(db: AsyncSession, user_id: str) -> int:
        """Get build priority for a user.
        
        Priority is determined by:
        1. User role (admin gets highest priority)
        2. Active license type (annual > monthly)
        3. Default priority for regular users
        
        Args:
            db: Database session
            user_id: User UUID
            
        Returns:
            Priority level (higher = more important)
        """
        try:
            # Get user
            result = await db.execute(
                select(User).where(User.uuid == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                logger.warning(f"User {user_id} not found for priority calculation")
                return PriorityManager.PRIORITY_DEFAULT
            
            # Admin users get highest priority
            if user.user_role == "admin":
                logger.info(f"User {user_id} has admin priority")
                return PriorityManager.PRIORITY_ADMIN
            
            # Check for active licenses
            result = await db.execute(
                select(License).where(
                    License.user_id == user_id,
                    License.status == "active"
                ).order_by(License.created_at.desc())
            )
            licenses = result.scalars().all()
            
            if not licenses:
                logger.info(f"User {user_id} has no active licenses, using default priority")
                return PriorityManager.PRIORITY_DEFAULT
            
            # Get highest priority license type
            for license in licenses:
                if license.license_type == "annual":
                    logger.info(f"User {user_id} has annual license priority")
                    return PriorityManager.PRIORITY_ANNUAL
            
            # If no annual license, check for monthly
            for license in licenses:
                if license.license_type == "monthly":
                    logger.info(f"User {user_id} has monthly license priority")
                    return PriorityManager.PRIORITY_MONTHLY
            
            # Default priority
            logger.info(f"User {user_id} using default priority")
            return PriorityManager.PRIORITY_DEFAULT
            
        except Exception as e:
            logger.error(f"Failed to get priority for user {user_id}: {e}")
            # Return default priority on error to avoid blocking builds
            return PriorityManager.PRIORITY_DEFAULT
    
    @staticmethod
    def get_priority_description(priority: int) -> str:
        """Get human-readable description of priority level.
        
        Args:
            priority: Priority level
            
        Returns:
            Description string
        """
        if priority >= PriorityManager.PRIORITY_ADMIN:
            return "Admin (Highest Priority)"
        elif priority >= PriorityManager.PRIORITY_ANNUAL:
            return "Annual License (High Priority)"
        elif priority >= PriorityManager.PRIORITY_MONTHLY:
            return "Monthly License (Medium Priority)"
        else:
            return "Standard (Normal Priority)"

