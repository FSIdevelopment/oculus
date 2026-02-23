"""Checkpoint management for build recovery."""
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.strategy_build import StrategyBuild

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages build checkpoints for recovery."""
    
    @staticmethod
    async def save_checkpoint(
        db: AsyncSession,
        build_id: str,
        phase: str,
        iteration: Optional[int] = None,
        step: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """Save a checkpoint for a build.
        
        Args:
            db: Database session
            build_id: Build UUID
            phase: Current phase (e.g., "designing", "training", "building_docker")
            iteration: Current iteration number (if applicable)
            step: Current step within phase (if applicable)
            data: Additional checkpoint data
        """
        try:
            result = await db.execute(
                select(StrategyBuild).where(StrategyBuild.uuid == build_id)
            )
            build = result.scalar_one_or_none()
            
            if not build:
                logger.error(f"Build {build_id} not found for checkpoint")
                return
            
            checkpoint = {
                "phase": phase,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            if iteration is not None:
                checkpoint["iteration"] = iteration
            
            if step:
                checkpoint["step"] = step
            
            if data:
                checkpoint["data"] = data
            
            build.last_checkpoint = checkpoint
            build.last_heartbeat = datetime.utcnow()
            
            await db.commit()
            
            logger.info(
                f"Checkpoint saved for build {build_id}: "
                f"phase={phase}, iteration={iteration}, step={step}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint for build {build_id}: {e}")
            await db.rollback()
    
    @staticmethod
    async def get_checkpoint(
        db: AsyncSession,
        build_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the last checkpoint for a build.
        
        Args:
            db: Database session
            build_id: Build UUID
            
        Returns:
            Checkpoint data or None
        """
        try:
            result = await db.execute(
                select(StrategyBuild).where(StrategyBuild.uuid == build_id)
            )
            build = result.scalar_one_or_none()
            
            if not build:
                return None
            
            return build.last_checkpoint
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint for build {build_id}: {e}")
            return None
    
    @staticmethod
    async def clear_checkpoint(db: AsyncSession, build_id: str):
        """Clear checkpoint for a build.
        
        Args:
            db: Database session
            build_id: Build UUID
        """
        try:
            result = await db.execute(
                select(StrategyBuild).where(StrategyBuild.uuid == build_id)
            )
            build = result.scalar_one_or_none()
            
            if build:
                build.last_checkpoint = None
                await db.commit()
                logger.info(f"Checkpoint cleared for build {build_id}")
            
        except Exception as e:
            logger.error(f"Failed to clear checkpoint for build {build_id}: {e}")
            await db.rollback()
    
    @staticmethod
    async def should_resume_from_checkpoint(
        db: AsyncSession,
        build_id: str
    ) -> bool:
        """Check if build should resume from checkpoint.
        
        Args:
            db: Database session
            build_id: Build UUID
            
        Returns:
            True if should resume, False otherwise
        """
        try:
            checkpoint = await CheckpointManager.get_checkpoint(db, build_id)
            
            if not checkpoint:
                return False
            
            # Check if checkpoint is recent (within last hour)
            checkpoint_time = datetime.fromisoformat(checkpoint["timestamp"])
            age = datetime.utcnow() - checkpoint_time
            
            # Resume if checkpoint is less than 1 hour old
            return age.total_seconds() < 3600
            
        except Exception as e:
            logger.error(f"Failed to check resume status for build {build_id}: {e}")
            return False

