"""Build recovery service for handling stuck and failed builds."""
import logging
from datetime import datetime, timedelta
from typing import List
import redis.asyncio as redis
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.strategy_build import StrategyBuild
from app.services.build_queue import BuildQueueManager
from app.services.worker_health import WorkerHealthManager
from app.services.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class BuildRecoveryService:
    """Handles recovery of stuck and failed builds."""

    MAX_RETRIES = 3
    PROGRESS_TIMEOUT = timedelta(minutes=5)  # No progress for 5 minutes
    BASE_RETRY_DELAY = timedelta(minutes=2)  # Base delay for exponential backoff
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize recovery service.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.queue_manager = BuildQueueManager(redis_client)
        self.worker_manager = WorkerHealthManager()
    
    async def recover_stuck_builds(self, db: AsyncSession) -> dict:
        """Recover builds that are stuck or assigned to dead workers.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with recovery statistics
        """
        stats = {
            "recovered": 0,
            "failed": 0,
            "requeued": 0,
            "dead_workers": 0
        }
        
        try:
            # Step 1: Mark dead workers
            dead_worker_ids = await self.worker_manager.mark_dead_workers(db)
            stats["dead_workers"] = len(dead_worker_ids)
            
            # Step 2: Reassign builds from dead workers
            for worker_id in dead_worker_ids:
                reassigned = await self.worker_manager.reassign_builds_from_dead_worker(db, worker_id)
                stats["requeued"] += len(reassigned)
                logger.warning(f"Reassigned {len(reassigned)} builds from dead worker {worker_id}")
            
            # Step 3: Find builds with no progress
            threshold = datetime.utcnow() - self.PROGRESS_TIMEOUT
            now = datetime.utcnow()

            result = await db.execute(
                select(StrategyBuild).where(
                    and_(
                        StrategyBuild.status.in_([
                            "building", "designing", "training",
                            "extracting_rules", "optimizing", "building_docker"
                        ]),
                        or_(
                            StrategyBuild.last_heartbeat < threshold,
                            StrategyBuild.last_heartbeat.is_(None)
                        ),
                        # Only retry if retry_after is None or has passed
                        or_(
                            StrategyBuild.retry_after.is_(None),
                            StrategyBuild.retry_after <= now
                        )
                    )
                )
            )
            stuck_builds = result.scalars().all()
            
            # Step 4: Recover or fail stuck builds
            for build in stuck_builds:
                if build.retry_count >= self.MAX_RETRIES:
                    # Max retries reached - fail the build
                    await self._fail_build(db, build, "Max retries exceeded")
                    stats["failed"] += 1
                else:
                    # Retry the build
                    await self._retry_build(db, build)
                    stats["recovered"] += 1
            
            logger.info(
                f"Build recovery complete: {stats['recovered']} recovered, "
                f"{stats['failed']} failed, {stats['requeued']} requeued, "
                f"{stats['dead_workers']} dead workers"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to recover stuck builds: {e}")
            return stats
    
    async def _retry_build(self, db: AsyncSession, build: StrategyBuild):
        """Retry a stuck build from last checkpoint with exponential backoff.

        Args:
            db: Database session
            build: StrategyBuild instance
        """
        try:
            # Increment retry count
            build.retry_count += 1

            # Calculate exponential backoff delay: 2^(retry_count) * BASE_DELAY
            # Retry 1: 2 minutes, Retry 2: 4 minutes, Retry 3: 8 minutes
            backoff_multiplier = 2 ** (build.retry_count - 1)
            retry_delay = self.BASE_RETRY_DELAY * backoff_multiplier
            build.retry_after = datetime.utcnow() + retry_delay

            # Clear worker assignment
            build.assigned_worker_id = None
            build.last_heartbeat = None

            # Set status back to queued
            build.status = "queued"

            # Requeue the build with lower priority (higher retry count = lower priority)
            await self.queue_manager.enqueue_build(db, build.uuid, priority=build.retry_count)

            await db.commit()

            logger.warning(
                f"Build {build.uuid} requeued for retry {build.retry_count}/{self.MAX_RETRIES} "
                f"(retry after: {build.retry_after.isoformat()}, "
                f"last checkpoint: {build.last_checkpoint})"
            )

        except Exception as e:
            logger.error(f"Failed to retry build {build.uuid}: {e}")
            await db.rollback()
    
    async def _fail_build(self, db: AsyncSession, build: StrategyBuild, reason: str):
        """Mark a build as failed.
        
        Args:
            db: Database session
            build: StrategyBuild instance
            reason: Failure reason
        """
        try:
            build.status = "failed"
            build.phase = f"Failed: {reason}"
            build.completed_at = datetime.utcnow()
            build.assigned_worker_id = None
            
            # Remove from queue if present
            await self.queue_manager.remove_from_queue(db, build.uuid)
            
            await db.commit()
            
            logger.error(f"Build {build.uuid} marked as failed: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to mark build {build.uuid} as failed: {e}")
            await db.rollback()

