"""Build queue management service with Redis persistence."""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import redis.asyncio as redis
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.strategy_build import StrategyBuild
from app.models.worker_health import WorkerHealth

logger = logging.getLogger(__name__)


class BuildQueueManager:
    """Manages build queue with Redis persistence and worker assignment."""
    
    QUEUE_KEY = "oculus:build_queue"
    BUILD_LOCK_PREFIX = "oculus:build:lock"
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize queue manager.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
    
    async def enqueue_build(self, db: AsyncSession, build_id: str, priority: int = 0) -> int:
        """Add build to queue with priority.

        Args:
            db: Database session
            build_id: Build UUID
            priority: Priority level (higher = more important, default 0)

        Returns:
            Queue position (0-based)
        """
        # Get build from database
        result = await db.execute(
            select(StrategyBuild).where(StrategyBuild.uuid == build_id)
        )
        build = result.scalar_one_or_none()
        if not build:
            raise ValueError(f"Build {build_id} not found")

        try:
            # Try to add to Redis sorted set (score = -priority for high-priority-first)
            # Use negative timestamp as tiebreaker for same priority
            score = -priority * 1000000 - datetime.utcnow().timestamp()
            await self.redis.zadd(self.QUEUE_KEY, {build_id: score})

            # Update queue positions for all builds
            await self._update_queue_positions(db)

            # Get position
            position = await self.redis.zrank(self.QUEUE_KEY, build_id)

            logger.info(f"Build {build_id} enqueued at position {position} with priority {priority}")
            return position or 0

        except Exception as e:
            # Redis failed - fall back to database-only mode
            logger.warning(f"Redis unavailable for build {build_id}, using database fallback: {e}")

            # Get current max queue position from database
            result = await db.execute(
                select(StrategyBuild.queue_position).where(
                    StrategyBuild.status == "queued"
                ).order_by(StrategyBuild.queue_position.desc()).limit(1)
            )
            max_position = result.scalar_one_or_none()

            # Assign next position
            position = (max_position + 1) if max_position is not None else 0
            build.queue_position = position
            await db.commit()

            logger.info(f"Build {build_id} enqueued at position {position} (database fallback)")
            return position
    
    async def dequeue_build(self, db: AsyncSession, worker_id: str) -> Optional[str]:
        """Dequeue next build and assign to worker.
        
        Args:
            db: Database session
            worker_id: Worker ID to assign build to
            
        Returns:
            Build ID if available, None otherwise
        """
        try:
            # Get highest priority build (lowest score)
            result = await self.redis.zpopmin(self.QUEUE_KEY, count=1)
            if not result:
                return None
            
            build_id, _ = result[0]
            
            # Assign to worker in database
            db_result = await db.execute(
                select(StrategyBuild).where(StrategyBuild.uuid == build_id)
            )
            build = db_result.scalar_one_or_none()
            if build:
                build.assigned_worker_id = worker_id
                build.last_heartbeat = datetime.utcnow()
                build.queue_position = None
                await db.commit()
            
            # Update remaining queue positions
            await self._update_queue_positions(db)
            
            logger.info(f"Build {build_id} dequeued and assigned to worker {worker_id}")
            return build_id
            
        except Exception as e:
            logger.error(f"Failed to dequeue build for worker {worker_id}: {e}")
            raise
    
    async def remove_from_queue(self, db: AsyncSession, build_id: str):
        """Remove build from queue.
        
        Args:
            db: Database session
            build_id: Build UUID
        """
        try:
            await self.redis.zrem(self.QUEUE_KEY, build_id)
            
            # Update queue positions
            await self._update_queue_positions(db)
            
            # Clear queue position in database
            result = await db.execute(
                select(StrategyBuild).where(StrategyBuild.uuid == build_id)
            )
            build = result.scalar_one_or_none()
            if build:
                build.queue_position = None
                await db.commit()
            
            logger.info(f"Build {build_id} removed from queue")
            
        except Exception as e:
            logger.error(f"Failed to remove build {build_id} from queue: {e}")
            raise
    
    async def get_queue_position(self, build_id: str) -> Optional[int]:
        """Get build's position in queue.

        Args:
            build_id: Build UUID

        Returns:
            Queue position (1-based) or None if not in queue
        """
        try:
            position = await self.redis.zrank(self.QUEUE_KEY, build_id)
            # Convert from 0-based to 1-based for better UX
            return position + 1 if position is not None else None
        except Exception as e:
            logger.error(f"Failed to get queue position for build {build_id}: {e}")
            return None

    async def get_queue_length(self) -> int:
        """Get total number of builds in queue.

        Returns:
            Queue length
        """
        try:
            return await self.redis.zcard(self.QUEUE_KEY)
        except Exception as e:
            logger.error(f"Failed to get queue length: {e}")
            return 0

    async def get_queue_status(self, db: AsyncSession) -> Dict[str, Any]:
        """Get comprehensive queue status for admin dashboard.

        UI Structure:
        - Build Queue: Builds waiting to start (status = 'queued')
        - Active Builds: ALL running builds (building, designing, waiting_for_worker, training, extracting_rules, optimizing, building_docker)
        - Training Queue: Subset of Active - waiting for worker (status = 'waiting_for_worker')

        Args:
            db: Database session

        Returns:
            Dictionary with queue statistics and build details
        """
        try:
            # Get all builds in queue from Redis (waiting to start)
            queue_items = await self.redis.zrange(self.QUEUE_KEY, 0, -1, withscores=True)
            build_ids = [item[0] for item in queue_items]

            # Get build details from database for queued builds (only status = 'queued')
            queued_builds = []
            if build_ids:
                result = await db.execute(
                    select(StrategyBuild).where(
                        StrategyBuild.uuid.in_(build_ids),
                        StrategyBuild.status == "queued"
                    )
                )
                queued_builds = result.scalars().all()

            # Get ALL active builds (any running status)
            active_result = await db.execute(
                select(StrategyBuild).where(
                    StrategyBuild.status.in_([
                        "building", "designing", "building_docker",
                        "waiting_for_worker",
                        "training", "extracting_rules", "optimizing"
                    ])
                )
            )
            active_builds = active_result.scalars().all()

            # Training queue: subset of active builds waiting for worker
            training_queue_builds = [b for b in active_builds if b.status == "waiting_for_worker"]

            # Get worker health
            worker_result = await db.execute(select(WorkerHealth))
            workers = worker_result.scalars().all()

            # Calculate statistics
            total_capacity = sum(w.capacity for w in workers if w.status == "active")
            active_jobs = sum(w.active_jobs for w in workers if w.status == "active")
            available_capacity = total_capacity - active_jobs

            return {
                # Build queue (waiting to start - status = 'queued')
                "queue_length": len(queued_builds),
                "queued_builds": [
                    {
                        "uuid": b.uuid,
                        "strategy_id": b.strategy_id,
                        "status": b.status,
                        "queue_position": b.queue_position,
                        "retry_count": b.retry_count,
                        "started_at": b.started_at.isoformat() if b.started_at else None,
                    }
                    for b in queued_builds
                ],

                # Active builds (ALL running builds)
                "active_builds_count": len(active_builds),
                "active_builds": [
                    {
                        "uuid": b.uuid,
                        "strategy_id": b.strategy_id,
                        "status": b.status,
                        "phase": b.phase,
                        "assigned_worker_id": getattr(b, 'assigned_worker_id', None),
                        "last_heartbeat": b.last_heartbeat.isoformat() if b.last_heartbeat else None,
                        "iteration_count": b.iteration_count,
                        "max_iterations": b.max_iterations,
                        "started_at": b.started_at.isoformat() if b.started_at else None,
                    }
                    for b in active_builds
                ],

                # Training queue (subset of active - waiting for worker)
                "training_queue_length": len(training_queue_builds),
                "training_queue_builds": [
                    {
                        "uuid": b.uuid,
                        "strategy_id": b.strategy_id,
                        "status": b.status,
                        "phase": b.phase,
                        "iteration_count": b.iteration_count,
                        "max_iterations": b.max_iterations,
                        "started_at": b.started_at.isoformat() if b.started_at else None,
                    }
                    for b in training_queue_builds
                ],

                # Worker capacity
                "workers": [
                    {
                        "worker_id": w.worker_id,
                        "hostname": w.hostname,
                        "status": w.status,
                        "capacity": w.capacity,
                        "active_jobs": w.active_jobs,
                        "last_heartbeat": w.last_heartbeat.isoformat() if w.last_heartbeat else None,
                    }
                    for w in workers
                ],
                "total_workers": len(workers),
                "active_workers": len([w for w in workers if w.status == "active"]),
                "total_capacity": total_capacity,
                "available_capacity": max(0, available_capacity),
            }

        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            raise

    async def _update_queue_positions(self, db: AsyncSession):
        """Update queue_position field for all builds in queue.

        Args:
            db: Database session
        """
        try:
            # Get all builds in queue with their positions
            queue_items = await self.redis.zrange(self.QUEUE_KEY, 0, -1)

            if not queue_items:
                return

            # Update positions in database
            for position, build_id in enumerate(queue_items):
                result = await db.execute(
                    select(StrategyBuild).where(StrategyBuild.uuid == build_id)
                )
                build = result.scalar_one_or_none()
                if build:
                    build.queue_position = position

            await db.commit()

        except Exception as e:
            logger.error(f"Failed to update queue positions: {e}")
            # Don't raise - this is a background operation

    async def restore_queue_from_db(self, db: AsyncSession):
        """Restore Redis queue from PostgreSQL (fallback when Redis is down/cleared).

        This method rebuilds the Redis queue from the database queue_position field.
        Useful for disaster recovery when Redis data is lost.

        Args:
            db: Database session
        """
        try:
            # Get all queued builds ordered by queue_position
            result = await db.execute(
                select(StrategyBuild).where(
                    and_(
                        StrategyBuild.status == "queued",
                        StrategyBuild.queue_position.isnot(None)
                    )
                ).order_by(StrategyBuild.queue_position)
            )
            queued_builds = result.scalars().all()

            if not queued_builds:
                logger.info("No queued builds found in database to restore")
                return

            # Clear existing queue in Redis
            await self.redis.delete(self.QUEUE_KEY)

            # Restore builds to Redis queue
            restored_count = 0
            for build in queued_builds:
                # Use queue_position as priority (lower position = higher priority)
                # Negative to ensure lower positions come first
                score = -build.queue_position * 1000000 - build.started_at.timestamp()
                await self.redis.zadd(self.QUEUE_KEY, {build.uuid: score})
                restored_count += 1

            logger.info(f"Restored {restored_count} builds to Redis queue from PostgreSQL")

        except Exception as e:
            logger.error(f"Failed to restore queue from database: {e}")
            raise

