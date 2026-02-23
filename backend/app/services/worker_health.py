"""Worker health tracking and management service."""
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.worker_health import WorkerHealth
from app.models.strategy_build import StrategyBuild

logger = logging.getLogger(__name__)


class WorkerHealthManager:
    """Manages worker health tracking and status."""
    
    HEARTBEAT_TIMEOUT = timedelta(seconds=90)  # Mark worker dead after 90s without heartbeat
    
    async def register_worker(
        self,
        db: AsyncSession,
        worker_id: str,
        hostname: str,
        capacity: int
    ) -> WorkerHealth:
        """Register or update worker.
        
        Args:
            db: Database session
            worker_id: Unique worker identifier
            hostname: Worker hostname
            capacity: Maximum concurrent jobs
            
        Returns:
            WorkerHealth instance
        """
        try:
            # Check if worker exists
            result = await db.execute(
                select(WorkerHealth).where(WorkerHealth.worker_id == worker_id)
            )
            worker = result.scalar_one_or_none()
            
            if worker:
                # Update existing worker
                worker.hostname = hostname
                worker.capacity = capacity
                worker.last_heartbeat = datetime.utcnow()
                worker.status = "active"
                worker.updated_at = datetime.utcnow()
            else:
                # Create new worker
                worker = WorkerHealth(
                    worker_id=worker_id,
                    hostname=hostname,
                    capacity=capacity,
                    active_jobs=0,
                    last_heartbeat=datetime.utcnow(),
                    status="active"
                )
                db.add(worker)
            
            await db.commit()
            await db.refresh(worker)
            
            logger.info(f"Worker {worker_id} registered: {hostname} (capacity: {capacity})")
            return worker
            
        except Exception as e:
            logger.error(f"Failed to register worker {worker_id}: {e}")
            await db.rollback()
            raise
    
    async def update_heartbeat(
        self,
        db: AsyncSession,
        worker_id: str,
        active_job_ids: List[str]
    ):
        """Update worker heartbeat and active jobs.
        
        Args:
            db: Database session
            worker_id: Worker identifier
            active_job_ids: List of currently active build IDs
        """
        try:
            result = await db.execute(
                select(WorkerHealth).where(WorkerHealth.worker_id == worker_id)
            )
            worker = result.scalar_one_or_none()
            
            if not worker:
                logger.warning(f"Worker {worker_id} not found for heartbeat update")
                return
            
            worker.last_heartbeat = datetime.utcnow()
            worker.active_jobs = len(active_job_ids)
            worker.status = "active"
            worker.updated_at = datetime.utcnow()
            
            await db.commit()
            
            logger.debug(f"Worker {worker_id} heartbeat updated: {len(active_job_ids)} active jobs")
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for worker {worker_id}: {e}")
            await db.rollback()
    
    async def mark_dead_workers(self, db: AsyncSession) -> List[str]:
        """Mark workers as dead if no heartbeat within timeout.
        
        Args:
            db: Database session
            
        Returns:
            List of worker IDs marked as dead
        """
        try:
            threshold = datetime.utcnow() - self.HEARTBEAT_TIMEOUT
            
            result = await db.execute(
                select(WorkerHealth).where(
                    and_(
                        WorkerHealth.status == "active",
                        WorkerHealth.last_heartbeat < threshold
                    )
                )
            )
            dead_workers = result.scalars().all()
            
            dead_worker_ids = []
            for worker in dead_workers:
                worker.status = "dead"
                worker.updated_at = datetime.utcnow()
                dead_worker_ids.append(worker.worker_id)
                logger.warning(
                    f"Worker {worker.worker_id} marked as dead "
                    f"(last heartbeat: {worker.last_heartbeat})"
                )
            
            if dead_worker_ids:
                await db.commit()
            
            return dead_worker_ids
            
        except Exception as e:
            logger.error(f"Failed to mark dead workers: {e}")
            await db.rollback()
            return []
    
    async def get_active_workers(self, db: AsyncSession) -> List[WorkerHealth]:
        """Get all active workers.

        Args:
            db: Database session

        Returns:
            List of active WorkerHealth instances
        """
        try:
            result = await db.execute(
                select(WorkerHealth).where(WorkerHealth.status == "active")
            )
            workers = result.scalars().all()
            return list(workers)

        except Exception as e:
            logger.error(f"Failed to get active workers: {e}")
            return []

    async def get_available_worker(self, db: AsyncSession) -> Optional[WorkerHealth]:
        """Get an available worker with capacity.

        Args:
            db: Database session

        Returns:
            WorkerHealth instance if available, None otherwise
        """
        try:
            result = await db.execute(
                select(WorkerHealth).where(WorkerHealth.status == "active")
            )
            workers = result.scalars().all()

            # Find worker with available capacity
            for worker in workers:
                if worker.active_jobs < worker.capacity:
                    return worker

            return None

        except Exception as e:
            logger.error(f"Failed to get available worker: {e}")
            return None

    async def reassign_builds_from_dead_worker(
        self,
        db: AsyncSession,
        worker_id: str
    ) -> List[str]:
        """Reassign builds from a dead worker back to queue.

        Args:
            db: Database session
            worker_id: Dead worker identifier

        Returns:
            List of build IDs that were reassigned
        """
        try:
            # Find all builds assigned to this worker
            result = await db.execute(
                select(StrategyBuild).where(
                    and_(
                        StrategyBuild.assigned_worker_id == worker_id,
                        StrategyBuild.status.in_([
                            "building", "designing", "training",
                            "extracting_rules", "optimizing", "building_docker"
                        ])
                    )
                )
            )
            builds = result.scalars().all()

            reassigned_ids = []
            for build in builds:
                # Clear worker assignment
                build.assigned_worker_id = None
                build.last_heartbeat = None

                # Mark for requeue (will be picked up by recovery service)
                logger.warning(
                    f"Build {build.uuid} reassigned from dead worker {worker_id} "
                    f"(status: {build.status}, phase: {build.phase})"
                )
                reassigned_ids.append(build.uuid)

            if reassigned_ids:
                await db.commit()

            return reassigned_ids

        except Exception as e:
            logger.error(f"Failed to reassign builds from worker {worker_id}: {e}")
            await db.rollback()
            return []

    async def get_worker_stats(self, db: AsyncSession) -> dict:
        """Get worker statistics for monitoring.

        Args:
            db: Database session

        Returns:
            Dictionary with worker statistics
        """
        try:
            result = await db.execute(select(WorkerHealth))
            workers = result.scalars().all()

            active_workers = [w for w in workers if w.status == "active"]
            dead_workers = [w for w in workers if w.status == "dead"]

            total_capacity = sum(w.capacity for w in active_workers)
            total_active_jobs = sum(w.active_jobs for w in active_workers)

            return {
                "total_workers": len(workers),
                "active_workers": len(active_workers),
                "dead_workers": len(dead_workers),
                "total_capacity": total_capacity,
                "total_active_jobs": total_active_jobs,
                "available_capacity": max(0, total_capacity - total_active_jobs),
                "utilization": (total_active_jobs / total_capacity * 100) if total_capacity > 0 else 0
            }

        except Exception as e:
            logger.error(f"Failed to get worker stats: {e}")
            return {
                "total_workers": 0,
                "active_workers": 0,
                "dead_workers": 0,
                "total_capacity": 0,
                "total_active_jobs": 0,
                "available_capacity": 0,
                "utilization": 0
            }

