"""Monitoring service for queue, workers, and recovery operations."""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import redis.asyncio as redis
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.strategy_build import StrategyBuild
from app.models.worker_health import WorkerHealth
from app.services.build_queue import BuildQueueManager
from app.services.worker_health import WorkerHealthManager

logger = logging.getLogger(__name__)


class MonitoringService:
    """Provides comprehensive metrics for monitoring the build system."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize monitoring service.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.queue_manager = BuildQueueManager(redis_client)
        self.worker_manager = WorkerHealthManager()
    
    async def get_system_health(self, db: AsyncSession) -> Dict[str, Any]:
        """Get overall system health metrics.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with system health metrics
        """
        try:
            # Get queue metrics
            queue_status = await self.queue_manager.get_queue_status(db)
            
            # Get worker metrics
            worker_stats = await self.worker_manager.get_worker_stats(db)
            
            # Get build metrics
            build_metrics = await self._get_build_metrics(db)
            
            # Get recovery metrics
            recovery_metrics = await self._get_recovery_metrics(db)
            
            # Calculate health score (0-100)
            health_score = self._calculate_health_score(
                queue_status, worker_stats, build_metrics
            )
            
            return {
                "health_score": health_score,
                "status": self._get_status_from_score(health_score),
                "timestamp": datetime.utcnow().isoformat(),
                "queue": {
                    "length": queue_status["queue_length"],
                    "active_builds": queue_status["active_builds"],
                    "avg_wait_time_minutes": await self._get_avg_wait_time(db),
                },
                "workers": {
                    "total": worker_stats["total_workers"],
                    "active": worker_stats["active_workers"],
                    "dead": worker_stats["dead_workers"],
                    "utilization_percent": worker_stats["utilization_percent"],
                    "available_capacity": worker_stats["available_capacity"],
                },
                "builds": build_metrics,
                "recovery": recovery_metrics,
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "health_score": 0,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    async def _get_build_metrics(self, db: AsyncSession) -> Dict[str, Any]:
        """Get build-related metrics.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with build metrics
        """
        try:
            # Count builds by status in last 24 hours
            since = datetime.utcnow() - timedelta(hours=24)
            
            result = await db.execute(
                select(
                    StrategyBuild.status,
                    func.count(StrategyBuild.uuid).label('count')
                ).where(
                    StrategyBuild.started_at >= since
                ).group_by(StrategyBuild.status)
            )
            
            status_counts = {row.status: row.count for row in result}
            
            # Calculate success rate
            total = sum(status_counts.values())
            completed = status_counts.get('complete', 0)
            failed = status_counts.get('failed', 0)
            success_rate = (completed / total * 100) if total > 0 else 0
            
            # Get average build duration for completed builds
            result = await db.execute(
                select(
                    func.avg(
                        func.extract('epoch', StrategyBuild.completed_at - StrategyBuild.started_at)
                    ).label('avg_duration')
                ).where(
                    and_(
                        StrategyBuild.status == 'complete',
                        StrategyBuild.completed_at.isnot(None),
                        StrategyBuild.started_at >= since
                    )
                )
            )
            avg_duration_seconds = result.scalar() or 0
            
            return {
                "last_24h": status_counts,
                "success_rate_percent": round(success_rate, 2),
                "avg_duration_minutes": round(avg_duration_seconds / 60, 2),
                "total_builds_24h": total,
            }
            
        except Exception as e:
            logger.error(f"Failed to get build metrics: {e}")
            return {}
    
    async def _get_recovery_metrics(self, db: AsyncSession) -> Dict[str, Any]:
        """Get recovery-related metrics.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with recovery metrics
        """
        try:
            # Count builds with retries in last 24 hours
            since = datetime.utcnow() - timedelta(hours=24)
            
            result = await db.execute(
                select(
                    func.count(StrategyBuild.uuid).label('total'),
                    func.sum(StrategyBuild.retry_count).label('total_retries'),
                    func.max(StrategyBuild.retry_count).label('max_retries'),
                ).where(
                    and_(
                        StrategyBuild.started_at >= since,
                        StrategyBuild.retry_count > 0
                    )
                )
            )
            row = result.first()
            
            return {
                "builds_with_retries_24h": row.total or 0,
                "total_retries_24h": row.total_retries or 0,
                "max_retry_count_24h": row.max_retries or 0,
            }
            
        except Exception as e:
            logger.error(f"Failed to get recovery metrics: {e}")
            return {}
    
    async def _get_avg_wait_time(self, db: AsyncSession) -> float:
        """Calculate average wait time for queued builds.
        
        Args:
            db: Database session
            
        Returns:
            Average wait time in minutes
        """
        try:
            result = await db.execute(
                select(
                    func.avg(
                        func.extract('epoch', datetime.utcnow() - StrategyBuild.started_at)
                    ).label('avg_wait')
                ).where(
                    StrategyBuild.status == 'queued'
                )
            )
            avg_wait_seconds = result.scalar() or 0
            return round(avg_wait_seconds / 60, 2)
            
        except Exception as e:
            logger.error(f"Failed to get average wait time: {e}")
            return 0.0
    
    def _calculate_health_score(
        self,
        queue_status: Dict[str, Any],
        worker_stats: Dict[str, Any],
        build_metrics: Dict[str, Any]
    ) -> int:
        """Calculate overall system health score (0-100).
        
        Args:
            queue_status: Queue status metrics
            worker_stats: Worker statistics
            build_metrics: Build metrics
            
        Returns:
            Health score from 0 to 100
        """
        score = 100
        
        # Deduct points for queue length
        queue_length = queue_status.get("queue_length", 0)
        if queue_length > 10:
            score -= min(30, queue_length * 2)
        
        # Deduct points for dead workers
        dead_workers = worker_stats.get("dead_workers", 0)
        if dead_workers > 0:
            score -= min(20, dead_workers * 10)
        
        # Deduct points for low success rate
        success_rate = build_metrics.get("success_rate_percent", 100)
        if success_rate < 90:
            score -= (90 - success_rate)
        
        # Deduct points for high utilization
        utilization = worker_stats.get("utilization_percent", 0)
        if utilization > 90:
            score -= 10
        
        return max(0, score)
    
    def _get_status_from_score(self, score: int) -> str:
        """Get status string from health score.

        Args:
            score: Health score (0-100)

        Returns:
            Status string
        """
        if score >= 90:
            return "healthy"
        elif score >= 70:
            return "degraded"
        elif score >= 50:
            return "warning"
        else:
            return "critical"

    async def get_scaling_recommendations(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get recommendations for worker auto-scaling.

        Args:
            db: Database session

        Returns:
            Dictionary with scaling recommendations
        """
        try:
            # Get current metrics
            queue_status = await self.queue_manager.get_queue_status(db)
            worker_stats = await self.worker_manager.get_worker_stats(db)

            queue_length = queue_status.get("queue_length", 0)
            active_builds = queue_status.get("active_builds", 0)
            active_workers = worker_stats.get("active_workers", 0)
            total_capacity = worker_stats.get("total_capacity", 0)
            available_capacity = worker_stats.get("available_capacity", 0)
            utilization = worker_stats.get("utilization_percent", 0)

            # Calculate average wait time
            avg_wait_time = await self._get_avg_wait_time(db)

            # Determine scaling recommendation
            recommendation = "maintain"
            reason = "System is operating normally"
            suggested_workers = active_workers

            # Scale up conditions
            if queue_length > 10 and utilization > 80:
                recommendation = "scale_up"
                reason = f"High queue length ({queue_length}) and high utilization ({utilization}%)"
                # Suggest adding workers to handle queue
                suggested_workers = active_workers + max(1, queue_length // 5)
            elif avg_wait_time > 10 and utilization > 70:
                recommendation = "scale_up"
                reason = f"Long average wait time ({avg_wait_time:.1f} min) and high utilization ({utilization}%)"
                suggested_workers = active_workers + 1
            elif available_capacity == 0 and queue_length > 5:
                recommendation = "scale_up"
                reason = f"No available capacity and {queue_length} builds queued"
                suggested_workers = active_workers + 1

            # Scale down conditions
            elif utilization < 20 and active_workers > 1 and queue_length == 0:
                recommendation = "scale_down"
                reason = f"Low utilization ({utilization}%) with no queued builds"
                suggested_workers = max(1, active_workers - 1)
            elif active_builds == 0 and queue_length == 0 and active_workers > 1:
                recommendation = "scale_down"
                reason = "No active or queued builds"
                suggested_workers = 1  # Keep at least one worker

            return {
                "recommendation": recommendation,
                "reason": reason,
                "current_workers": active_workers,
                "suggested_workers": suggested_workers,
                "metrics": {
                    "queue_length": queue_length,
                    "active_builds": active_builds,
                    "utilization_percent": utilization,
                    "available_capacity": available_capacity,
                    "avg_wait_time_minutes": avg_wait_time,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get scaling recommendations: {e}")
            return {
                "recommendation": "error",
                "reason": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

