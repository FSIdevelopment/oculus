"""Worker health tracking endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List
import logging

from app.database import get_db
from app.services.worker_health import WorkerHealthManager

logger = logging.getLogger(__name__)

router = APIRouter()


class WorkerRegistration(BaseModel):
    """Worker registration request."""
    worker_id: str
    hostname: str
    capacity: int


class WorkerHeartbeat(BaseModel):
    """Worker heartbeat request."""
    worker_id: str
    active_job_ids: List[str]


@router.post("/api/worker/register")
async def register_worker(
    registration: WorkerRegistration,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a worker with the backend.
    
    - Creates or updates worker health record
    - Sets worker status to active
    - Returns worker information
    """
    try:
        manager = WorkerHealthManager()
        
        worker = await manager.register_worker(
            db=db,
            worker_id=registration.worker_id,
            hostname=registration.hostname,
            capacity=registration.capacity
        )
        
        return {
            "worker_id": worker.worker_id,
            "hostname": worker.hostname,
            "capacity": worker.capacity,
            "status": worker.status,
            "message": "Worker registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to register worker {registration.worker_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register worker: {str(e)}"
        )


@router.post("/api/worker/heartbeat")
async def worker_heartbeat(
    heartbeat: WorkerHeartbeat,
    db: AsyncSession = Depends(get_db)
):
    """
    Receive heartbeat from worker.
    
    - Updates worker's last heartbeat timestamp
    - Updates active jobs count
    - Keeps worker status as active
    """
    try:
        manager = WorkerHealthManager()
        
        await manager.update_heartbeat(
            db=db,
            worker_id=heartbeat.worker_id,
            active_job_ids=heartbeat.active_job_ids
        )
        
        return {
            "worker_id": heartbeat.worker_id,
            "active_jobs": len(heartbeat.active_job_ids),
            "message": "Heartbeat received"
        }
        
    except Exception as e:
        logger.error(f"Failed to process heartbeat from {heartbeat.worker_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process heartbeat: {str(e)}"
        )


@router.get("/api/worker/stats")
async def get_worker_stats(db: AsyncSession = Depends(get_db)):
    """
    Get worker statistics.

    Returns:
    - Total workers
    - Active workers
    - Dead workers
    - Total capacity
    - Active jobs
    - Available capacity
    - Utilization percentage
    """
    try:
        manager = WorkerHealthManager()
        stats = await manager.get_worker_stats(db)

        return stats

    except Exception as e:
        logger.error(f"Failed to get worker stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get worker stats: {str(e)}"
        )


@router.post("/api/worker/cleanup-dead")
async def cleanup_dead_workers(db: AsyncSession = Depends(get_db)):
    """
    Manually cleanup dead workers that have been dead for > 1 hour.

    This is useful for immediately removing old dead workers from the dashboard
    without waiting for the automatic cleanup scheduler.

    Returns:
    - Number of workers cleaned up
    """
    try:
        manager = WorkerHealthManager()
        cleaned_count = await manager.cleanup_dead_workers(db)

        return {
            "cleaned_count": cleaned_count,
            "message": f"Cleaned up {cleaned_count} dead worker(s)"
        }

    except Exception as e:
        logger.error(f"Failed to cleanup dead workers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup dead workers: {str(e)}"
        )

