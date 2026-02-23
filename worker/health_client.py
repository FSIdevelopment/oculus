"""
Worker health client for communicating with backend health tracking.

Sends heartbeats and registers worker with the backend.
"""

import asyncio
import logging
import socket
import uuid
from typing import List, Optional
import httpx

logger = logging.getLogger(__name__)


class WorkerHealthClient:
    """Client for worker health tracking with backend.

    Thread-safe job tracking using asyncio.Lock to prevent race conditions
    when multiple async tasks are adding/removing jobs concurrently.
    """

    def __init__(self, backend_url: str, worker_id: Optional[str] = None, capacity: int = 4):
        """Initialize health client.

        Args:
            backend_url: Backend API base URL (e.g., "http://localhost:8000")
            worker_id: Unique worker identifier (auto-generated if not provided)
            capacity: Maximum concurrent jobs this worker can handle
        """
        self.backend_url = backend_url.rstrip('/')
        self.worker_id = worker_id or self._generate_worker_id()
        self.capacity = capacity
        self.hostname = socket.gethostname()
        self.active_job_ids: List[str] = []
        self._job_lock = asyncio.Lock()  # Thread-safe access to active_job_ids

        logger.info(f"Worker health client initialized: {self.worker_id} @ {self.hostname}")
    
    def _generate_worker_id(self) -> str:
        """Generate a unique worker ID based on hostname and UUID."""
        hostname = socket.gethostname()
        unique_id = str(uuid.uuid4())[:8]
        return f"{hostname}-{unique_id}"
    
    async def register(self) -> bool:
        """Register worker with backend.
        
        Returns:
            True if registration successful, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.backend_url}/api/worker/register",
                    json={
                        "worker_id": self.worker_id,
                        "hostname": self.hostname,
                        "capacity": self.capacity
                    }
                )
                
                if response.status_code == 200:
                    logger.info(f"Worker {self.worker_id} registered successfully")
                    return True
                else:
                    logger.error(
                        f"Failed to register worker: {response.status_code} - {response.text}"
                    )
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to register worker: {e}")
            return False
    
    async def send_heartbeat(self) -> bool:
        """Send heartbeat to backend with current active jobs.

        Returns:
            True if heartbeat sent successfully, False otherwise
        """
        try:
            # Get a thread-safe snapshot of active jobs
            async with self._job_lock:
                active_jobs_snapshot = self.active_job_ids.copy()

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.backend_url}/api/worker/heartbeat",
                    json={
                        "worker_id": self.worker_id,
                        "active_job_ids": active_jobs_snapshot
                    }
                )

                if response.status_code == 200:
                    logger.debug(
                        f"Heartbeat sent: {self.worker_id} "
                        f"({len(active_jobs_snapshot)} active jobs)"
                    )
                    return True
                else:
                    logger.warning(
                        f"Failed to send heartbeat: {response.status_code} - {response.text}"
                    )
                    return False

        except Exception as e:
            logger.warning(f"Failed to send heartbeat: {e}")
            return False
    
    async def add_job(self, build_id: str):
        """Add a job to the active jobs list (thread-safe).

        Args:
            build_id: Build ID to add
        """
        async with self._job_lock:
            if build_id not in self.active_job_ids:
                self.active_job_ids.append(build_id)
                logger.info(f"✓ Job {build_id} checked in - Active jobs: {len(self.active_job_ids)}/{self.capacity}")
            else:
                logger.warning(f"Job {build_id} already in active jobs list")

    async def remove_job(self, build_id: str):
        """Remove a job from the active jobs list (thread-safe).

        Args:
            build_id: Build ID to remove
        """
        async with self._job_lock:
            if build_id in self.active_job_ids:
                self.active_job_ids.remove(build_id)
                logger.info(f"✓ Job {build_id} checked out - Active jobs: {len(self.active_job_ids)}/{self.capacity}")
            else:
                logger.warning(f"Job {build_id} not found in active jobs list")

    async def get_active_job_count(self) -> int:
        """Get the number of active jobs (thread-safe).

        Returns:
            Number of active jobs
        """
        async with self._job_lock:
            return len(self.active_job_ids)

    async def has_capacity(self) -> bool:
        """Check if worker has capacity for more jobs (thread-safe).

        Returns:
            True if worker can accept more jobs, False otherwise
        """
        async with self._job_lock:
            return len(self.active_job_ids) < self.capacity

    async def get_active_jobs_snapshot(self) -> List[str]:
        """Get a thread-safe snapshot of active job IDs.

        Returns:
            Copy of active job IDs list
        """
        async with self._job_lock:
            return self.active_job_ids.copy()

