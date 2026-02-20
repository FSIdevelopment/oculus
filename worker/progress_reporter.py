"""
Progress reporter for publishing real-time training updates to Redis pub/sub.
"""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import redis

from worker.config import config

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Publishes training progress updates to Redis pub/sub."""

    def __init__(self, job_id: str, redis_client: redis.Redis,
                 channel_id: Optional[str] = None):
        """Initialize progress reporter.

        Args:
            job_id: Unique job identifier (used in message payloads).
            redis_client: Redis client instance.
            channel_id: Optional channel override. When provided the reporter
                publishes to ``{prefix}:{channel_id}`` instead of
                ``{prefix}:{job_id}``.  Pass ``"build:{build_uuid}"`` so the
                worker's messages land on the same channel that the backend
                WebSocket subscribes to (the full job_id contains an extra
                ``:iter:{uuid}`` suffix that would otherwise cause a mismatch).
        """
        self.job_id = job_id
        self.redis = redis_client
        effective_id = channel_id if channel_id is not None else job_id
        self.channel = f"{config.progress_channel_prefix}:{effective_id}"

    def report_phase(self, phase: str, status: str = "in_progress",
                     details: Optional[Dict[str, Any]] = None,
                     message: Optional[str] = None):
        """Report a training phase update.

        Args:
            phase: Phase name (e.g., "data_fetch", "feature_engineering", "training")
            status: Phase status ("in_progress", "complete", "error")
            details: Optional additional details dict.
            message: Optional human-readable description shown in the UI Design
                Thinking section.  Only ``in_progress`` events typically need
                this; ``complete`` events can omit it.
        """
        payload: Dict[str, Any] = {
            "job_id": self.job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase,
            "status": status,
            "details": details or {},
        }
        if message is not None:
            payload["message"] = message

        try:
            self.redis.publish(self.channel, json.dumps(payload))
            logger.info(f"[{self.job_id}] {phase}: {status}")
        except Exception as e:
            logger.error(f"Failed to publish progress: {e}")

    def report_error(self, phase: str, error: str, details: Optional[Dict] = None):
        """Report an error during training.

        Args:
            phase: Phase where error occurred
            error: Error message
            details: Optional error details
        """
        message = {
            "job_id": self.job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase,
            "status": "error",
            "error": error,
            "details": details or {}
        }

        try:
            self.redis.publish(self.channel, json.dumps(message))
            logger.error(f"[{self.job_id}] {phase}: ERROR - {error}")
        except Exception as e:
            logger.error(f"Failed to publish error: {e}")

    def report_metrics(self, metrics: Dict[str, Any]):
        """Report model metrics.

        Args:
            metrics: Dictionary of model metrics
        """
        message = {
            "job_id": self.job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "phase": "metrics",
            "status": "complete",
            "metrics": metrics
        }

        try:
            self.redis.publish(self.channel, json.dumps(message))
            logger.info(f"[{self.job_id}] Metrics reported")
        except Exception as e:
            logger.error(f"Failed to publish metrics: {e}")

