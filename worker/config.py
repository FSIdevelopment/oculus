"""
Worker configuration management.

Loads configuration from .env file and environment variables.
"""

import os
from pathlib import Path
from typing import Optional


class WorkerConfig:
    """Worker configuration."""

    def __init__(self):
        """Initialize config from environment."""
        self._load_env()

    def _load_env(self):
        """Load .env file if it exists."""
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

    @property
    def redis_url(self) -> str:
        """Redis connection URL.

        Checks WORKER_REDIS_URL first (worker-specific override),
        then REDIS_URL, defaulting to localhost.
        The worker runs locally alongside the Redis container,
        so it should connect via localhost, not through the ngrok tunnel.
        """
        return os.getenv(
            'WORKER_REDIS_URL',
            os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        )

    @property
    def redis_host(self) -> str:
        """Redis host."""
        return os.getenv('REDIS_HOST', 'localhost')

    @property
    def redis_port(self) -> int:
        """Redis port."""
        return int(os.getenv('REDIS_PORT', '6379'))

    @property
    def redis_db(self) -> int:
        """Redis database number."""
        return int(os.getenv('REDIS_DB', '0'))

    @property
    def job_queue(self) -> str:
        """Redis queue name for Oculus training jobs (namespaced to avoid collision with other workers)."""
        return os.getenv('JOB_QUEUE', 'oculus:training_queue')

    @property
    def expected_job_source(self) -> str:
        """Expected source field in job payloads. Jobs without this source are skipped."""
        return os.getenv('EXPECTED_JOB_SOURCE', 'oculus')

    @property
    def result_key_prefix(self) -> str:
        """Redis key prefix for storing results."""
        return os.getenv('RESULT_KEY_PREFIX', 'result')

    @property
    def progress_channel_prefix(self) -> str:
        """Redis channel prefix for progress updates."""
        return os.getenv('PROGRESS_CHANNEL_PREFIX', 'oculus:training:progress')

    @property
    def worker_concurrency(self) -> int:
        """Number of concurrent jobs to process."""
        return int(os.getenv('WORKER_CONCURRENCY', '2'))

    @property
    def gpu_enabled(self) -> bool:
        """Enable GPU acceleration (MPS for Apple Silicon)."""
        return os.getenv('GPU_ENABLED', 'true').lower() == 'true'

    @property
    def polygon_api_key(self) -> Optional[str]:
        """Polygon API key for market data."""
        return os.getenv('POLYGON_API_KEY')

    @property
    def log_level(self) -> str:
        """Logging level."""
        return os.getenv('LOG_LEVEL', 'INFO')

    @property
    def timeout_seconds(self) -> int:
        """Job timeout in seconds."""
        return int(os.getenv('JOB_TIMEOUT_SECONDS', '3600'))

    @property
    def backend_url(self) -> str:
        """Backend API URL for health tracking."""
        return os.getenv('BACKEND_URL', 'http://localhost:8000')


# Global config instance
config = WorkerConfig()

