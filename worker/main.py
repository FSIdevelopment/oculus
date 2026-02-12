"""
Local Training Worker - Main entry point.

Listens on Redis queue for training jobs and processes them using the ML pipeline.

Usage:
    python -m worker.main
    python -m worker.main --config /path/to/.env
"""

import argparse
import asyncio
import json
import logging
import signal
import sys

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import redis

from worker.config import config
from worker.job_processor import JobProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingWorker:
    """Redis-based training worker for ML jobs."""

    def __init__(self, redis_client: redis.Redis, concurrency: int = 2):
        """Initialize worker.

        Args:
            redis_client: Redis client instance
            concurrency: Number of concurrent jobs
        """
        self.redis = redis_client
        self.concurrency = concurrency
        self.processor = JobProcessor(redis_client)
        self.executor = ThreadPoolExecutor(max_workers=concurrency)
        self.running = False

    async def start(self):
        """Start the worker and listen for jobs."""
        self.running = True
        logger.info(f"Starting training worker (concurrency={self.concurrency})")
        logger.info(f"Listening on queue: {config.job_queue}")

        try:
            while self.running:
                # Block and wait for job ID from queue (timeout=1 to allow graceful shutdown)
                # Wrap blpop in try/except to handle Redis connection errors gracefully
                try:
                    job_data = await asyncio.to_thread(self.redis.blpop, config.job_queue, 1)
                except Exception as blpop_error:
                    logger.warning(f"⚠️  blpop error (will retry): {blpop_error}")
                    await asyncio.sleep(1)
                    continue

                if job_data is None:
                    continue

                # Unpack job ID from Redis list
                _, job_id = job_data
                try:
                    # Fetch full job payload from Redis
                    job_json = self.redis.get(job_id)
                    if job_json is None:
                        logger.error(f"Job data not found for ID: {job_id}")
                        continue

                    job = json.loads(job_json)
                    logger.info(f"Received job: {job_id}")

                    # Process job asynchronously
                    asyncio.create_task(self._process_and_report(job, job_id))

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse job JSON: {e}")

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.exception(f"Worker error: {e}")
        finally:
            self.stop()

    async def _process_and_report(self, job: dict, job_id: str):
        """Process a job and report results back to Redis.

        Args:
            job: Job configuration dictionary
            job_id: Job ID from Redis queue
        """
        try:
            # Process job
            result = await self.processor.process_job(job, job_id)

            # Store result in Redis with 24h TTL
            result_key = f"{config.result_key_prefix}:{job_id}"
            self.redis.set(result_key, json.dumps(result), ex=86400)
            logger.info(f"[{job_id}] Result stored at {result_key}")

        except Exception as e:
            logger.exception(f"[{job_id}] Failed to process job: {e}")
            error_result = {
                "build_id": job.get('build_id', 'unknown'),
                "status": "error",
                "error": str(e)
            }
            result_key = f"{config.result_key_prefix}:{job_id}"
            self.redis.set(result_key, json.dumps(error_result), ex=86400)

    def stop(self):
        """Stop the worker gracefully."""
        logger.info("Stopping worker...")
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Worker stopped")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Local Training Worker")
    parser.add_argument("--config", type=str, help="Path to .env config file")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Number of concurrent jobs")
    args = parser.parse_args()

    # Load config if specified
    if args.config:
        import os
        if os.path.exists(args.config):
            with open(args.config) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

    # Connect to Redis using URL-based connection with timeout settings
    try:
        redis_client = redis.Redis.from_url(
            config.redis_url,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=10,
        )
        redis_client.ping()
        logger.info(f"Connected to Redis: {config.redis_url}")
    except (redis.ConnectionError, redis.exceptions.TimeoutError) as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)

    # Create and start worker
    concurrency = args.concurrency or config.worker_concurrency
    worker = TrainingWorker(redis_client, concurrency=concurrency)

    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        worker.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run worker
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())

