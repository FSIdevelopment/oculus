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
                # Block and wait for job (timeout=1 to allow graceful shutdown)
                job_data = self.redis.blpop(config.job_queue, timeout=1)

                if job_data is None:
                    continue

                # Unpack job from Redis list
                _, job_json = job_data
                try:
                    job = json.loads(job_json)
                    job_id = job.get('job_id', 'unknown')
                    logger.info(f"Received job: {job_id}")

                    # Process job asynchronously
                    asyncio.create_task(self._process_and_report(job))

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse job JSON: {e}")

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.exception(f"Worker error: {e}")
        finally:
            self.stop()

    async def _process_and_report(self, job: dict):
        """Process a job and report results back to Redis.

        Args:
            job: Job configuration dictionary
        """
        job_id = job.get('job_id', 'unknown')
        try:
            # Process job
            result = await self.processor.process_job(job)

            # Publish result to Redis
            result_channel = f"{config.result_channel}:{job_id}"
            self.redis.publish(result_channel, json.dumps(result))
            logger.info(f"[{job_id}] Result published to {result_channel}")

        except Exception as e:
            logger.exception(f"[{job_id}] Failed to process job: {e}")
            error_result = {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }
            result_channel = f"{config.result_channel}:{job_id}"
            self.redis.publish(result_channel, json.dumps(error_result))

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

    # Connect to Redis
    try:
        redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            decode_responses=True
        )
        redis_client.ping()
        logger.info(f"Connected to Redis: {config.redis_host}:{config.redis_port}")
    except redis.ConnectionError as e:
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

