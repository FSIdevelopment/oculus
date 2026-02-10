"""
Integration test for the training worker.

Sends a test job to Redis and verifies results come back.

Usage:
    python -m worker.test_integration
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

import redis

from worker.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_worker():
    """Test the training worker with a sample job."""
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
        return False

    # Create test job
    job_id = str(uuid.uuid4())[:8]
    job = {
        "job_id": job_id,
        "symbols": ["NVDA", "AMD"],  # Semiconductor stocks
        "years": 2,
        "hp_iterations": 10,  # Reduced for faster testing
        "strategy_type": "momentum"
    }

    logger.info(f"Sending test job: {job_id}")
    logger.info(f"  Symbols: {job['symbols']}")
    logger.info(f"  Years: {job['years']}")
    logger.info(f"  HP Iterations: {job['hp_iterations']}")

    # Send job to queue
    redis_client.rpush(config.job_queue, json.dumps(job))
    logger.info(f"Job queued on {config.job_queue}")

    # Subscribe to result channel
    result_channel = f"{config.result_channel}:{job_id}"
    progress_channel = f"{config.progress_channel_prefix}:{job_id}"

    pubsub = redis_client.pubsub()
    pubsub.subscribe(result_channel, progress_channel)

    logger.info(f"Listening for results on {result_channel}")
    logger.info(f"Listening for progress on {progress_channel}")

    # Wait for result (timeout after 1 hour)
    timeout = time.time() + 3600
    result = None

    try:
        for message in pubsub.listen():
            if time.time() > timeout:
                logger.error("Test timeout - no result received")
                break

            if message['type'] != 'message':
                continue

            channel = message['channel']
            data = json.loads(message['data'])

            if channel == progress_channel:
                phase = data.get('phase', 'unknown')
                status = data.get('status', 'unknown')
                logger.info(f"  Progress: {phase} - {status}")

            elif channel == result_channel:
                result = data
                logger.info(f"Result received!")
                break

    except KeyboardInterrupt:
        logger.info("Test interrupted")
    finally:
        pubsub.close()

    # Verify result
    if result is None:
        logger.error("No result received")
        return False

    logger.info("\n" + "="*60)
    logger.info("TEST RESULT")
    logger.info("="*60)

    status = result.get('status', 'unknown')
    logger.info(f"Status: {status}")

    if status == "success":
        metrics = result.get('model_metrics', {})
        logger.info(f"Model: {metrics.get('model_name', 'unknown')}")
        logger.info(f"F1 Score: {metrics.get('f1', 0):.4f}")
        logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall: {metrics.get('recall', 0):.4f}")
        logger.info(f"AUC: {metrics.get('auc', 0):.4f}")

        rules = result.get('extracted_rules', {})
        entry_rules = rules.get('entry_rules', [])
        exit_rules = rules.get('exit_rules', [])
        logger.info(f"Entry Rules: {len(entry_rules)}")
        logger.info(f"Exit Rules: {len(exit_rules)}")

        logger.info("\n✅ TEST PASSED")
        return True
    else:
        error = result.get('error', 'unknown')
        logger.error(f"Error: {error}")
        logger.info("\n❌ TEST FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_worker())
    exit(0 if success else 1)

