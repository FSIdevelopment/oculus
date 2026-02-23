"""Scheduler service for cron jobs using APScheduler."""
import logging
import asyncio
import os
import multiprocessing
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload
import redis.asyncio as redis

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.strategy_build import StrategyBuild
from app.models.license import License
from app.models.user import User
from app.models.strategy import Strategy
from app.services.email_service import EmailService
from app.services.build_recovery import BuildRecoveryService

logger = logging.getLogger(__name__)

# Initialize scheduler
scheduler = AsyncIOScheduler()

# Redis client for distributed locking and notification tracking
redis_client = None

# Track the parent process ID to determine if we're in the master process
_parent_pid = os.getpid()


async def get_redis_client():
    """Get or create Redis client."""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return redis_client


async def acquire_lock(lock_name: str, timeout: int = 300) -> bool:
    """
    Acquire a distributed lock using Redis.
    
    Args:
        lock_name: Name of the lock
        timeout: Lock timeout in seconds
        
    Returns:
        True if lock acquired, False otherwise
    """
    try:
        client = await get_redis_client()
        # Use SET with NX (only set if not exists) and EX (expiry)
        result = await client.set(f"oculus:lock:{lock_name}", "1", nx=True, ex=timeout)
        return result is not None
    except Exception as e:
        logger.error(f"Failed to acquire lock {lock_name}: {e}")
        return False


async def release_lock(lock_name: str):
    """Release a distributed lock."""
    try:
        client = await get_redis_client()
        await client.delete(f"oculus:lock:{lock_name}")
    except Exception as e:
        logger.error(f"Failed to release lock {lock_name}: {e}")


async def update_stuck_builds():
    """Update builds stuck in queued or stopping status."""
    lock_name = "update_stuck_builds"
    
    if not await acquire_lock(lock_name):
        logger.info(f"Skipping {lock_name} - another instance is running")
        return
    
    try:
        logger.info("Running update_stuck_builds job")
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()
            
            # Find builds stuck in "queued" for > 30 minutes
            # Exclude builds that are actively processing (completed_at is None and phase is not "resuming")
            # Also exclude builds that were just restarted (phase = "resuming")
            queued_threshold = now - timedelta(minutes=30)
            queued_query = select(StrategyBuild).where(
                and_(
                    StrategyBuild.status == "queued",
                    StrategyBuild.started_at < queued_threshold,
                    StrategyBuild.completed_at.is_(None),  # Only check builds that haven't completed
                    StrategyBuild.phase != "resuming"  # Exclude restarted builds
                )
            )
            queued_result = await session.execute(queued_query)
            queued_builds = queued_result.scalars().all()

            for build in queued_builds:
                build.status = "failed"
                build.phase = "Timeout - stuck in queue"
                build.completed_at = now
                logger.warning(f"Failed stuck queued build {build.uuid}")
            
            # Find builds stuck in "stopping" for > 15 minutes
            stopping_threshold = now - timedelta(minutes=15)
            stopping_query = select(StrategyBuild).where(
                and_(
                    StrategyBuild.status == "stopping",
                    StrategyBuild.started_at < stopping_threshold
                )
            )
            stopping_result = await session.execute(stopping_query)
            stopping_builds = stopping_result.scalars().all()
            
            for build in stopping_builds:
                build.status = "stopped"
                build.completed_at = now
                logger.warning(f"Marked stuck stopping build {build.uuid} as stopped")
            
            await session.commit()
            logger.info(f"Updated {len(queued_builds)} queued and {len(stopping_builds)} stopping builds")
            
    except Exception as e:
        logger.error(f"Error in update_stuck_builds: {e}")
    finally:
        await release_lock(lock_name)


async def reconnect_workers():
    """Check worker connectivity and handle stuck training builds."""
    lock_name = "reconnect_workers"
    
    if not await acquire_lock(lock_name):
        logger.info(f"Skipping {lock_name} - another instance is running")
        return
    
    try:
        logger.info("Running reconnect_workers job")
        
        # Ping Redis to check connectivity
        client = await get_redis_client()
        await client.ping()
        logger.info("Redis connection healthy")
        
        # Check for builds stuck in "training" for > 60 minutes
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()
            training_threshold = now - timedelta(minutes=60)
            
            query = select(StrategyBuild).where(
                and_(
                    StrategyBuild.status == "training",
                    StrategyBuild.started_at < training_threshold
                )
            )
            result = await session.execute(query)
            stuck_builds = result.scalars().all()
            
            for build in stuck_builds:
                build.status = "failed"
                build.phase = "Timeout - training took too long"
                build.completed_at = now
                logger.warning(f"Failed stuck training build {build.uuid}")
            
            await session.commit()
            logger.info(f"Checked worker connectivity, failed {len(stuck_builds)} stuck training builds")
            
    except Exception as e:
        logger.error(f"Error in reconnect_workers: {e}")
    finally:
        await release_lock(lock_name)


async def reconnect_websockets():
    """Check Redis pub/sub connectivity for WebSocket connections."""
    lock_name = "reconnect_websockets"

    if not await acquire_lock(lock_name):
        logger.info(f"Skipping {lock_name} - another instance is running")
        return

    try:
        logger.info("Running reconnect_websockets job")

        # Ping Redis pub/sub to check connectivity
        client = await get_redis_client()
        await client.ping()

        # Publish heartbeat for active builds
        async with AsyncSessionLocal() as session:
            active_statuses = ["building", "designing", "training", "extracting_rules", "optimizing", "building_docker"]
            query = select(StrategyBuild).where(StrategyBuild.status.in_(active_statuses))
            result = await session.execute(query)
            active_builds = result.scalars().all()

            for build in active_builds:
                channel = f"build:{build.uuid}:progress"
                message = {
                    "status": build.status,
                    "phase": build.phase,
                    "heartbeat": True
                }
                await client.publish(channel, str(message))

            logger.info(f"Published heartbeat for {len(active_builds)} active builds")

    except Exception as e:
        logger.error(f"Error in reconnect_websockets: {e}")
    finally:
        await release_lock(lock_name)


async def recover_stuck_builds():
    """Recover stuck builds and reassign from dead workers."""
    lock_name = "recover_stuck_builds"

    if not await acquire_lock(lock_name):
        logger.info(f"Skipping {lock_name} - another instance is running")
        return

    try:
        logger.info("Running recover_stuck_builds job")

        # Get Redis client
        client = await get_redis_client()

        # Create recovery service
        recovery_service = BuildRecoveryService(client)

        # Run recovery
        async with AsyncSessionLocal() as session:
            stats = await recovery_service.recover_stuck_builds(session)

            logger.info(
                f"Build recovery complete: {stats['recovered']} recovered, "
                f"{stats['failed']} failed, {stats['requeued']} requeued, "
                f"{stats['dead_workers']} dead workers"
            )

    except Exception as e:
        logger.error(f"Error in recover_stuck_builds: {e}")
    finally:
        await release_lock(lock_name)


async def update_expired_licenses():
    """Update licenses that have expired."""
    lock_name = "update_expired_licenses"

    if not await acquire_lock(lock_name):
        logger.info(f"Skipping {lock_name} - another instance is running")
        return

    try:
        logger.info("Running update_expired_licenses job")
        async with AsyncSessionLocal() as session:
            now = datetime.utcnow()

            query = select(License).where(
                and_(
                    License.status == "active",
                    License.expires_at < now
                )
            )
            result = await session.execute(query)
            expired_licenses = result.scalars().all()

            for license in expired_licenses:
                license.status = "expired"
                logger.info(f"Expired license {license.uuid}")

            await session.commit()
            logger.info(f"Updated {len(expired_licenses)} expired licenses")

    except Exception as e:
        logger.error(f"Error in update_expired_licenses: {e}")
    finally:
        await release_lock(lock_name)


async def send_build_complete_emails():
    """Send emails for completed builds."""
    lock_name = "send_build_complete_emails"

    if not await acquire_lock(lock_name):
        logger.info(f"Skipping {lock_name} - another instance is running")
        return

    try:
        logger.info("Running send_build_complete_emails job")
        client = await get_redis_client()

        async with AsyncSessionLocal() as session:
            # Query builds completed in the last 24 hours
            since = datetime.utcnow() - timedelta(hours=24)
            query = (
                select(StrategyBuild)
                .options(selectinload(StrategyBuild.strategy), selectinload(StrategyBuild.user))
                .where(
                    and_(
                        StrategyBuild.status == "complete",
                        StrategyBuild.completed_at >= since
                    )
                )
            )
            result = await session.execute(query)
            completed_builds = result.scalars().all()

            sent_count = 0
            for build in completed_builds:
                # Check if email already sent
                redis_key = f"oculus:notif:build:{build.uuid}:complete_email"
                already_sent = await client.get(redis_key)

                if not already_sent:
                    # Send email
                    success = EmailService.send_build_complete_email(
                        user=build.user,
                        strategy=build.strategy,
                        build=build
                    )

                    if success:
                        # Mark as sent in Redis (expire after 7 days)
                        await client.set(redis_key, "1", ex=604800)
                        sent_count += 1

            logger.info(f"Sent {sent_count} build complete emails")

    except Exception as e:
        logger.error(f"Error in send_build_complete_emails: {e}")
    finally:
        await release_lock(lock_name)


async def send_build_failed_emails():
    """Send emails for failed builds."""
    lock_name = "send_build_failed_emails"

    if not await acquire_lock(lock_name):
        logger.info(f"Skipping {lock_name} - another instance is running")
        return

    try:
        logger.info("Running send_build_failed_emails job")
        client = await get_redis_client()

        async with AsyncSessionLocal() as session:
            # Query builds failed in the last 24 hours
            since = datetime.utcnow() - timedelta(hours=24)
            query = (
                select(StrategyBuild)
                .options(selectinload(StrategyBuild.strategy), selectinload(StrategyBuild.user))
                .where(
                    and_(
                        StrategyBuild.status == "failed",
                        StrategyBuild.completed_at >= since
                    )
                )
            )
            result = await session.execute(query)
            failed_builds = result.scalars().all()

            sent_count = 0
            for build in failed_builds:
                # Check if email already sent
                redis_key = f"oculus:notif:build:{build.uuid}:failed_email"
                already_sent = await client.get(redis_key)

                if not already_sent:
                    # Send email
                    success = EmailService.send_build_failed_email(
                        user=build.user,
                        strategy=build.strategy,
                        build=build
                    )

                    if success:
                        # Mark as sent in Redis (expire after 7 days)
                        await client.set(redis_key, "1", ex=604800)
                        sent_count += 1

            logger.info(f"Sent {sent_count} build failed emails")

    except Exception as e:
        logger.error(f"Error in send_build_failed_emails: {e}")
    finally:
        await release_lock(lock_name)


async def assign_waiting_builds_to_workers():
    """Assign builds waiting for workers to available workers.

    This job runs every 15 seconds to check for builds in 'waiting_for_worker' status
    and dispatches them to the training queue when worker capacity becomes available.
    """
    lock_name = "assign_waiting_builds"

    if not await acquire_lock(lock_name):
        logger.debug(f"Skipping {lock_name} - another instance is running")
        return

    try:
        logger.debug("Running assign_waiting_builds_to_workers job")
        client = await get_redis_client()

        async with AsyncSessionLocal() as session:
            from app.services.worker_health import WorkerHealthManager

            worker_manager = WorkerHealthManager()

            # Get available worker capacity
            workers = await worker_manager.get_active_workers(session)
            total_capacity = sum(w.capacity for w in workers)
            active_jobs = sum(w.active_jobs for w in workers)
            available_capacity = total_capacity - active_jobs

            if available_capacity <= 0:
                logger.debug("No available worker capacity for waiting builds")
                return

            # Get builds waiting for workers (ordered by created_at)
            result = await session.execute(
                select(StrategyBuild)
                .where(StrategyBuild.status == "waiting_for_worker")
                .order_by(StrategyBuild.created_at)
                .limit(available_capacity)
            )
            waiting_builds = result.scalars().all()

            if not waiting_builds:
                logger.debug("No builds waiting for workers")
                return

            logger.info(f"Available capacity: {available_capacity}, Waiting builds: {len(waiting_builds)}")

            # Dispatch waiting builds to training queue
            builds_dispatched = 0
            for build in waiting_builds:
                try:
                    # Get pending job payload from Redis
                    job_id = f"build:{build.uuid}:iter:*"  # Pattern match
                    pending_keys = await client.keys(f"pending:{job_id}")

                    if not pending_keys:
                        logger.warning(f"No pending job found for build {build.uuid}, skipping")
                        # Reset status so it can be retried
                        build.status = "building"
                        build.phase = "Retrying training dispatch"
                        await session.commit()
                        continue

                    # Get the job payload
                    job_payload_str = await client.get(pending_keys[0])
                    if not job_payload_str:
                        logger.warning(f"Pending job payload empty for build {build.uuid}")
                        continue

                    job_payload = json.loads(job_payload_str)
                    actual_job_id = pending_keys[0].replace("pending:", "")

                    # Dispatch to training queue
                    await client.delete(f"result:{actual_job_id}")  # Clean up stale results
                    await client.set(actual_job_id, json.dumps(job_payload), ex=86400)
                    await client.lpush(settings.TRAINING_QUEUE, actual_job_id)

                    # Remove pending job
                    await client.delete(pending_keys[0])

                    # Update build status to training
                    build.status = "training"
                    build.phase = "Training ML model"
                    await session.commit()

                    builds_dispatched += 1
                    logger.info(f"Dispatched waiting build {build.uuid} to training queue")

                except Exception as e:
                    logger.error(f"Failed to dispatch waiting build {build.uuid}: {e}", exc_info=True)
                    await session.rollback()
                    continue

            logger.info(f"Dispatched {builds_dispatched} waiting builds to training queue")

    except Exception as e:
        logger.error(f"Error in assign_waiting_builds_to_workers: {e}", exc_info=True)
    finally:
        await release_lock(lock_name)


async def process_build_queue():
    """Process queued builds and start them when workers have capacity."""
    lock_name = "process_build_queue"

    if not await acquire_lock(lock_name):
        logger.debug(f"Skipping {lock_name} - another instance is running")
        return

    try:
        logger.debug("Running process_build_queue job")
        client = await get_redis_client()

        async with AsyncSessionLocal() as session:
            # Import here to avoid circular dependency
            from app.services.build_queue import BuildQueueManager
            from app.services.worker_health import WorkerHealthManager

            queue_manager = BuildQueueManager(client)
            worker_manager = WorkerHealthManager()

            # Get available worker capacity
            workers = await worker_manager.get_active_workers(session)
            total_capacity = sum(w.capacity for w in workers)
            active_jobs = sum(w.active_jobs for w in workers)
            available_capacity = total_capacity - active_jobs

            if available_capacity <= 0:
                logger.debug("No available worker capacity")
                return

            # Get queued builds from Redis
            queue_items = await client.zrange(queue_manager.QUEUE_KEY, 0, -1)
            if not queue_items:
                logger.debug("No builds in queue")
                return

            logger.info(f"Available capacity: {available_capacity}, Queued builds: {len(queue_items)}")

            # Process builds up to available capacity
            builds_started = 0
            for build_id in queue_items[:available_capacity]:
                try:
                    # Get build from database
                    result = await session.execute(
                        select(StrategyBuild).where(StrategyBuild.uuid == build_id)
                    )
                    build = result.scalar_one_or_none()

                    if not build:
                        logger.warning(f"Build {build_id} not found in database, removing from queue")
                        await client.zrem(queue_manager.QUEUE_KEY, build_id)
                        continue

                    # Only process builds that are still queued
                    if build.status != "queued":
                        logger.info(f"Build {build_id} status is {build.status}, removing from queue")
                        await client.zrem(queue_manager.QUEUE_KEY, build_id)
                        continue

                    # Start the build by triggering the background task
                    # Import the start_build_background function
                    from app.routers.builds import start_build_background
                    import asyncio

                    # Remove from queue first
                    await client.zrem(queue_manager.QUEUE_KEY, build_id)
                    build.queue_position = None
                    build.status = "building"
                    await session.commit()

                    # Start build in background
                    asyncio.create_task(start_build_background(
                        build_id=build.uuid,
                        user_id=build.user_id,
                        strategy_id=build.strategy_id,
                        max_iterations=build.max_iterations or 5
                    ))

                    builds_started += 1
                    logger.info(f"Started build {build_id} from queue")

                except Exception as e:
                    logger.error(f"Failed to start build {build_id}: {e}", exc_info=True)
                    continue

            if builds_started > 0:
                logger.info(f"Started {builds_started} builds from queue")

    except Exception as e:
        logger.error(f"Error in process_build_queue: {e}", exc_info=True)
    finally:
        await release_lock(lock_name)


async def send_license_expiry_warnings():
    """Send emails for expiring licenses at 15, 5, 3, and 1 day marks."""
    lock_name = "send_license_expiry_warnings"

    if not await acquire_lock(lock_name):
        logger.info(f"Skipping {lock_name} - another instance is running")
        return

    try:
        logger.info("Running send_license_expiry_warnings job")
        client = await get_redis_client()
        now = datetime.utcnow()
        today_str = now.strftime("%Y-%m-%d")

        # Define warning thresholds (in days)
        thresholds = [15, 5, 3, 1]

        async with AsyncSessionLocal() as session:
            sent_count = 0

            for days in thresholds:
                # Calculate the expiry window for this threshold
                # We want licenses expiring between (days) and (days+1) days from now
                start_time = now + timedelta(days=days)
                end_time = now + timedelta(days=days + 1)

                query = (
                    select(License)
                    .options(selectinload(License.strategy), selectinload(License.user))
                    .where(
                        and_(
                            License.status == "active",
                            License.expires_at >= start_time,
                            License.expires_at < end_time
                        )
                    )
                )
                result = await session.execute(query)
                expiring_licenses = result.scalars().all()

                for license in expiring_licenses:
                    # Check if we already sent this warning today
                    redis_key = f"oculus:notif:license:{license.uuid}:warning:{days}d:{today_str}"
                    already_sent = await client.get(redis_key)

                    if not already_sent:
                        # Send email
                        success = EmailService.send_license_expiry_warning_email(
                            user=license.user,
                            license=license,
                            strategy=license.strategy,
                            days_remaining=days
                        )

                        if success:
                            # Mark as sent for today (expire after 2 days)
                            await client.set(redis_key, "1", ex=172800)
                            sent_count += 1

            logger.info(f"Sent {sent_count} license expiry warning emails")

    except Exception as e:
        logger.error(f"Error in send_license_expiry_warnings: {e}")
    finally:
        await release_lock(lock_name)


def start_scheduler():
    """Start the APScheduler with all cron jobs."""
    # Only start scheduler on the FIRST worker process
    # When uvicorn runs with --workers, the parent process doesn't run the app
    # Worker processes have names like "SpawnProcess-1", "SpawnProcess-2", etc.
    # We run the scheduler ONLY on SpawnProcess-1 to avoid duplication
    current_pid = os.getpid()
    current_process_name = multiprocessing.current_process().name

    # Skip scheduler on all workers except SpawnProcess-1
    if current_process_name != "SpawnProcess-1":
        logger.info(f"Skipping scheduler on {current_process_name} (PID: {current_pid}) - scheduler only runs on SpawnProcess-1")
        return

    logger.info(f"Starting scheduler on {current_process_name} (PID: {current_pid})...")

    # Job 0: Process build queue every 30 seconds
    scheduler.add_job(
        process_build_queue,
        trigger=IntervalTrigger(seconds=30),
        id="process_build_queue",
        name="Process build queue",
        replace_existing=True
    )

    # Job 0.5: Assign waiting builds to workers every 15 seconds
    scheduler.add_job(
        assign_waiting_builds_to_workers,
        trigger=IntervalTrigger(seconds=15),
        id="assign_waiting_builds",
        name="Assign waiting builds to workers",
        replace_existing=True
    )

    # Job 1: Update stuck builds every 10 minutes (staggered: starts at :00)
    scheduler.add_job(
        update_stuck_builds,
        trigger=IntervalTrigger(minutes=10),
        id="update_stuck_builds",
        name="Update stuck builds",
        replace_existing=True
    )

    # Job 2: Reconnect workers every 8 minutes (staggered: starts at :02)
    scheduler.add_job(
        reconnect_workers,
        trigger=IntervalTrigger(minutes=8, start_date=datetime.utcnow() + timedelta(minutes=2)),
        id="reconnect_workers",
        name="Reconnect workers",
        replace_existing=True
    )

    # Job 3: Reconnect websockets every 5 minutes (staggered: starts at :01)
    scheduler.add_job(
        reconnect_websockets,
        trigger=IntervalTrigger(minutes=5, start_date=datetime.utcnow() + timedelta(minutes=1)),
        id="reconnect_websockets",
        name="Reconnect websockets",
        replace_existing=True
    )

    # Job 3.5: Recover stuck builds every 2 minutes (staggered: starts at :00:30)
    scheduler.add_job(
        recover_stuck_builds,
        trigger=IntervalTrigger(minutes=2, start_date=datetime.utcnow() + timedelta(seconds=30)),
        id="recover_stuck_builds",
        name="Recover stuck builds",
        replace_existing=True
    )

    # Job 4: Update expired licenses every 1 hour (staggered: starts at :05)
    scheduler.add_job(
        update_expired_licenses,
        trigger=IntervalTrigger(hours=1, start_date=datetime.utcnow() + timedelta(minutes=5)),
        id="update_expired_licenses",
        name="Update expired licenses",
        replace_existing=True
    )

    # Job 5: Send build complete emails every 10 minutes (staggered: starts at :03)
    scheduler.add_job(
        send_build_complete_emails,
        trigger=IntervalTrigger(minutes=10, start_date=datetime.utcnow() + timedelta(minutes=3)),
        id="send_build_complete_emails",
        name="Send build complete emails",
        replace_existing=True
    )

    # Job 6: Send build failed emails every 10 minutes (staggered: starts at :06)
    scheduler.add_job(
        send_build_failed_emails,
        trigger=IntervalTrigger(minutes=10, start_date=datetime.utcnow() + timedelta(minutes=6)),
        id="send_build_failed_emails",
        name="Send build failed emails",
        replace_existing=True
    )

    # Job 7: Send license expiry warnings once per day at 9:00 AM UTC
    scheduler.add_job(
        send_license_expiry_warnings,
        trigger=CronTrigger(hour=9, minute=0),
        id="send_license_expiry_warnings",
        name="Send license expiry warnings",
        replace_existing=True
    )

    scheduler.start()
    logger.info("Scheduler started with 9 staggered cron jobs on master process")


def stop_scheduler():
    """Stop the APScheduler."""
    # Only stop scheduler on SpawnProcess-1 (where it's running)
    current_pid = os.getpid()
    current_process_name = multiprocessing.current_process().name

    if current_process_name != "SpawnProcess-1":
        logger.info(f"Skipping scheduler shutdown on {current_process_name} (PID: {current_pid})")
        return

    logger.info(f"Stopping scheduler on {current_process_name} (PID: {current_pid})...")
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")
    else:
        logger.info("Scheduler was not running")

