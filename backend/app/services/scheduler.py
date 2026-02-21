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
    logger.info("Scheduler started with 7 staggered cron jobs on master process")


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

