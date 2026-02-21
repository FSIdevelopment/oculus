# Production Server Crash Fix - Cron Job Scheduler Issue

## Problem Summary

The production server was crashing with exit code 128 due to **multiple scheduler instances running simultaneously**.

### Root Cause

1. **4 Uvicorn workers** were configured in `start.sh` (`UVICORN_WORKERS=4`)
2. **Each worker** started its own APScheduler instance in the FastAPI `lifespan` function
3. **Each scheduler** added all 7 cron jobs
4. **Result**: 28 concurrent cron job instances (4 workers × 7 jobs) competing for resources

### Evidence from Logs

```
api 2026-02-21T14:16:29 [INFO] app.services.scheduler: Scheduler started with 7 cron jobs  # Worker 1
api 2026-02-21T14:16:29 [INFO] app.services.scheduler: Scheduler started with 7 cron jobs  # Worker 2
api 2026-02-21T14:16:30 [INFO] app.services.scheduler: Scheduler started with 7 cron jobs  # Worker 3
api 2026-02-21T14:16:31 [INFO] app.services.scheduler: Scheduler started with 7 cron jobs  # Worker 4
```

Even with Redis distributed locking, having 4 schedulers competing for locks every few minutes caused:
- Resource contention
- Database connection pool exhaustion
- Memory pressure
- Server crashes (exit code 128)

## Solution Implemented

### 1. Master-Only Scheduler Execution

Modified `backend/app/services/scheduler.py` to only run the scheduler on the **master process**:

```python
def start_scheduler():
    """Start the APScheduler with all cron jobs."""
    import os
    
    # Only start scheduler on the master worker process
    # Uvicorn sets UVICORN_WORKER_ID for worker processes (not set for master)
    worker_id = os.environ.get("UVICORN_WORKER_ID")
    
    # Skip scheduler on worker processes (only run on master)
    if worker_id is not None:
        logger.info(f"Skipping scheduler on worker {worker_id} - scheduler only runs on master process")
        return
    
    logger.info("Starting scheduler on master process...")
    # ... rest of scheduler setup
```

### 2. Staggered Cron Job Execution

Added staggered start times to prevent all jobs from running simultaneously:

| Job | Interval | Start Offset | Purpose |
|-----|----------|--------------|---------|
| Update stuck builds | 10 min | :00 (immediate) | Check for stuck builds |
| Reconnect websockets | 5 min | :01 (+1 min) | WebSocket heartbeat |
| Reconnect workers | 8 min | :02 (+2 min) | Worker connectivity |
| Send build complete emails | 10 min | :03 (+3 min) | Email notifications |
| Update expired licenses | 1 hour | :05 (+5 min) | License management |
| Send build failed emails | 10 min | :06 (+6 min) | Email notifications |
| Send license expiry warnings | Daily 9:00 AM UTC | N/A | Daily warnings |

### 3. Graceful Shutdown

Updated `stop_scheduler()` to only shutdown on the master process:

```python
def stop_scheduler():
    """Stop the APScheduler."""
    import os
    
    worker_id = os.environ.get("UVICORN_WORKER_ID")
    
    if worker_id is not None:
        logger.info(f"Skipping scheduler shutdown on worker {worker_id}")
        return
    
    logger.info("Stopping scheduler on master process...")
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")
```

## Expected Behavior After Fix

### Before (4 workers × 7 jobs = 28 instances)
```
Worker 0 (master): 7 schedulers running
Worker 1: 7 schedulers running
Worker 2: 7 schedulers running
Worker 3: 7 schedulers running
Total: 28 scheduler instances
```

### After (1 master × 7 jobs = 7 instances)
```
Worker 0 (master): 7 schedulers running (staggered)
Worker 1: No schedulers (API requests only)
Worker 2: No schedulers (API requests only)
Worker 3: No schedulers (API requests only)
Total: 7 scheduler instances
```

## Deployment

The fix has been committed and is ready to deploy. After deployment, you should see:

1. **Only 1 log line** showing scheduler startup (from master process)
2. **3 log lines** showing scheduler skipped on workers
3. **No more crashes** due to scheduler contention
4. **Staggered job execution** reducing simultaneous database load

## Monitoring

After deployment, monitor for:
- ✅ Single scheduler startup message
- ✅ Worker processes skipping scheduler
- ✅ No exit code 128 crashes
- ✅ Cron jobs executing at staggered intervals
- ✅ Redis locks being acquired/released properly

## Files Modified

- `backend/app/services/scheduler.py` - Added master-only execution and staggered job timing

