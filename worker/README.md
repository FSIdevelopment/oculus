# Local Training Worker

Redis-based ML training worker for the Oculus Strategy Platform. Runs on the M3 Max machine and processes training jobs from the backend via Redis.

## Overview

The worker:
1. Listens on Redis queue `training_queue` for training jobs
2. Fetches job payload from Redis key
3. Invokes the existing ML pipeline (EnhancedMLTrainer, RuleExtractor)
4. Publishes real-time progress updates via Redis pub/sub
5. Stores results back in Redis for the backend to retrieve
6. Supports GPU acceleration (PyTorch MPS) for Apple Silicon

## Architecture

```
Backend (DO Server)
    ↓ (dispatch job)
Redis Queue: training_queue
    ↓ (worker polls)
Local Worker (M3 Max)
    ├─ EnhancedMLTrainer (train models)
    ├─ RuleExtractor (extract rules)
    └─ StrategyOptimizer (optimize params)
    ↓ (publish progress)
Redis Pub/Sub: oculus:training:progress:{job_id}
    ↓ (store results)
Redis Key: result:{job_id}
    ↓ (backend polls)
Backend (retrieve results)
```

## Installation

1. Install dependencies:
```bash
pip install -r worker/requirements.txt
```

2. Configure environment variables in `.env`:
```bash
cp .env.example .env
# Edit .env with your Redis connection and API keys
```

## Configuration

Key environment variables:

- `REDIS_HOST` — Redis server host (default: localhost)
- `REDIS_PORT` — Redis server port (default: 6379)
- `REDIS_DB` — Redis database number (default: 0)
- `JOB_QUEUE` — Queue name for training jobs (default: training_queue)
- `RESULT_KEY_PREFIX` — Prefix for result keys (default: result)
- `WORKER_CONCURRENCY` — Number of concurrent jobs (default: 2)
- `GPU_ENABLED` — Enable GPU acceleration (default: true)
- `POLYGON_MASSIVE_API_KEY` — Polygon API key for market data
- `LOG_LEVEL` — Logging level (default: INFO)

## Usage

### Start the worker

```bash
# Using default config from .env
python -m worker.main

# Using custom config file
python -m worker.main --config /path/to/.env

# With custom concurrency
python -m worker.main --concurrency 4
```

### Run integration test

```bash
python -m worker.test_integration
```

This sends a test job to Redis and verifies results come back.

## Job Format

Jobs are dispatched by the backend in this format:

```json
{
  "build_id": "build:uuid",
  "user_id": "user:uuid",
  "symbols": ["NVDA", "AMD"],
  "timeframe": "1d",
  "design": {
    "years": 2,
    "hp_iterations": 30,
    "strategy_type": "momentum"
  }
}
```

## Result Format

Results are stored in Redis with this format:

```json
{
  "build_id": "build:uuid",
  "status": "success",
  "phase": "complete",
  "model_metrics": {
    "model_name": "XGBoost",
    "f1": 0.65,
    "precision": 0.70,
    "recall": 0.60,
    "auc": 0.72
  },
  "extracted_rules": {
    "entry_rules": [...],
    "exit_rules": [...],
    "top_features": [...]
  },
  "backtest_results": {
    "status": "pending"
  }
}
```

## Progress Updates

Real-time progress is published to `oculus:training:progress:{job_id}`:

```json
{
  "job_id": "build:uuid",
  "timestamp": "2024-01-15T10:30:00",
  "phase": "training",
  "status": "in_progress",
  "details": {}
}
```

## GPU Acceleration

The worker automatically detects and uses PyTorch MPS (Metal Performance Shaders) on Apple Silicon when available. Set `GPU_ENABLED=false` to disable.

## Logging

Logs are printed to stdout with timestamps. Set `LOG_LEVEL` to control verbosity:
- `DEBUG` — Detailed debugging information
- `INFO` — General information (default)
- `WARNING` — Warning messages
- `ERROR` — Error messages only

## Troubleshooting

### Redis connection failed
- Ensure Redis is running: `redis-cli ping`
- Check `REDIS_HOST` and `REDIS_PORT` in `.env`

### No jobs being processed
- Check that jobs are being pushed to `training_queue`
- Verify Redis connection with: `redis-cli LLEN training_queue`

### GPU not being used
- Check PyTorch installation: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Ensure `GPU_ENABLED=true` in `.env`

### Polygon API errors
- Set `POLYGON_MASSIVE_API_KEY` in `.env`
- Check API key validity at https://polygon.io

## Development

### Running tests
```bash
python -m worker.test_integration
```

### Code structure
- `main.py` — Entry point, Redis connection, job loop
- `job_processor.py` — ML pipeline invocation
- `progress_reporter.py` — Progress pub/sub
- `config.py` — Configuration management
- `test_integration.py` — Integration test

## Performance

On M3 Max with 2 concurrent workers:
- Data fetch: ~30-60s (depends on symbols and years)
- Feature engineering: ~20-40s
- Model training: ~2-5 minutes (depends on hyperparameter iterations)
- Rule extraction: ~10-20s
- **Total: ~3-7 minutes per job**

Increase `WORKER_CONCURRENCY` to process multiple jobs in parallel.

