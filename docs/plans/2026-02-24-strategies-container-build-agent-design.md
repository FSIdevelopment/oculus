# Design: strategies_container Build Agent

**Date:** 2026-02-24
**Status:** Approved
**Priority:** Critical — production SignalSynk environment is down

---

## Problem

The Oculus backend runs on DigitalOcean App Platform (a container environment). Docker CLI is not installed in the container and there is no Docker daemon socket available. All `docker build` / `docker push` commands in `DockerBuilder` silently fail with exit code 127 (command not found). The build pipeline marks strategies as `complete` but no images are ever pushed to the `registry.digitalocean.com/oculus-strategies` registry. Every strategy that has ever been built on production is missing its container image.

---

## Root Cause

`backend/Dockerfile` installs only `gcc`, `curl`, and `postgresql-client` — no Docker CLI.
`docker-compose.yml` does not mount `/var/run/docker.sock` into the backend container.
The GitHub Actions workflow deploys to **App Platform** (`doctl apps create-deployment`), not a Droplet, so no Docker daemon is available at runtime.

---

## Solution

Delegate all Docker build and push operations to the existing `signalSynk-Strategies` DigitalOcean Droplet (same DO account), which has full Docker access and already runs strategy containers. A lightweight **Build Agent** FastAPI service is deployed on the Droplet. The Oculus backend calls it via HTTP instead of running Docker commands locally.

---

## Architecture

```
Oculus Backend (App Platform)
        │
        │  POST /build  { strategy_name, version, files[10] }
        │  X-Build-Api-Key: <shared secret>
        ▼
signalSynk-Strategies Droplet  :8088
        │
        ├─ writes files to /tmp/<uuid>/
        ├─ docker build -t registry.digitalocean.com/oculus-strategies/<name>:<version>
        ├─ docker push ...
        └─ returns { success, image_tag }
        │
        ▼
registry.digitalocean.com/oculus-strategies
```

---

## Components

### 1. `strategies_container/` (new top-level folder)

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — `POST /build`, `GET /health` |
| `builder.py` | Temp dir creation, `docker build`, `docker push` logic |
| `requirements.txt` | `fastapi`, `uvicorn`, `python-dotenv` |
| `deploy.sh` | One-shot Droplet setup: installs Python deps, creates systemd unit, starts service |
| `recover_builds.py` | Standalone recovery script — rebuilds all missing production images |

### 2. Backend changes

| File | Change |
|------|--------|
| `backend/app/config.py` | Add `BUILD_AGENT_URL: str` and `BUILD_AGENT_API_KEY: str` |
| `backend/app/services/docker_builder.py` | Add `remote_build_and_push(strategy, build, files_dict)` method |
| `backend/app/routers/builds.py` | Replace both Docker sections (lines ~1353 and ~1620) with `remote_build_and_push` call using `BuildIteration.strategy_files` |

---

## API Contract

### `POST /build`

**Request headers:**
```
X-Build-Api-Key: <BUILD_AGENT_API_KEY>
Content-Type: application/json
```

**Request body:**
```json
{
  "strategy_name": "momentum-spy",
  "version": 3,
  "files": {
    "strategy.py": "...",
    "backtest.py": "...",
    "data_provider.py": "...",
    "risk_manager.py": "...",
    "main.py": "...",
    "Dockerfile": "...",
    "docker-compose.yml": "...",
    "config.json": "...",
    "requirements.txt": "...",
    "extracted_rules.json": "..."
  }
}
```

**Success response (200):**
```json
{
  "success": true,
  "image_tag": "registry.digitalocean.com/oculus-strategies/momentum-spy:3",
  "error": null
}
```

**Failure response (200 with success=false):**
```json
{
  "success": false,
  "image_tag": null,
  "error": "docker build failed with exit code 1: ..."
}
```

### `GET /health`
Returns `{ "status": "ok" }`.

---

## Security

- `X-Build-Api-Key` header required on all `/build` requests
- Key stored as env var on both the Droplet (`BUILD_API_KEY`) and App Platform (`BUILD_AGENT_API_KEY`)
- Droplet firewall: port 8088 open only to App Platform outbound IPs (or restricted by DO VPC if available)
- Registry credentials (`DO_REGISTRY_TOKEN`, `DO_REGISTRY_URL`) stored as env vars on the Droplet only — never sent over the wire

---

## Version Matching (Recovery)

`Strategy.version` is incremented once per retrain trigger (before the build starts). Version assignment in the recovery script exactly mirrors `admin.py:577-591`:

```
completed_builds = all StrategyBuilds where status='complete', ordered by completed_at DESC
version(build at index i) = strategy.version - i
```

This guarantees the Docker image tags in the registry match the version numbers shown in Oculus.

---

## Environment Variables

### App Platform (Oculus backend)
| Variable | Description |
|----------|-------------|
| `BUILD_AGENT_URL` | `http://<droplet-ip>:8088` |
| `BUILD_AGENT_API_KEY` | Shared secret for build agent auth |

### signalSynk-Strategies Droplet
| Variable | Description |
|----------|-------------|
| `BUILD_API_KEY` | Same shared secret |
| `DO_REGISTRY_TOKEN` | DigitalOcean API token for registry login |
| `DO_REGISTRY_URL` | `registry.digitalocean.com/oculus-strategies` |

---

## Recovery Script (`recover_builds.py`)

Runs once, standalone, against the production database:

1. Connect to DB via `DATABASE_URL` env var
2. For each strategy: fetch all completed `StrategyBuild`s ordered by `completed_at` DESC
3. Assign version = `strategy.version - i` for position i
4. For each build: find the best `BuildIteration` where `status='complete'` and `strategy_files` is populated
5. POST all 10 files to `POST /build` with the correct version
6. On success: update `strategy.docker_image_url` to the latest tag
7. Print progress table; skip builds with no `strategy_files`
8. Run serially (one at a time) to avoid overwhelming the Droplet

---

## Files NOT Changed

- `backend/Dockerfile` — no Docker CLI needed, App Platform stays as-is
- `docker-compose.yml` — local dev only, unchanged
- Existing `build_image`, `test_container`, `push_image` methods in `docker_builder.py` — kept for local dev use
