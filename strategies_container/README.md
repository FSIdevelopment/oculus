# Strategies Container Build Agent

A lightweight FastAPI service that runs on the **signalSynk-Strategies Droplet** and handles all Docker image building and pushing for Oculus strategy containers.

## Why This Exists

The Oculus backend runs on DigitalOcean App Platform, which is a managed container environment with no Docker daemon access. When a strategy build completes, the backend needs to package the strategy files into a Docker image and push it to the container registry — but it cannot do this itself. This service solves that by acting as a remote build agent: the backend sends all strategy files over HTTP, and this agent does the actual `docker build` and `docker push` on the Droplet where Docker is available.

---

## Architecture

```
Oculus Backend (App Platform)
        │
        │  POST /build
        │  { strategy_name, version, files: { 10 files } }
        │  X-Build-Api-Key: <shared secret>
        ▼
signalSynk-Strategies Droplet :8088
        │
        ├─ Writes all files to /tmp/<uuid>/
        ├─ docker build -t registry.../strategy-name:<version>
        ├─ docker push (versioned tag + :latest)
        └─ Returns { success, image_tag }
        │
        ▼
registry.digitalocean.com/oculus-strategies
```

Each completed strategy build produces one versioned Docker image in the registry (e.g. `registry.digitalocean.com/oculus-strategies/momentum-spy:3`). The version number matches the strategy version shown in Oculus.

---

## API Endpoints

### `GET /health`

Liveness check. No authentication required.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /build`

Builds and pushes a Docker image for a strategy.

**Headers:**
```
X-Build-Api-Key: <BUILD_API_KEY>
Content-Type: application/json
```

**Request body:**
```json
{
  "strategy_name": "momentum-spy",
  "version": 3,
  "files": {
    "strategy.py":        "...",
    "backtest.py":        "...",
    "data_provider.py":   "...",
    "risk_manager.py":    "...",
    "main.py":            "...",
    "Dockerfile":         "...",
    "docker-compose.yml": "...",
    "config.json":        "...",
    "requirements.txt":   "...",
    "extracted_rules.json": "..."
  }
}
```

All 10 files are stored in `BuildIteration.strategy_files` in the Oculus database and are sent verbatim by the backend.

**Success response:**
```json
{
  "success": true,
  "image_tag": "registry.digitalocean.com/oculus-strategies/momentum-spy:3",
  "error": null
}
```

**Failure response:**
```json
{
  "success": false,
  "image_tag": null,
  "error": "docker build failed (exit 1): ..."
}
```

The agent always returns HTTP 200. Check `success` to determine the outcome.

---

## Environment Variables

### On the Droplet (`/opt/strategies-build-agent/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `BUILD_API_KEY` | Yes | Shared secret — must match `BUILD_AGENT_API_KEY` in Oculus |
| `DO_REGISTRY_URL` | Yes | `registry.digitalocean.com/oculus-strategies` |
| `DO_REGISTRY_TOKEN` | Yes | DigitalOcean API token with registry write access |

### In Oculus App Platform

| Variable | Description |
|----------|-------------|
| `BUILD_AGENT_URL` | `http://<droplet-ip>:8088` |
| `BUILD_AGENT_API_KEY` | Same secret as `BUILD_API_KEY` on the Droplet |

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — request validation, auth, routing |
| `builder.py` | Docker build/push logic — writes files to temp dir, runs docker CLI |
| `requirements.txt` | Service dependencies (fastapi, uvicorn, python-dotenv) |
| `deploy.sh` | Setup script that runs **on the Droplet** — installs deps, creates systemd unit |
| `deploy_to_droplet.sh` | Deployment script that runs **locally** — SCPs files and triggers setup |
| `recover_builds.py` | One-time recovery script to push all historical strategy images |
| `tests/` | Unit tests for builder and API (11 tests, no Docker required) |

---

## Deploy Sequence

### First-time setup

**1. Add your Droplet IP to `.env` in the project root:**
```
DROPLET_IP=<your-droplet-ip>
```

**2. Run the deploy script from the project root:**
```bash
bash strategies_container/deploy_to_droplet.sh
```

This will:
- Copy `main.py`, `builder.py`, `requirements.txt`, `deploy.sh` to the Droplet
- Install Python dependencies on the Droplet
- Create and enable a `systemd` service (`strategies-build-agent`)
- Start the service on port `8088`
- Verify the `/health` endpoint responds

**3. Set real credentials on the Droplet:**
```bash
ssh root@<droplet-ip>
nano /opt/strategies-build-agent/.env
```

Fill in:
```
BUILD_API_KEY=<generate with: openssl rand -hex 32>
DO_REGISTRY_URL=registry.digitalocean.com/oculus-strategies
DO_REGISTRY_TOKEN=<your DO API token>
```

Then restart the service:
```bash
systemctl restart strategies-build-agent
curl http://localhost:8088/health
```

**4. Add env vars to Oculus in App Platform:**

In the DigitalOcean dashboard → Oculus app → Settings → Environment Variables:

```
BUILD_AGENT_URL=http://<droplet-ip>:8088
BUILD_AGENT_API_KEY=<the BUILD_API_KEY you set in step 3>
```

Trigger a new App Platform deployment to pick up the new vars.

### Subsequent deploys

Any time `main.py` or `builder.py` changes, redeploy with:
```bash
bash strategies_container/deploy_to_droplet.sh
```

The script is idempotent — it overwrites files and restarts the service cleanly.

---

## Recovering Historical Builds

All strategies built before this agent existed have no Docker image in the registry. The `recover_builds.py` script re-pushes all of them in the correct version order.

**Run from the project root (requires `asyncpg` and `httpx`):**
```bash
pip install asyncpg httpx

DATABASE_URL="<production DATABASE_URL>" \
BUILD_AGENT_URL="http://<droplet-ip>:8088" \
BUILD_AGENT_API_KEY="<secret>" \
python3 strategies_container/recover_builds.py
```

The script:
1. Checks the build agent is reachable before touching the database
2. Finds all completed strategies with completed builds
3. Assigns versions by ordering builds by `completed_at DESC` — matching the version history shown in the Oculus admin panel exactly
4. Pushes one Docker image per completed build (all versions, not just the latest)
5. Updates `strategy.docker_image_url` in the database to the most recently pushed tag
6. Prints a progress summary showing every push, skip, and failure

---

## Running Tests

```bash
cd strategies_container
pip install fastapi uvicorn pytest pytest-asyncio httpx
pytest tests/ -v
```

All 11 tests mock the Docker CLI — no Docker installation needed to run them.

---

## Troubleshooting

**Service not starting:**
```bash
ssh root@<droplet-ip>
systemctl status strategies-build-agent
journalctl -u strategies-build-agent -n 50
```

**Build failing with "Registry login failed":**
Check that `DO_REGISTRY_TOKEN` in `/opt/strategies-build-agent/.env` is a valid DigitalOcean API token with registry write access.

**Build agent unreachable from App Platform:**
Ensure port `8088` is open in the Droplet's firewall:
```bash
ufw allow 8088/tcp
```
Or use DigitalOcean's Cloud Firewall to restrict access to App Platform outbound IPs only.

**Check what images are in the registry:**
```bash
doctl registry repository list-tags oculus-strategies
```
