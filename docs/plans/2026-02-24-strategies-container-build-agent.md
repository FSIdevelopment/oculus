# strategies_container Build Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy a FastAPI build agent on the signalSynk-Strategies Droplet so the Oculus App Platform backend can delegate all Docker build/push operations there, and recover all existing production strategy images that were never pushed.

**Architecture:** A new `strategies_container/` FastAPI service receives all 10 strategy files via HTTP POST, runs `docker build && push` on the Droplet where Docker is available, and returns the image tag. The Oculus backend replaces its dead local docker commands with an HTTP call to this agent. A standalone recovery script re-pushes every completed build.

**Tech Stack:** FastAPI, uvicorn, httpx (new backend dep), asyncpg (recovery script), Docker CLI on Droplet, systemd

---

## Task 1: Create the build agent service files

**Files:**
- Create: `strategies_container/__init__.py`
- Create: `strategies_container/main.py`
- Create: `strategies_container/builder.py`
- Create: `strategies_container/requirements.txt`

**Step 1: Create the directory and empty init**

```bash
mkdir -p strategies_container
touch strategies_container/__init__.py
```

**Step 2: Create `strategies_container/requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-dotenv==1.0.1
```

**Step 3: Create `strategies_container/builder.py`**

```python
"""Docker build and push logic for the build agent."""
import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REGISTRY_URL = os.environ.get("DO_REGISTRY_URL", "registry.digitalocean.com/oculus-strategies")
REGISTRY_TOKEN = os.environ.get("DO_REGISTRY_TOKEN", "")


def sanitize_tag(name: str) -> str:
    tag = name.lower().replace("_", "-").replace(" ", "-")
    tag = re.sub(r"[^a-z0-9-]", "", tag)
    tag = tag.strip("-")
    return tag[:128] or "strategy"


async def _run(cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 1, "", f"Command timed out after {timeout}s"


async def build_and_push(
    strategy_name: str,
    version: int,
    files: dict[str, str],
) -> dict:
    """
    Write strategy files to a temp dir, docker build, docker push.
    Returns {"success": bool, "image_tag": str|None, "error": str|None}.
    """
    tag_name = sanitize_tag(strategy_name)
    image_tag = f"{REGISTRY_URL}/{tag_name}:{version}"
    latest_tag = f"{REGISTRY_URL}/{tag_name}:latest"

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write all files
        for filename, content in files.items():
            if not filename or "/" in filename or "\\" in filename:
                continue
            (Path(tmp_dir) / filename).write_text(content, encoding="utf-8")

        # Login to registry
        if REGISTRY_TOKEN:
            rc, _, err = await _run(
                f"echo '{REGISTRY_TOKEN}' | docker login registry.digitalocean.com --username unused --password-stdin",
                timeout=60,
            )
            if rc != 0:
                return {"success": False, "image_tag": None, "error": f"Registry login failed: {err[:500]}"}

        # Build
        logger.info("Building %s", image_tag)
        rc, _, err = await _run(
            f"docker build -t {image_tag} -t {latest_tag} {tmp_dir}",
            timeout=600,
        )
        if rc != 0:
            return {"success": False, "image_tag": None, "error": f"docker build failed (exit {rc}): {err[:2000]}"}

        # Push versioned tag
        rc, _, err = await _run(f"docker push {image_tag}", timeout=300)
        if rc != 0:
            return {"success": False, "image_tag": None, "error": f"docker push failed (exit {rc}): {err[:2000]}"}

        # Push latest (best effort — don't fail if this breaks)
        await _run(f"docker push {latest_tag}", timeout=300)

    logger.info("Built and pushed: %s", image_tag)
    return {"success": True, "image_tag": image_tag, "error": None}
```

**Step 4: Create `strategies_container/main.py`**

```python
"""Build agent FastAPI service.

Receives strategy files from the Oculus backend and runs
docker build + push on this Droplet where Docker is available.
"""
import logging
import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import builder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Strategies Container Build Agent")

_API_KEY = os.environ.get("BUILD_API_KEY", "")


def _check_key(key: str) -> None:
    if not _API_KEY:
        raise HTTPException(status_code=500, detail="BUILD_API_KEY not configured on agent")
    if key != _API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


class BuildRequest(BaseModel):
    strategy_name: str
    version: int
    files: dict[str, str]


class BuildResponse(BaseModel):
    success: bool
    image_tag: Optional[str] = None
    error: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/build", response_model=BuildResponse)
async def build(
    request: BuildRequest,
    x_build_api_key: str = Header(..., alias="X-Build-Api-Key"),
):
    _check_key(x_build_api_key)

    if not request.files:
        raise HTTPException(status_code=422, detail="files dict is empty")

    logger.info(
        "Build request: strategy=%s version=%d files=%d",
        request.strategy_name,
        request.version,
        len(request.files),
    )

    result = await builder.build_and_push(
        strategy_name=request.strategy_name,
        version=request.version,
        files=request.files,
    )
    return BuildResponse(**result)
```

**Step 5: Commit**

```bash
git add strategies_container/
git commit -m "feat: add strategies_container build agent service (main.py + builder.py)"
```

---

## Task 2: Add tests for the build agent

**Files:**
- Create: `strategies_container/tests/__init__.py`
- Create: `strategies_container/tests/test_builder.py`
- Create: `strategies_container/tests/test_main.py`

**Step 1: Write the failing tests**

`strategies_container/tests/test_builder.py`:
```python
"""Tests for builder.py — mock _run so no real Docker needed."""
import pytest
from unittest.mock import AsyncMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import builder


@pytest.mark.asyncio
async def test_sanitize_tag():
    assert builder.sanitize_tag("My Strategy_V2") == "my-strategy-v2"
    assert builder.sanitize_tag("---test---") == "test"
    assert builder.sanitize_tag("Strategy@#$") == "strategy"


@pytest.mark.asyncio
@patch("builder._run", new_callable=AsyncMock)
@patch.dict(os.environ, {"DO_REGISTRY_TOKEN": "", "DO_REGISTRY_URL": "registry.digitalocean.com/test"})
async def test_build_and_push_success(mock_run):
    mock_run.return_value = (0, "sha256:abc", "")
    result = await builder.build_and_push(
        strategy_name="test-strat",
        version=2,
        files={"Dockerfile": "FROM python:3.11-slim\n"},
    )
    assert result["success"] is True
    assert result["image_tag"] == "registry.digitalocean.com/test/test-strat:2"
    assert result["error"] is None


@pytest.mark.asyncio
@patch("builder._run", new_callable=AsyncMock)
@patch.dict(os.environ, {"DO_REGISTRY_TOKEN": "", "DO_REGISTRY_URL": "registry.digitalocean.com/test"})
async def test_build_failure_returns_error(mock_run):
    mock_run.return_value = (1, "", "COPY failed: file not found")
    result = await builder.build_and_push(
        strategy_name="bad-strat",
        version=1,
        files={"Dockerfile": "FROM scratch\n"},
    )
    assert result["success"] is False
    assert result["image_tag"] is None
    assert "docker build failed" in result["error"]


@pytest.mark.asyncio
@patch("builder._run", new_callable=AsyncMock)
@patch.dict(os.environ, {"DO_REGISTRY_TOKEN": "mytoken", "DO_REGISTRY_URL": "registry.digitalocean.com/test"})
async def test_login_failure_short_circuits(mock_run):
    # First call is the login — make it fail
    mock_run.side_effect = [(1, "", "unauthorized"), (0, "", "")]
    result = await builder.build_and_push(
        strategy_name="strat",
        version=1,
        files={"Dockerfile": "FROM python:3.11-slim\n"},
    )
    assert result["success"] is False
    assert "Registry login failed" in result["error"]
```

`strategies_container/tests/test_main.py`:
```python
"""Tests for main.py FastAPI endpoints."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["BUILD_API_KEY"] = "test-secret"

import main

client = TestClient(main.app)

VALID_PAYLOAD = {
    "strategy_name": "momentum-spy",
    "version": 1,
    "files": {
        "Dockerfile": "FROM python:3.11-slim\nCMD [\"python\", \"main.py\"]",
        "strategy.py": "class TradingStrategy: pass",
    },
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_build_rejects_missing_key():
    response = client.post("/build", json=VALID_PAYLOAD)
    assert response.status_code == 422  # missing header


def test_build_rejects_wrong_key():
    response = client.post(
        "/build",
        json=VALID_PAYLOAD,
        headers={"X-Build-Api-Key": "wrong"},
    )
    assert response.status_code == 401


@patch("main.builder.build_and_push", new_callable=AsyncMock)
def test_build_success(mock_build):
    mock_build.return_value = {
        "success": True,
        "image_tag": "registry.digitalocean.com/test/momentum-spy:1",
        "error": None,
    }
    response = client.post(
        "/build",
        json=VALID_PAYLOAD,
        headers={"X-Build-Api-Key": "test-secret"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["image_tag"] == "registry.digitalocean.com/test/momentum-spy:1"
    mock_build.assert_awaited_once_with(
        strategy_name="momentum-spy",
        version=1,
        files=VALID_PAYLOAD["files"],
    )


@patch("main.builder.build_and_push", new_callable=AsyncMock)
def test_build_failure_propagated(mock_build):
    mock_build.return_value = {
        "success": False,
        "image_tag": None,
        "error": "docker build failed: syntax error",
    }
    response = client.post(
        "/build",
        json=VALID_PAYLOAD,
        headers={"X-Build-Api-Key": "test-secret"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "docker build failed" in data["error"]
```

**Step 2: Install test deps and run — expect failures**

```bash
cd strategies_container
pip install fastapi uvicorn pytest pytest-asyncio httpx
pytest tests/ -v
```

Expected: tests fail because `strategies_container/tests/` has no `__init__.py` yet and `builder` module may have import issues.

**Step 3: Create `strategies_container/tests/__init__.py`**

```bash
touch strategies_container/tests/__init__.py
```

**Step 4: Run tests — expect all green**

```bash
cd strategies_container
pytest tests/ -v
```

Expected output:
```
test_builder.py::test_sanitize_tag PASSED
test_builder.py::test_build_and_push_success PASSED
test_builder.py::test_build_failure_returns_error PASSED
test_builder.py::test_login_failure_short_circuits PASSED
test_main.py::test_health PASSED
test_main.py::test_build_rejects_missing_key PASSED
test_main.py::test_build_rejects_wrong_key PASSED
test_main.py::test_build_success PASSED
test_main.py::test_build_failure_propagated PASSED
9 passed
```

**Step 5: Commit**

```bash
git add strategies_container/tests/
git commit -m "test: add build agent unit tests"
```

---

## Task 3: Add deploy script for the Droplet

**Files:**
- Create: `strategies_container/deploy.sh`

**Step 1: Create `strategies_container/deploy.sh`**

This script is run once on the `signalSynk-Strategies` Droplet as root. It installs the service and starts it.

```bash
#!/bin/bash
# Deploy the Strategies Container Build Agent on the signalSynk-Strategies Droplet.
# Run as root: bash deploy.sh
set -e

DEPLOY_DIR="/opt/strategies-build-agent"
SERVICE_NAME="strategies-build-agent"

echo "=== Deploying Strategies Container Build Agent ==="

# Install Python deps
echo "Installing Python dependencies..."
pip3 install fastapi==0.115.0 "uvicorn[standard]==0.30.6" python-dotenv==1.0.1

# Create deploy directory and copy service files
echo "Copying files to $DEPLOY_DIR..."
mkdir -p "$DEPLOY_DIR"
cp main.py builder.py requirements.txt "$DEPLOY_DIR/"

# Create .env file placeholder if it doesn't exist
if [ ! -f "$DEPLOY_DIR/.env" ]; then
    cat > "$DEPLOY_DIR/.env" << 'ENVFILE'
BUILD_API_KEY=REPLACE_WITH_SHARED_SECRET
DO_REGISTRY_URL=registry.digitalocean.com/oculus-strategies
DO_REGISTRY_TOKEN=REPLACE_WITH_DO_TOKEN
ENVFILE
    echo "Created $DEPLOY_DIR/.env — EDIT THIS FILE with real values before starting the service"
fi

# Create systemd unit
echo "Creating systemd unit..."
cat > "/etc/systemd/system/$SERVICE_NAME.service" << UNIT
[Unit]
Description=Strategies Container Build Agent
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=$DEPLOY_DIR
EnvironmentFile=$DEPLOY_DIR/.env
ExecStart=/usr/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8088
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo ""
echo "=== Done ==="
echo "Service status:"
systemctl status "$SERVICE_NAME" --no-pager
echo ""
echo "IMPORTANT: Edit $DEPLOY_DIR/.env with real credentials, then: systemctl restart $SERVICE_NAME"
echo "Test with: curl http://localhost:8088/health"
```

**Step 2: Make it executable and commit**

```bash
chmod +x strategies_container/deploy.sh
git add strategies_container/deploy.sh
git commit -m "feat: add Droplet deploy script for build agent"
```

---

## Task 4: Add httpx + new config vars to backend

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/config.py`

**Step 1: Add `httpx` to backend requirements**

In `backend/requirements.txt`, add after the `anthropic` line:
```
httpx>=0.27.0
```

**Step 2: Add build agent config to `backend/app/config.py`**

In the `Settings` class, after the `DO_REGISTRY_TOKEN` line (line 83), add:

```python
    # Build Agent (signalSynk-Strategies Droplet)
    BUILD_AGENT_URL: str = ""          # e.g. http://<droplet-ip>:8088
    BUILD_AGENT_API_KEY: str = ""      # Shared secret matching Droplet BUILD_API_KEY
```

**Step 3: Verify config loads without error**

```bash
cd backend
python3 -c "from app.config import settings; print('BUILD_AGENT_URL:', repr(settings.BUILD_AGENT_URL))"
```

Expected: `BUILD_AGENT_URL: ''`

**Step 4: Commit**

```bash
git add backend/requirements.txt backend/app/config.py
git commit -m "feat: add BUILD_AGENT_URL and BUILD_AGENT_API_KEY config + httpx dep"
```

---

## Task 5: Add `remote_build_and_push` to DockerBuilder

**Files:**
- Modify: `backend/app/services/docker_builder.py`
- Modify: `backend/tests/test_docker_builder.py`

**Step 1: Write the failing test first**

In `backend/tests/test_docker_builder.py`, add after the last test:

```python
@pytest.mark.asyncio
async def test_remote_build_and_push_success(test_db, test_strategy, test_build):
    """Test remote_build_and_push calls the build agent and updates strategy."""
    import respx
    import httpx

    builder = DockerBuilder(test_db)

    files = {"strategy.py": "class TradingStrategy: pass", "Dockerfile": "FROM python:3.11-slim"}

    with patch("app.services.docker_builder.settings") as mock_settings:
        mock_settings.BUILD_AGENT_URL = "http://fake-droplet:8088"
        mock_settings.BUILD_AGENT_API_KEY = "test-key"
        mock_settings.DO_REGISTRY_URL = "registry.digitalocean.com/oculus-strategies"

        with respx.mock:
            respx.post("http://fake-droplet:8088/build").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "success": True,
                        "image_tag": "registry.digitalocean.com/oculus-strategies/test-strategy:1",
                        "error": None,
                    },
                )
            )
            result = await builder.remote_build_and_push(test_strategy, test_build, files)

    assert result is True
    assert test_strategy.docker_image_url == "registry.digitalocean.com/oculus-strategies/test-strategy:1"
    assert test_build.status == "complete"


@pytest.mark.asyncio
async def test_remote_build_and_push_agent_failure(test_db, test_strategy, test_build):
    """Test remote_build_and_push handles build agent failure response."""
    import respx
    import httpx

    builder = DockerBuilder(test_db)

    with patch("app.services.docker_builder.settings") as mock_settings:
        mock_settings.BUILD_AGENT_URL = "http://fake-droplet:8088"
        mock_settings.BUILD_AGENT_API_KEY = "test-key"
        mock_settings.DO_REGISTRY_URL = "registry.digitalocean.com/oculus-strategies"

        with respx.mock:
            respx.post("http://fake-droplet:8088/build").mock(
                return_value=httpx.Response(
                    200,
                    json={"success": False, "image_tag": None, "error": "docker build failed"},
                )
            )
            result = await builder.remote_build_and_push(test_strategy, test_build, {})

    assert result is False


@pytest.mark.asyncio
async def test_remote_build_and_push_no_url(test_db, test_strategy, test_build):
    """Test remote_build_and_push returns False when BUILD_AGENT_URL is not set."""
    builder = DockerBuilder(test_db)

    with patch("app.services.docker_builder.settings") as mock_settings:
        mock_settings.BUILD_AGENT_URL = ""
        mock_settings.BUILD_AGENT_API_KEY = ""

        result = await builder.remote_build_and_push(test_strategy, test_build, {})

    assert result is False
```

**Step 2: Run the tests — expect ImportError on `respx`**

```bash
cd backend
pip install respx
pytest tests/test_docker_builder.py -v -k "remote"
```

Expected: FAIL — `remote_build_and_push` doesn't exist yet.

**Step 3: Implement `remote_build_and_push` in `docker_builder.py`**

Add this method to the `DockerBuilder` class, after the `build_and_push` method (line ~273):

```python
    async def remote_build_and_push(
        self,
        strategy: Strategy,
        build: StrategyBuild,
        files: dict,
    ) -> bool:
        """
        Send strategy files to the remote build agent on the signalSynk-Strategies
        Droplet for Docker build and push.

        Args:
            strategy: Strategy model instance
            build: StrategyBuild model instance
            files: Dict of {filename: content} — all 10 strategy files from
                   BuildIteration.strategy_files

        Returns:
            True if the image was built and pushed successfully, False otherwise
        """
        import httpx

        if not settings.BUILD_AGENT_URL:
            logger.error("BUILD_AGENT_URL not configured — cannot build remotely")
            build.logs = f"{build.logs or ''}\n\nRemote build agent not configured (BUILD_AGENT_URL missing)"
            await self.db.commit()
            return False

        if not files:
            logger.error("No strategy files provided for remote build")
            build.logs = f"{build.logs or ''}\n\nNo strategy files available for Docker build"
            await self.db.commit()
            return False

        version = strategy.version or 1
        payload = {
            "strategy_name": strategy.name,
            "version": version,
            "files": files,
        }

        logger.info(
            "Sending build request to agent: strategy=%s version=%d files=%d",
            strategy.name,
            version,
            len(files),
        )

        try:
            async with httpx.AsyncClient(timeout=900.0) as client:
                response = await client.post(
                    f"{settings.BUILD_AGENT_URL}/build",
                    json=payload,
                    headers={"X-Build-Api-Key": settings.BUILD_AGENT_API_KEY},
                )

            if response.status_code != 200:
                error_msg = f"Build agent HTTP {response.status_code}: {response.text[:500]}"
                logger.error(error_msg)
                build.logs = f"{build.logs or ''}\n\n{error_msg}"
                await self.db.commit()
                return False

            result = response.json()

            if not result.get("success"):
                error_msg = f"Remote build failed: {result.get('error', 'unknown error')}"
                logger.error(error_msg)
                build.logs = f"{build.logs or ''}\n\n{error_msg}"
                await self.db.commit()
                return False

            image_tag = result["image_tag"]
            strategy.docker_registry = self.registry_url
            strategy.docker_image_url = image_tag
            build.status = "complete"
            build.completed_at = datetime.utcnow()
            await self.db.commit()

            logger.info("Remote build successful: %s", image_tag)
            return True

        except Exception as e:
            error_msg = f"Remote build agent error: {str(e)}"
            logger.error(error_msg)
            build.logs = f"{build.logs or ''}\n\n{error_msg}"
            await self.db.commit()
            return False
```

**Step 4: Run tests — expect all green**

```bash
cd backend
pytest tests/test_docker_builder.py -v
```

Expected: all tests pass including the 3 new `remote` tests.

**Step 5: Commit**

```bash
git add backend/app/services/docker_builder.py backend/tests/test_docker_builder.py
git commit -m "feat: add remote_build_and_push to DockerBuilder (delegates to build agent)"
```

---

## Task 6: Replace Docker sections in builds.py

**Files:**
- Modify: `backend/app/routers/builds.py:1353-1377` (Path 1 — target met early)
- Modify: `backend/app/routers/builds.py:1620-1644` (Path 2 — best iteration after all iters)

Both sections replace the same 3-step pattern (`build_image → test_container → push_image`) with a single `remote_build_and_push` call.

**Step 1: Replace Path 1 Docker section (lines 1353–1377)**

Find this exact block:
```python
                    # --- Docker build, test, and push ---
                    strat_result = await bg_db.execute(
                        select(Strategy).where(Strategy.uuid == strategy_id)
                    )
                    strategy_obj = strat_result.scalar_one_or_none()
                    docker_builder = DockerBuilder(bg_db)

                    # Build the Docker image
                    image_tag = await docker_builder.build_image(strategy_obj, build, strategy_output_dir)

                    if image_tag:
                        # Test the container
                        test_result = await docker_builder.test_container(image_tag, strategy_output_dir)

                        if test_result["passed"]:
                            # Push to Digital Ocean Container Registry
                            push_success = await docker_builder.push_image(image_tag, build)
                            if push_success:
                                iteration_logs.append(f"Iteration {iteration}: Docker image built, tested, and pushed")
                            else:
                                iteration_logs.append(f"Iteration {iteration}: Docker push failed")
                        else:
                            iteration_logs.append(f"Iteration {iteration}: Container test failed: {test_result.get('error', 'Unknown')}")
                    else:
                        iteration_logs.append(f"Iteration {iteration}: Docker build failed")
```

Replace with:
```python
                    # --- Remote Docker build and push ---
                    strat_result = await bg_db.execute(
                        select(Strategy).where(Strategy.uuid == strategy_id)
                    )
                    strategy_obj = strat_result.scalar_one_or_none()
                    docker_builder = DockerBuilder(bg_db)

                    strategy_files = (
                        training_results.get("strategy_files") or {}
                        if isinstance(training_results, dict)
                        else {}
                    )
                    push_success = await docker_builder.remote_build_and_push(
                        strategy_obj, build, strategy_files
                    )
                    if push_success:
                        iteration_logs.append(f"Iteration {iteration}: Docker image built and pushed remotely")
                    else:
                        iteration_logs.append(f"Iteration {iteration}: Remote Docker build failed")
```

**Step 2: Replace Path 2 Docker section (lines 1620–1644)**

Find this exact block:
```python
                    # --- Docker build, test, and push ---
                    strat_result = await bg_db.execute(
                        select(Strategy).where(Strategy.uuid == strategy_id)
                    )
                    strategy_obj = strat_result.scalar_one_or_none()
                    docker_builder = DockerBuilder(bg_db)

                    # Build the Docker image
                    image_tag = await docker_builder.build_image(strategy_obj, build, strategy_output_dir)

                    if image_tag:
                        # Test the container
                        test_result = await docker_builder.test_container(image_tag, strategy_output_dir)

                        if test_result["passed"]:
                            # Push to Docker Hub
                            push_success = await docker_builder.push_image(image_tag, build)
                            if push_success:
                                iteration_logs.append("Docker image built, tested, and pushed for best iteration")
                            else:
                                iteration_logs.append("Docker push failed for best iteration")
                        else:
                            iteration_logs.append(f"Container test failed for best iteration: {test_result.get('error', 'Unknown')}")
                    else:
                        iteration_logs.append("Docker build failed for best iteration")
```

Replace with:
```python
                    # --- Remote Docker build and push ---
                    strat_result = await bg_db.execute(
                        select(Strategy).where(Strategy.uuid == strategy_id)
                    )
                    strategy_obj = strat_result.scalar_one_or_none()
                    docker_builder = DockerBuilder(bg_db)

                    strategy_files = best_iter.strategy_files or {}
                    push_success = await docker_builder.remote_build_and_push(
                        strategy_obj, build, strategy_files
                    )
                    if push_success:
                        iteration_logs.append("Docker image built and pushed remotely for best iteration")
                    else:
                        iteration_logs.append("Remote Docker build failed for best iteration")
```

**Step 3: Verify backend still imports cleanly**

```bash
cd backend
python3 -c "from app.routers.builds import router; print('OK')"
```

Expected: `OK`

**Step 4: Run existing backend tests**

```bash
cd backend
pytest tests/ -v --ignore=tests/test_docker_builder.py -x
```

Expected: all existing tests pass.

**Step 5: Commit**

```bash
git add backend/app/routers/builds.py
git commit -m "feat: replace local docker commands with remote_build_and_push in builds.py"
```

---

## Task 7: Create the recovery script

**Files:**
- Create: `strategies_container/recover_builds.py`

This script runs once, standalone, against the production database to push all missing images. It uses raw `asyncpg` (no SQLAlchemy models needed) so it can run anywhere with Python 3.11+.

**Step 1: Create `strategies_container/recover_builds.py`**

```python
#!/usr/bin/env python3
"""
Recovery script: push Docker images for all completed strategy builds
that are missing from the container registry.

Usage:
    DATABASE_URL=postgresql://... \
    BUILD_AGENT_URL=http://<droplet-ip>:8088 \
    BUILD_AGENT_API_KEY=<secret> \
    python3 recover_builds.py

The script assigns versions by ordering each strategy's completed builds
by completed_at DESC, exactly matching the admin version-history algorithm
in backend/app/routers/admin.py.
"""
import asyncio
import os
import sys

import asyncpg
import httpx


DATABASE_URL = os.environ.get("DATABASE_URL", "")
BUILD_AGENT_URL = os.environ.get("BUILD_AGENT_URL", "")
BUILD_AGENT_API_KEY = os.environ.get("BUILD_AGENT_API_KEY", "")

# asyncpg uses 'host' kwarg style, not URL sslmode param
def _pg_dsn(url: str) -> str:
    """Convert postgresql+asyncpg:// or postgresql:// to a bare DSN asyncpg accepts."""
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    # Replace sslmode= with ssl= for asyncpg compatibility
    url = url.replace("sslmode=require", "ssl=require")
    return url


async def main() -> None:
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL env var required", file=sys.stderr)
        sys.exit(1)
    if not BUILD_AGENT_URL:
        print("ERROR: BUILD_AGENT_URL env var required", file=sys.stderr)
        sys.exit(1)
    if not BUILD_AGENT_API_KEY:
        print("ERROR: BUILD_AGENT_API_KEY env var required", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to database...")
    conn = await asyncpg.connect(_pg_dsn(DATABASE_URL))

    print(f"Build agent: {BUILD_AGENT_URL}")
    print()

    # 1. Fetch all complete strategies
    strategies = await conn.fetch(
        """
        SELECT uuid, name, version, docker_image_url
        FROM strategies
        WHERE status = 'complete'
        ORDER BY name
        """
    )

    print(f"Found {len(strategies)} complete strategies\n")

    total_pushed = 0
    total_skipped = 0
    total_failed = 0

    for strat in strategies:
        strat_id = strat["uuid"]
        strat_name = strat["name"]
        current_version = strat["version"] or 1

        # 2. Fetch all completed builds for this strategy ordered by completed_at DESC
        #    (matches admin.py version reconstruction logic exactly)
        builds = await conn.fetch(
            """
            SELECT uuid, completed_at
            FROM strategy_builds
            WHERE strategy_id = $1
              AND status = 'complete'
            ORDER BY completed_at DESC NULLS LAST
            """,
            strat_id,
        )

        if not builds:
            continue

        print(f"Strategy: {strat_name!r} (current v{current_version}) — {len(builds)} completed build(s)")

        latest_image_tag = None

        for i, build in enumerate(builds):
            build_id = build["uuid"]
            version = current_version - i

            if version < 1:
                print(f"  [{build_id[:8]}] version {version} < 1 — skipping")
                total_skipped += 1
                continue

            # 3. Find the best BuildIteration with strategy_files populated
            #    (take the highest iteration_number with non-null strategy_files)
            iteration = await conn.fetchrow(
                """
                SELECT strategy_files
                FROM build_iterations
                WHERE build_id = $1
                  AND status = 'complete'
                  AND strategy_files IS NOT NULL
                ORDER BY iteration_number DESC
                LIMIT 1
                """,
                build_id,
            )

            if not iteration or not iteration["strategy_files"]:
                print(f"  [{build_id[:8]}] v{version}: no strategy_files — skipping")
                total_skipped += 1
                continue

            import json
            files = iteration["strategy_files"]
            if isinstance(files, str):
                files = json.loads(files)

            if not isinstance(files, dict) or len(files) < 4:
                print(f"  [{build_id[:8]}] v{version}: strategy_files malformed — skipping")
                total_skipped += 1
                continue

            print(f"  [{build_id[:8]}] v{version}: pushing {len(files)} files...", end=" ", flush=True)

            # 4. Call the build agent
            try:
                async with httpx.AsyncClient(timeout=900.0) as client:
                    response = await client.post(
                        f"{BUILD_AGENT_URL}/build",
                        json={
                            "strategy_name": strat_name,
                            "version": version,
                            "files": files,
                        },
                        headers={"X-Build-Api-Key": BUILD_AGENT_API_KEY},
                    )

                result = response.json()

                if result.get("success"):
                    image_tag = result["image_tag"]
                    print(f"OK → {image_tag}")
                    total_pushed += 1
                    # Track the latest (i=0 is newest build = current version)
                    if i == 0:
                        latest_image_tag = image_tag
                else:
                    print(f"FAILED: {result.get('error', 'unknown')}")
                    total_failed += 1

            except Exception as e:
                print(f"ERROR: {e}")
                total_failed += 1

        # 5. Update strategy.docker_image_url with the latest successfully pushed tag
        if latest_image_tag:
            await conn.execute(
                """
                UPDATE strategies
                SET docker_image_url = $1,
                    docker_registry = 'registry.digitalocean.com/oculus-strategies'
                WHERE uuid = $2
                """,
                latest_image_tag,
                strat_id,
            )
            print(f"  → Updated strategy.docker_image_url = {latest_image_tag}")

    await conn.close()

    print()
    print("=" * 60)
    print(f"Recovery complete:")
    print(f"  {total_pushed} images pushed")
    print(f"  {total_skipped} builds skipped (no strategy_files or version < 1)")
    print(f"  {total_failed} builds failed")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Install deps needed to run the recovery script**

```bash
pip install asyncpg httpx
```

**Step 3: Test the script connects (dry run against local DB if available)**

With real credentials set as env vars, first test connectivity:
```bash
cd strategies_container
DATABASE_URL="postgresql://..." \
BUILD_AGENT_URL="http://<droplet-ip>:8088" \
BUILD_AGENT_API_KEY="<secret>" \
python3 -c "
import asyncio, asyncpg, os
async def test():
    conn = await asyncpg.connect(os.environ['DATABASE_URL'].replace('postgresql+asyncpg://', 'postgresql://'))
    rows = await conn.fetch('SELECT count(*) FROM strategies WHERE status=\'complete\'')
    print('Complete strategies:', rows[0][0])
    await conn.close()
asyncio.run(test())
"
```

Expected: prints the count of complete strategies with no errors.

**Step 4: Commit**

```bash
git add strategies_container/recover_builds.py
git commit -m "feat: add recovery script to push all missing strategy build images"
```

---

## Task 8: Deployment checklist

This task is manual steps — no code to write.

**Step 1: Copy agent files to the Droplet**

From your local machine:
```bash
scp strategies_container/main.py \
    strategies_container/builder.py \
    strategies_container/requirements.txt \
    strategies_container/deploy.sh \
    root@<droplet-ip>:/tmp/build-agent/
```

**Step 2: Run the deploy script on the Droplet**

SSH into the Droplet:
```bash
ssh root@<droplet-ip>
cd /tmp/build-agent
bash deploy.sh
```

**Step 3: Edit the `.env` file on the Droplet with real credentials**

```bash
nano /opt/strategies-build-agent/.env
# Set:
#   BUILD_API_KEY=<generate a strong random secret, e.g. openssl rand -hex 32>
#   DO_REGISTRY_URL=registry.digitalocean.com/oculus-strategies
#   DO_REGISTRY_TOKEN=<your DO API token>
systemctl restart strategies-build-agent
```

**Step 4: Verify the agent is running**

```bash
curl http://localhost:8088/health
# Expected: {"status":"ok"}
```

**Step 5: Add env vars to App Platform**

In the DigitalOcean App Platform console for Oculus, add:
- `BUILD_AGENT_URL` = `http://<droplet-ip>:8088`
- `BUILD_AGENT_API_KEY` = the same secret set in the Droplet `.env`

Trigger a new App Platform deployment.

**Step 6: Run the recovery script**

From your local machine with production DB credentials:
```bash
cd strategies_container
DATABASE_URL="<production DATABASE_URL>" \
BUILD_AGENT_URL="http://<droplet-ip>:8088" \
BUILD_AGENT_API_KEY="<secret>" \
python3 recover_builds.py
```

Watch the output — every completed build gets an image pushed.

**Step 7: Verify images in the registry**

```bash
doctl registry repository list-tags oculus-strategies
```

Expected: one tag per strategy version.

---

## Commit Summary

| Commit | Content |
|--------|---------|
| `feat: add strategies_container build agent service` | Task 1 |
| `test: add build agent unit tests` | Task 2 |
| `feat: add Droplet deploy script for build agent` | Task 3 |
| `feat: add BUILD_AGENT_URL/KEY config + httpx dep` | Task 4 |
| `feat: add remote_build_and_push to DockerBuilder` | Task 5 |
| `feat: replace local docker commands in builds.py` | Task 6 |
| `feat: add recovery script for missing strategy images` | Task 7 |
