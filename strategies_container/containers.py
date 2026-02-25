"""Docker container lifecycle management for strategy containers.

Provides start / stop / restart / logs / status / list operations
for SignalSynk strategy containers running on this droplet.

Container naming convention: ``signalsynk-strategy-{strategy_id}``
All managed containers carry the label ``signalsynk.managed=true``.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

_CONTAINER_PREFIX = "signalsynk-strategy-"


def container_name(strategy_id: int) -> str:
    """Return the Docker container name for a strategy ID."""
    return f"{_CONTAINER_PREFIX}{strategy_id}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _run(cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    """Execute a shell command.  Returns (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 1, "", f"Command timed out after {timeout}s"


async def _run_args(args: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Execute a command via exec (no shell) — safe for untrusted values."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 1, "", f"Command timed out after {timeout}s"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def start(
    strategy_id: int,
    image_name: str,
    env_vars: dict[str, str],
) -> dict:
    """Start a strategy container.

    Stops and removes any existing container with the same name first,
    then starts a fresh container from *image_name* with the supplied
    environment variables.

    Returns::

        {"success": bool, "container_id": str | None, "error": str | None}
    """
    name = container_name(strategy_id)

    # Remove any stale container with this name (ignore errors)
    await _run(f"docker stop --time 10 {name} 2>/dev/null; docker rm {name} 2>/dev/null")

    # Build the argument list — avoids shell-injection on arbitrary env values
    args = [
        "docker", "run", "-d",
        "--name", name,
        "--restart", "unless-stopped",
        "--label", f"signalsynk.strategy_id={strategy_id}",
        "--label", "signalsynk.managed=true",
    ]
    for k, v in env_vars.items():
        args.extend(["-e", f"{k}={v}"])
    args.append(image_name)

    rc, stdout, stderr = await _run_args(args, timeout=60)
    if rc != 0:
        logger.error("docker run failed for strategy %d: %s", strategy_id, stderr[:500])
        return {
            "success": False,
            "container_id": None,
            "error": f"docker run failed: {stderr[:500]}",
        }

    container_id = stdout.strip()
    logger.info(
        "Started container %s (id=%s) for strategy %d",
        name,
        container_id[:12] if container_id else "?",
        strategy_id,
    )
    return {"success": True, "container_id": container_id, "error": None}


async def stop(strategy_id: int) -> dict:
    """Stop and remove a strategy container.

    Returns::

        {"success": bool, "error": str | None}
    """
    name = container_name(strategy_id)
    await _run(f"docker stop --time 10 {name} 2>/dev/null")
    await _run(f"docker rm {name} 2>/dev/null")
    logger.info("Stopped container for strategy %d", strategy_id)
    return {"success": True, "error": None}


async def restart(
    strategy_id: int,
    image_name: str,
    env_vars: dict[str, str],
) -> dict:
    """Restart a strategy container.

    Stops the existing container and starts a fresh one.

    Returns::

        {"success": bool, "container_id": str | None, "error": str | None}
    """
    await stop(strategy_id)
    return await start(strategy_id, image_name, env_vars)


async def get_logs(strategy_id: int, tail: int = 100) -> dict:
    """Retrieve recent logs from a strategy container (stdout + stderr merged).

    Returns::

        {"success": bool, "logs": str | None, "error": str | None}
    """
    name = container_name(strategy_id)
    rc, stdout, stderr = await _run(
        f"docker logs --tail {tail} --timestamps {name} 2>&1",
        timeout=30,
    )
    if rc != 0:
        return {
            "success": False,
            "logs": None,
            "error": f"docker logs failed: {stderr[:500]}",
        }
    return {"success": True, "logs": stdout, "error": None}


async def get_status(strategy_id: int) -> dict:
    """Return the current Docker status of a strategy container.

    Returns::

        {
            "success": bool,
            "status": str,          # e.g. "running", "exited", "not_found"
            "container_id": str | None,
            "error": str | None,
        }
    """
    name = container_name(strategy_id)
    rc, stdout, _ = await _run(
        f"docker inspect --format '{{{{.State.Status}}}}:{{{{.Id}}}}' {name} 2>/dev/null",
        timeout=15,
    )
    if rc != 0 or not stdout.strip():
        return {"success": True, "status": "not_found", "container_id": None, "error": None}

    parts = stdout.strip().split(":", 1)
    status = parts[0]
    container_id = parts[1][:12] if len(parts) > 1 and parts[1] else None
    return {"success": True, "status": status, "container_id": container_id, "error": None}


async def list_all() -> dict:
    """List all SignalSynk-managed containers on this host.

    Returns::

        {
            "success": bool,
            "containers": [
                {
                    "container_id": str,
                    "name": str,
                    "status": str,
                    "image": str,
                },
                ...
            ],
            "error": str | None,
        }
    """
    rc, stdout, stderr = await _run(
        "docker ps -a --filter label=signalsynk.managed=true "
        "--format '{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}'",
        timeout=30,
    )
    if rc != 0:
        return {"success": False, "containers": [], "error": stderr[:500]}

    containers = []
    for line in stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            containers.append({
                "container_id": parts[0],
                "name": parts[1],
                "status": parts[2],
                "image": parts[3],
            })

    return {"success": True, "containers": containers, "error": None}


async def pull(image_ref: str) -> dict:
    """Pull a Docker image from a registry.

    Returns::

        {"success": bool, "image": str | None, "error": str | None}
    """
    logger.info("Pulling image: %s", image_ref)
    rc, stdout, stderr = await _run(
        f"docker pull {image_ref}",
        timeout=300,
    )
    if rc != 0:
        logger.error("docker pull failed for %s: %s", image_ref, stderr[:500])
        return {
            "success": False,
            "image": None,
            "error": f"docker pull failed: {stderr[:500]}",
        }
    logger.info("Pulled image: %s", image_ref)
    return {"success": True, "image": image_ref, "error": None}


async def image_exists(image_ref: str) -> dict:
    """Check whether a Docker image is present on this host.

    Returns::

        {"success": bool, "exists": bool, "error": str | None}
    """
    rc, stdout, _ = await _run(
        f"docker image inspect {image_ref} --format '{{{{.Id}}}}' 2>/dev/null",
        timeout=15,
    )
    exists = rc == 0 and bool(stdout.strip())
    return {"success": True, "exists": exists, "error": None}
