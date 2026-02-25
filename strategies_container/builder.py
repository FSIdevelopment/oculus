"""Docker build and push logic for the build agent."""
import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


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

    Reads DO_REGISTRY_URL and DO_REGISTRY_TOKEN from the environment at call
    time (not import time) so tests can patch os.environ cleanly.
    """
    registry_url = os.environ.get("DO_REGISTRY_URL", "registry.digitalocean.com/oculus-strategies")
    registry_token = os.environ.get("DO_REGISTRY_TOKEN", "")

    tag_name = sanitize_tag(strategy_name)
    image_tag = f"{registry_url}/{tag_name}:{version}"
    latest_tag = f"{registry_url}/{tag_name}:latest"

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write all files — skip filenames with path separators to prevent traversal
        for filename, content in files.items():
            if not filename or "/" in filename or "\\" in filename:
                continue
            (Path(tmp_dir) / filename).write_text(content, encoding="utf-8")

        # Login to registry
        if registry_token:
            rc, _, err = await _run(
                f"echo '{registry_token}' | docker login registry.digitalocean.com --username unused --password-stdin",
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
