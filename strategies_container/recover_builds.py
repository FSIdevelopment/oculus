#!/usr/bin/env python3
"""
Recovery script: push Docker images for all completed strategy builds
that are missing from the container registry.

Usage:
    DATABASE_URL="postgresql://user:pass@host/db" \
    BUILD_AGENT_URL="http://<droplet-ip>:8088" \
    BUILD_AGENT_API_KEY="<secret>" \
    python3 recover_builds.py

Version assignment mirrors admin.py:577-591 exactly:
  - All completed builds for a strategy ordered by completed_at DESC
  - Build at index 0 → strategy.version (most recent)
  - Build at index i → strategy.version - i
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
import httpx


DATABASE_URL = os.environ.get("DATABASE_URL", "")
BUILD_AGENT_URL = os.environ.get("BUILD_AGENT_URL", "")
BUILD_AGENT_API_KEY = os.environ.get("BUILD_AGENT_API_KEY", "")

# Apply the same safeguard as builds.py: always use the canonical template
# for main.py and requirements.txt, regardless of what's stored in the DB.
# The DB's strategy_files were saved before the safeguard copy in builds.py ran,
# so they may contain stub versions of these files.
_TEMPLATE_DIR = Path(__file__).parent.parent / "backend" / "templates" / "strategy_container"
_CANONICAL_MAIN_PY = (_TEMPLATE_DIR / "main.py").read_text(encoding="utf-8")
_CANONICAL_REQUIREMENTS_TXT = (_TEMPLATE_DIR / "requirements.txt").read_text(encoding="utf-8")


def _apply_safeguards(files: dict) -> dict:
    """Override main.py and requirements.txt with canonical templates."""
    files["main.py"] = _CANONICAL_MAIN_PY
    files["requirements.txt"] = _CANONICAL_REQUIREMENTS_TXT
    return files


def _pg_dsn(url: str) -> tuple:
    """Normalise URL to a form asyncpg accepts. Returns (dsn, ssl_required)."""
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    ssl_required = False
    for param in ("sslmode=require", "ssl=require"):
        if param in url:
            ssl_required = True
            url = url.replace("?" + param, "").replace("&" + param, "").replace(param, "")
            url = url.rstrip("?&")
    return url, ssl_required


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

    # Verify build agent is reachable before touching the DB
    print(f"Checking build agent at {BUILD_AGENT_URL} ...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{BUILD_AGENT_URL}/health")
            if r.status_code != 200 or r.json().get("status") != "ok":
                print(f"ERROR: Build agent health check failed: {r.text}", file=sys.stderr)
                sys.exit(1)
        print("Build agent: OK\n")
    except Exception as e:
        print(f"ERROR: Cannot reach build agent: {e}", file=sys.stderr)
        sys.exit(1)

    print("Connecting to database...")
    dsn, ssl_required = _pg_dsn(DATABASE_URL)
    conn = await asyncpg.connect(dsn, ssl="require" if ssl_required else None)
    print("Database: connected\n")

    # Fetch all complete strategies
    strategies = await conn.fetch(
        """
        SELECT uuid, name, version, docker_image_url
        FROM strategies
        WHERE status = 'complete'
        ORDER BY name
        """
    )

    print(f"Found {len(strategies)} complete strategies")
    print("=" * 60)

    total_pushed = 0
    total_skipped = 0
    total_failed = 0

    for strat in strategies:
        strat_id = strat["uuid"]
        strat_name = strat["name"]
        current_version = strat["version"] or 1

        # All completed builds ordered by completed_at DESC — matches admin.py exactly
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

        print(f"\nStrategy: {strat_name!r}  current_version={current_version}  builds={len(builds)}")

        latest_pushed_tag = None

        for i, build in enumerate(builds):
            build_id = build["uuid"]
            version = current_version - i

            if version < 1:
                print(f"  [{build_id[:8]}] v{version} < 1 — skipping")
                total_skipped += 1
                continue

            # Find the best BuildIteration: highest iteration_number with strategy_files
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

            files = iteration["strategy_files"]
            if isinstance(files, str):
                files = json.loads(files)

            if not isinstance(files, dict) or len(files) < 4:
                print(f"  [{build_id[:8]}] v{version}: strategy_files malformed ({type(files).__name__}, len={len(files) if isinstance(files, dict) else '?'}) — skipping")
                total_skipped += 1
                continue

            # Apply the same safeguard as builds.py: override main.py and
            # requirements.txt with the canonical templates before pushing.
            files = _apply_safeguards(files)

            print(f"  [{build_id[:8]}] v{version}: pushing {len(files)} files...", end=" ", flush=True)

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
                    if i == 0:
                        latest_pushed_tag = image_tag
                else:
                    print(f"FAILED: {result.get('error', 'unknown')}")
                    total_failed += 1

            except Exception as e:
                print(f"ERROR: {e}")
                total_failed += 1

        # Update strategy.docker_image_url with the most recent successfully pushed tag
        if latest_pushed_tag:
            await conn.execute(
                """
                UPDATE strategies
                SET docker_image_url = $1,
                    docker_registry  = 'registry.digitalocean.com/oculus-strategies'
                WHERE uuid = $2
                """,
                latest_pushed_tag,
                strat_id,
            )
            print(f"  → DB updated: docker_image_url = {latest_pushed_tag}")

    await conn.close()

    print()
    print("=" * 60)
    print("Recovery complete:")
    print(f"  {total_pushed:>4} images pushed")
    print(f"  {total_skipped:>4} builds skipped (no strategy_files or version < 1)")
    print(f"  {total_failed:>4} builds failed")


if __name__ == "__main__":
    asyncio.run(main())
