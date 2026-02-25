#!/usr/bin/env python3
"""
Repair script: fix main.py (and requirements.txt / strategy.py) in
strategy_files for all build_iterations that have the stub main.py, then
rebuild and re-push the Docker images via the build agent.

Detection: stub main.py does NOT contain "StrategyRunner".
Also patches strategy.py to add get_interval() and analyze() if missing.

Usage:
    DATABASE_URL="postgresql://user:pass@host/db" \\
    BUILD_AGENT_URL="http://<droplet-ip>:8088" \\
    BUILD_AGENT_API_KEY="<secret>" \\
    python3 repair_main_py.py [--dry-run]

Pass --dry-run to scan and report without writing anything.
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

# Template directory relative to this script
_TEMPLATE_DIR = Path(__file__).parent.parent / "backend" / "templates" / "strategy_container"
CORRECT_MAIN_PY = (_TEMPLATE_DIR / "main.py").read_text(encoding="utf-8")
CORRECT_REQUIREMENTS_TXT = (_TEMPLATE_DIR / "requirements.txt").read_text(encoding="utf-8")

# Methods to inject into strategy.py if they're missing
_GET_INTERVAL_METHOD = '''
    def get_interval(self) -> int:
        """Return signal generation interval in seconds (read from config)."""
        trading_config = self.config.get('trading', {})
        interval_str = (
            trading_config.get('signal_generation_interval')
            or trading_config.get('timeframe', '1d')
        )
        timeframe_map = {
            '30s': 30, '1m': 60, '2m': 120, '5m': 300, '10m': 600,
            '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400,
            '1d': 86400, '1D': 86400,
        }
        return timeframe_map.get(interval_str, 86400)
'''

_GET_LOOP_INTERVAL_METHOD = '''
    def get_loop_interval(self) -> int:
        """Return how often (seconds) main.py should re-run the strategy loop.

        Mapping (timeframe -> loop interval):
          1d  -> 1800s (30 min)
          4h  ->  900s (15 min)
          1h  ->  300s  (5 min)
          30m ->  300s  (5 min)
          15m ->  900s (15 min)
          10m ->  600s (10 min)
          5m  ->  300s  (5 min)
          1m  ->   60s  (1 min)
        """
        trading_config = self.config.get('trading', {})
        timeframe = trading_config.get('timeframe', '1d')
        loop_map = {
            '1d': 1800, '1D': 1800,
            '4h': 900,
            '1h': 300,
            '30m': 300, '30min': 300,
            '15m': 900, '15min': 900,
            '10m': 600, '10min': 600,
            '5m': 300, '5min': 300,
            '2m': 120, '2min': 120,
            '1m': 60, '1min': 60,
        }
        return loop_map.get(timeframe, 1800)
'''

_ANALYZE_METHOD = '''
    async def analyze(self) -> list:
        """Fetch live data, calculate indicators, and return trade signals.

        Called periodically by main.py during market hours.
        """
        from data_provider import DataProvider
        signals = []
        provider = DataProvider()

        for symbol in self.symbols:
            try:
                df = await provider.get_historical_data_async(symbol, days=365)
                if df is None or len(df) < 50:
                    continue
                indicators = self.calculate_indicators(df)
                if not indicators:
                    continue
                signal = self.generate_signal(symbol, indicators)
                if signal in ('BUY', 'SELL'):
                    signals.append({
                        'ticker': symbol,
                        'action': signal,
                        'quantity': self.default_quantity,
                        'order_type': 'MARKET',
                        'price': indicators.get('close'),
                        'indicators': {
                            k: round(float(v), 4)
                            for k, v in indicators.items()
                            if isinstance(v, (int, float)) and k != 'volume'
                        },
                    })
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error analyzing {symbol}: {e}")

        return signals
'''


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


def needs_repair(main_py_content: str) -> bool:
    """Return True if main.py needs to be replaced.

    Catches both the old stub (no WebSocket runner) and images built
    before the loop-interval fix (no get_loop_interval call).
    """
    if "StrategyRunner" not in main_py_content:
        return True  # old stub
    if "get_loop_interval" not in main_py_content:
        return True  # missing loop-interval fix
    return False


def patch_strategy_py(content: str) -> tuple[str, bool]:
    """
    Add get_interval(), get_loop_interval(), and analyze() to strategy.py
    if they're missing.
    Returns (patched_content, was_changed).
    """
    changed = False
    additions = []

    if "def get_interval" not in content:
        additions.append(_GET_INTERVAL_METHOD)
        changed = True

    if "def get_loop_interval" not in content:
        additions.append(_GET_LOOP_INTERVAL_METHOD)
        changed = True

    if "async def analyze" not in content:
        additions.append(_ANALYZE_METHOD)
        changed = True

    if additions:
        content = content.rstrip() + "\n" + "".join(additions) + "\n"

    return content, changed


async def main() -> None:
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("DRY RUN — no changes will be written\n")

    if not DATABASE_URL:
        print("ERROR: DATABASE_URL env var required", file=sys.stderr)
        sys.exit(1)
    if not dry_run and not BUILD_AGENT_URL:
        print("ERROR: BUILD_AGENT_URL env var required", file=sys.stderr)
        sys.exit(1)
    if not dry_run and not BUILD_AGENT_API_KEY:
        print("ERROR: BUILD_AGENT_API_KEY env var required", file=sys.stderr)
        sys.exit(1)

    if not dry_run:
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

    # Fetch all complete strategies (same ordering logic as recover_builds.py)
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

    total_scanned = 0
    total_stub = 0
    total_fixed_db = 0
    total_pushed = 0
    total_failed = 0
    total_skipped = 0

    for strat in strategies:
        strat_id = strat["uuid"]
        strat_name = strat["name"]
        current_version = strat["version"] or 1

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

        print(f"\nStrategy: {strat_name!r}  version={current_version}  builds={len(builds)}")

        latest_pushed_tag = None

        for i, build in enumerate(builds):
            build_id = build["uuid"]
            version = current_version - i

            if version < 1:
                print(f"  [{build_id[:8]}] v{version} < 1 — skipping")
                total_skipped += 1
                continue

            iteration = await conn.fetchrow(
                """
                SELECT uuid, strategy_files
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
                print(f"  [{build_id[:8]}] v{version}: strategy_files malformed — skipping")
                total_skipped += 1
                continue

            total_scanned += 1

            main_py = files.get("main.py", "")
            if not needs_repair(main_py):
                print(f"  [{build_id[:8]}] v{version}: main.py OK — skipping")
                total_skipped += 1
                continue

            total_stub += 1
            reason = "STUB" if "StrategyRunner" not in main_py else "MISSING loop_interval"
            print(f"  [{build_id[:8]}] v{version}: {reason} — repairing...", end=" ", flush=True)

            # Patch main.py and requirements.txt
            files["main.py"] = CORRECT_MAIN_PY
            files["requirements.txt"] = CORRECT_REQUIREMENTS_TXT

            # Patch strategy.py if it exists and is missing methods
            if "strategy.py" in files:
                patched, changed = patch_strategy_py(files["strategy.py"])
                if changed:
                    files["strategy.py"] = patched

            if dry_run:
                print("(dry-run: would update DB + push image)")
                total_fixed_db += 1
                continue

            # Update strategy_files in the DB
            iteration_id = iteration["uuid"]
            await conn.execute(
                """
                UPDATE build_iterations
                SET strategy_files = $1
                WHERE uuid = $2
                """,
                json.dumps(files),
                iteration_id,
            )
            total_fixed_db += 1

            # Push the fixed image via build agent
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

        # Update docker_image_url for the most-recently-pushed image
        if not dry_run and latest_pushed_tag:
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
    print("Repair complete:")
    print(f"  {total_scanned:>4} build iterations scanned")
    print(f"  {total_stub:>4} had stub main.py")
    print(f"  {total_fixed_db:>4} DB records updated")
    if not dry_run:
        print(f"  {total_pushed:>4} images rebuilt and pushed")
        print(f"  {total_failed:>4} image pushes failed")
    print(f"  {total_skipped:>4} skipped (OK or no data)")


if __name__ == "__main__":
    asyncio.run(main())
