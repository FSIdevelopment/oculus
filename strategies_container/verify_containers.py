#!/usr/bin/env python3
"""
Verification script: run backtest.py inside each strategy container and compare
results against what is stored in build_iterations.backtest_results.

Run this ON the signalSynk-Strategies Droplet (where Docker is available):

    DATABASE_URL="postgresql://..." \
    POLYGON_API_KEY="..." \
    ALPHAVANTAGE_API_KEY="..." \
    python3 /tmp/verify_containers.py

Note: containers re-fetch live market data with BACKTEST_PERIOD=1Y so metrics will
differ from DB values (which were computed at build time). The script flags containers
that ERROR or produce results wildly outside the DB baseline.
"""
import asyncio
import json
import os
import subprocess
import sys
import time

import asyncpg

DATABASE_URL = os.environ.get("DATABASE_URL", "")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
REGISTRY = "registry.digitalocean.com/oculus-strategies"

# Thresholds for flagging divergence
MAX_RETURN_DELTA_PP = 100.0   # percentage points on total_return
MAX_WINRATE_DELTA_PP = 30.0   # percentage points on win_rate
MAX_TRADES_DELTA_PCT = 50.0   # percent on total_trades


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


def _sanitize_tag(name: str) -> str:
    """Convert strategy name to a valid Docker tag component (mirrors builder.py)."""
    import re
    tag = name.lower()
    tag = re.sub(r"[^a-z0-9\-]", "-", tag)
    tag = re.sub(r"-{2,}", "-", tag)
    tag = tag.strip("-")
    return tag


def _get_metric(results: dict, *keys) -> float | None:
    """Try multiple key names to handle both _pct and non-_pct conventions."""
    for k in keys:
        if k in results and results[k] is not None:
            return float(results[k])
    return None


def _run(cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _compare(db: dict | None, container: dict) -> tuple[bool, list[str]]:
    """
    Compare DB backtest_results with container output.
    Returns (passed, list_of_issues).
    """
    if not db:
        return True, ["(no DB baseline — skipping metric comparison)"]

    issues = []
    details = []

    # Normalise total_return (DB may store as total_return or total_return_pct)
    db_return = _get_metric(db, "total_return_pct", "total_return")
    ct_return = _get_metric(container, "total_return_pct", "total_return")

    # Normalise win_rate
    db_wr = _get_metric(db, "win_rate_pct", "win_rate")
    ct_wr = _get_metric(container, "win_rate_pct", "win_rate")

    # Sharpe
    db_sharpe = _get_metric(db, "sharpe_ratio")
    ct_sharpe = _get_metric(container, "sharpe_ratio")

    # Trades
    db_trades = _get_metric(db, "total_trades")
    ct_trades = _get_metric(container, "total_trades")

    # --- return check ---
    if db_return is not None and ct_return is not None:
        delta = ct_return - db_return
        details.append(f"return  DB={db_return:+.1f}%  Cont={ct_return:+.1f}%  Δ={delta:+.1f}pp")
        if abs(delta) > MAX_RETURN_DELTA_PP:
            issues.append(f"return delta too large ({delta:+.1f}pp > {MAX_RETURN_DELTA_PP}pp)")
        if db_return != 0 and (db_return > 0) != (ct_return > 0):
            issues.append(f"return sign flipped (DB={db_return:+.1f}%, Cont={ct_return:+.1f}%)")
    elif ct_return is not None:
        details.append(f"return  DB=n/a  Cont={ct_return:+.1f}%")

    # --- win_rate check ---
    if db_wr is not None and ct_wr is not None:
        delta = ct_wr - db_wr
        details.append(f"win_rate DB={db_wr:.1f}%  Cont={ct_wr:.1f}%  Δ={delta:+.1f}pp")
        if abs(delta) > MAX_WINRATE_DELTA_PP:
            issues.append(f"win_rate delta too large ({delta:+.1f}pp > {MAX_WINRATE_DELTA_PP}pp)")
    elif ct_wr is not None:
        details.append(f"win_rate DB=n/a  Cont={ct_wr:.1f}%")

    # --- sharpe check ---
    if db_sharpe is not None and ct_sharpe is not None:
        delta = ct_sharpe - db_sharpe
        details.append(f"sharpe  DB={db_sharpe:.2f}  Cont={ct_sharpe:.2f}  Δ={delta:+.2f}")
        if db_sharpe != 0 and (db_sharpe > 0) != (ct_sharpe > 0):
            issues.append(f"sharpe sign flipped (DB={db_sharpe:.2f}, Cont={ct_sharpe:.2f})")
    elif ct_sharpe is not None:
        details.append(f"sharpe  DB=n/a  Cont={ct_sharpe:.2f}")

    # --- trades check ---
    if db_trades is not None and ct_trades is not None and db_trades > 0:
        pct = abs(ct_trades - db_trades) / db_trades * 100
        details.append(f"trades  DB={int(db_trades)}  Cont={int(ct_trades)}  Δ={pct:.0f}%")
        if pct > MAX_TRADES_DELTA_PCT:
            issues.append(f"trade count delta too large ({pct:.0f}% > {MAX_TRADES_DELTA_PCT}%)")
    elif ct_trades is not None:
        details.append(f"trades  DB=n/a  Cont={int(ct_trades)}")

    return len(issues) == 0, details + (["ISSUES: " + ", ".join(issues)] if issues else [])


async def main() -> None:
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL env var required", file=sys.stderr)
        sys.exit(1)
    if not POLYGON_API_KEY:
        print("ERROR: POLYGON_API_KEY env var required", file=sys.stderr)
        sys.exit(1)

    print("Connecting to database...")
    dsn, ssl_required = _pg_dsn(DATABASE_URL)
    conn = await asyncpg.connect(dsn, ssl="require" if ssl_required else None)
    print("Database: connected\n")

    # Fetch all complete strategies — include docker_image_url so we use the
    # exact tag format that was actually pushed (avoids sanitize_tag mismatches)
    strategies = await conn.fetch(
        """
        SELECT uuid, name, version, docker_image_url
        FROM strategies
        WHERE status = 'complete'
        ORDER BY name
        """
    )
    print(f"Found {len(strategies)} complete strategies")
    print("=" * 70)

    total = 0
    passed = 0
    failed = 0
    failures = []

    for strat in strategies:
        strat_id = strat["uuid"]
        strat_name = strat["name"]
        latest_image_url = strat["docker_image_url"] or ""

        # Derive base_tag and the pushed version from docker_image_url.
        # This is more reliable than reconstructing from strategy name — it uses
        # the exact tag format that was written during recover_builds.
        if ":" in latest_image_url:
            base_tag, pushed_version_str = latest_image_url.rsplit(":", 1)
            try:
                current_version = int(pushed_version_str)
            except ValueError:
                current_version = strat["version"] or 1
                base_tag = f"{REGISTRY}/{_sanitize_tag(strat_name)}"
        else:
            current_version = strat["version"] or 1
            base_tag = f"{REGISTRY}/{_sanitize_tag(strat_name)}"

        builds = await conn.fetch(
            """
            SELECT sb.uuid AS build_id,
                   bi.uuid AS iter_id,
                   bi.backtest_results,
                   bi.strategy_files
            FROM strategy_builds sb
            JOIN build_iterations bi ON bi.build_id = sb.uuid
            WHERE sb.strategy_id = $1
              AND sb.status = 'complete'
              AND bi.status = 'complete'
              AND bi.strategy_files IS NOT NULL
            ORDER BY sb.completed_at DESC NULLS LAST,
                     bi.iteration_number DESC
            """,
            strat_id,
        )

        # Deduplicate: keep only the highest-iteration row per build_id
        seen_builds: dict[str, asyncpg.Record] = {}
        for row in builds:
            bid = row["build_id"]
            if bid not in seen_builds:
                seen_builds[bid] = row

        build_rows = list(seen_builds.values())

        if not build_rows:
            print(f"\nStrategy: {strat_name!r} — no completed builds with files, skipping")
            continue

        for i, row in enumerate(build_rows):
            version = current_version - i
            if version < 1:
                continue

            image_tag = f"{base_tag}:{version}"
            safe_name = base_tag.split("/")[-1]  # last path component, already sanitized
            container_name = f"verify_{safe_name}_v{version}"
            results_path = f"/tmp/verify_{safe_name}_v{version}.json"
            total += 1

            print(f"\nStrategy: {strat_name!r}  v{version}")
            print(f"  Image: {image_tag}")

            # 1. Pull image
            print("  Pull:  ", end="", flush=True)
            rc, _, err = _run(f"docker pull {image_tag}", timeout=120)
            if rc != 0:
                msg = f"FAIL — docker pull failed: {err[:200]}"
                print(msg)
                failed += 1
                failures.append(f"{strat_name} v{version}: {msg}")
                continue
            print("OK")

            # 2. Run backtest
            print("  Run:   ", end="", flush=True)
            t0 = time.time()
            run_cmd = (
                f"docker run --name {container_name} "
                f"-e BACKTEST_MODE=true "
                f"-e BACKTEST_PERIOD=1Y "
                f"-e STRATEGY_ID={strat_id} "
                f"-e POLYGON_API_KEY={POLYGON_API_KEY} "
                f"-e ALPHAVANTAGE_API_KEY={ALPHAVANTAGE_API_KEY} "
                f"-e LOG_LEVEL=WARNING "
                f"{image_tag} python backtest.py"
            )
            rc, stdout, stderr = _run(run_cmd, timeout=300)
            elapsed = time.time() - t0

            if rc != 0:
                _run(f"docker rm -f {container_name}")
                msg = f"FAIL — container exited {rc}: {(stderr or stdout)[:300]}"
                print(msg)
                failed += 1
                failures.append(f"{strat_name} v{version}: {msg}")
                continue
            print(f"OK ({elapsed:.1f}s)")

            # 3. Extract results
            print("  Copy:  ", end="", flush=True)
            rc, _, err = _run(f"docker cp {container_name}:/app/backtest_results.json {results_path}")
            _run(f"docker rm -f {container_name}")

            if rc != 0:
                msg = f"FAIL — docker cp failed: {err[:200]}"
                print(msg)
                failed += 1
                failures.append(f"{strat_name} v{version}: {msg}")
                continue
            print("OK")

            # 4. Parse + compare
            try:
                with open(results_path) as f:
                    container_results = json.load(f)
            except Exception as e:
                msg = f"FAIL — cannot parse backtest_results.json: {e}"
                print(f"  Parse: {msg}")
                failed += 1
                failures.append(f"{strat_name} v{version}: {msg}")
                continue

            db_results = row["backtest_results"]
            if isinstance(db_results, str):
                try:
                    db_results = json.loads(db_results)
                except Exception:
                    db_results = None

            ok, details = _compare(db_results, container_results)
            for line in details:
                print(f"         {line}")

            status = "PASS" if ok else "FAIL"
            print(f"  Result: {status}")

            if ok:
                passed += 1
            else:
                failed += 1
                failures.append(f"{strat_name} v{version}: metric divergence — {details[-1]}")

    await conn.close()

    print()
    print("=" * 70)
    print("Verification complete:")
    print(f"  {total:>3} containers tested")
    print(f"  {passed:>3} PASS")
    print(f"  {failed:>3} FAIL")

    if failures:
        print()
        print("Failures:")
        for f in failures:
            print(f"  - {f}")


if __name__ == "__main__":
    asyncio.run(main())
