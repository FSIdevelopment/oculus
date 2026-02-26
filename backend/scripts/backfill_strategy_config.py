"""
Backfill strategy.config and strategy.description for strategies where these are NULL.

Both values come from the best BuildIteration's strategy_files["config.json"].

- strategy.config  → the parsed config.json dict
- strategy.description → config["description"] (only populated when currently NULL)

Usage:
    python scripts/backfill_strategy_config.py [--strategy-id UUID] [--dry-run]

Options:
    --strategy-id UUID  Process a specific strategy only
    --dry-run           Show what would be done without making changes
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, or_
from app.database import AsyncSessionLocal
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.models.build_iteration import BuildIteration
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _composite_score(iteration: BuildIteration) -> float:
    """Same composite scoring used by the build process and README generator."""
    bt = iteration.backtest_results or {}
    total_return = bt.get("total_return", 0)
    sharpe = bt.get("sharpe_ratio", 0)
    win_rate = bt.get("win_rate", 0)
    max_drawdown = abs(bt.get("max_drawdown", 0))
    total_trades = bt.get("total_trades", 0)

    sharpe_score = sharpe * 10
    win_score = (win_rate - 50) * 0.5
    drawdown_penalty = max_drawdown * 0.3
    trade_bonus = min(total_trades / 10, 5)

    return total_return + sharpe_score + win_score - drawdown_penalty + trade_bonus


async def get_best_config_for_strategy(db, strategy_id: str) -> dict | None:
    """
    Return the parsed config.json dict from the best completed iteration
    across all complete builds for this strategy.
    Returns None if no usable config is found.
    """
    # Get all complete builds for this strategy, newest first
    build_result = await db.execute(
        select(StrategyBuild.uuid)
        .where(
            StrategyBuild.strategy_id == strategy_id,
            StrategyBuild.status == "complete",
        )
        .order_by(StrategyBuild.completed_at.desc())
    )
    build_ids = build_result.scalars().all()

    if not build_ids:
        return None

    # Try each build, pick the one whose best iteration has the highest composite score
    best_config = None
    best_score = float("-inf")

    for build_id in build_ids:
        iter_result = await db.execute(
            select(BuildIteration)
            .where(
                BuildIteration.build_id == build_id,
                BuildIteration.status == "complete",
            )
            .order_by(BuildIteration.iteration_number.asc())
        )
        iterations = iter_result.scalars().all()

        for iteration in iterations:
            sf = iteration.strategy_files or {}
            raw = sf.get("config.json")
            if not raw:
                continue

            # Parse if stored as a string
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue
            elif isinstance(raw, dict):
                parsed = raw
            else:
                continue

            score = _composite_score(iteration)
            if score > best_score:
                best_score = score
                best_config = parsed

    return best_config


async def backfill_strategy(db, strategy: Strategy, dry_run: bool) -> tuple[bool, str]:
    """
    Populate strategy.config and/or strategy.description from build data.
    Returns (changed, reason_string).
    """
    needs_config = strategy.config is None
    needs_description = strategy.description is None

    if not needs_config and not needs_description:
        return False, "already populated"

    config = await get_best_config_for_strategy(db, strategy.uuid)
    if config is None:
        return False, "no config.json found in any completed iteration"

    changes = []

    if needs_config:
        if not dry_run:
            strategy.config = config
        changes.append("config")

    if needs_description:
        desc = config.get("description", "").strip()
        if desc and desc != "A custom trading strategy. Generated automatically — see strategy_name and indicators for details.":
            if not dry_run:
                strategy.description = desc
            changes.append(f"description='{desc[:60]}...'")
        else:
            # Fall back to strategy_name from config if description is generic/missing
            sname = config.get("strategy_name", "").strip()
            if sname and sname != "Template Strategy":
                if not dry_run:
                    strategy.description = sname
                changes.append(f"description='{sname}'")

    if not changes:
        return False, "config found but nothing useful to populate"

    return True, ", ".join(changes)


async def main(strategy_id: str | None = None, dry_run: bool = False):
    async with AsyncSessionLocal() as db:
        query = select(Strategy).where(
            Strategy.status == "complete",
            or_(Strategy.config.is_(None), Strategy.description.is_(None)),
        )

        if strategy_id:
            query = query.where(Strategy.uuid == strategy_id)

        result = await db.execute(query)
        strategies = result.scalars().all()

        if not strategies:
            logger.info("No strategies found that need backfilling")
            return

        logger.info(f"Found {len(strategies)} strategy/strategies to process")

        success = fail = skipped = 0

        for strategy in strategies:
            label = f"Strategy '{strategy.name}' ({strategy.uuid})"
            try:
                changed, reason = await backfill_strategy(db, strategy, dry_run)
                if changed:
                    prefix = "[DRY RUN] Would set" if dry_run else "Updated"
                    logger.info(f"  ✓ {prefix} {label}: {reason}")
                    success += 1
                else:
                    logger.info(f"  - Skipped {label}: {reason}")
                    skipped += 1
            except Exception as e:
                logger.error(f"  ✗ Failed {label}: {e}")
                fail += 1

        if not dry_run and success > 0:
            await db.commit()
            logger.info(f"\nCommitted changes to database")

        logger.info(f"\nSummary: {success} updated, {skipped} skipped, {fail} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill strategy.config and strategy.description from build iteration data"
    )
    parser.add_argument("--strategy-id", type=str, help="Process a specific strategy UUID only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without writing")
    args = parser.parse_args()

    asyncio.run(main(strategy_id=args.strategy_id, dry_run=args.dry_run))
