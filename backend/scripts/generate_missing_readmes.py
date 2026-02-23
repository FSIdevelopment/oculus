"""
Generate READMEs for existing builds that are missing them.

This script:
1. Finds all completed builds without a README
2. Retrieves the best iteration's data from the database
3. Generates a README using the LLM
4. Saves it to the database

Usage:
    python scripts/generate_missing_readmes.py [--build-id BUILD_ID] [--dry-run]

Options:
    --build-id BUILD_ID  Generate README for a specific build only
    --dry-run           Show what would be done without making changes
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import AsyncSessionLocal
from app.models.strategy_build import StrategyBuild
from app.models.build_iteration import BuildIteration
from app.models.strategy import Strategy
from app.models.user import User
from app.services.build_orchestrator import BuildOrchestrator
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_best_iteration(db: AsyncSession, build_id: str) -> BuildIteration | None:
    """Get the best completed iteration for a build using the same composite scoring as the build process.

    This matches the logic in backend/app/routers/builds.py around line 1491.
    """
    result = await db.execute(
        select(BuildIteration)
        .where(
            BuildIteration.build_id == build_id,
            BuildIteration.status == "complete"
        )
        .order_by(BuildIteration.iteration_number.asc())
    )
    iterations = result.scalars().all()

    if not iterations:
        return None

    # Use the same composite scoring function as the build process
    def _composite_score(iteration: BuildIteration) -> float:
        """Calculate composite score for an iteration.

        This is the EXACT same logic used in the build process to select the winning iteration.
        """
        bt = iteration.backtest_results or {}
        total_return = bt.get("total_return", 0)
        sharpe = bt.get("sharpe_ratio", 0)
        win_rate = bt.get("win_rate", 0)
        max_drawdown = abs(bt.get("max_drawdown", 0))
        total_trades = bt.get("total_trades", 0)

        # Return: primary driver
        return_score = total_return

        # Sharpe: scale by 10 to make it comparable to return %
        sharpe_score = sharpe * 10

        # Win rate: 50% is break-even, reward above that
        win_score = (win_rate - 50) * 0.5  # e.g. 60% -> 5 points

        # Drawdown penalty: higher drawdown = lower score
        drawdown_penalty = max_drawdown * 0.3  # e.g. 10% drawdown -> -3 points

        # Trade count bonus: penalize too few trades (unreliable)
        trade_bonus = min(total_trades / 10, 5)  # max 5 points for 50+ trades

        return return_score + sharpe_score + win_score - drawdown_penalty + trade_bonus

    # Select the iteration with the highest composite score (same as build process)
    best_iter = max(iterations, key=_composite_score)

    return best_iter


async def generate_readme_for_build(
    db: AsyncSession,
    build: StrategyBuild,
    strategy: Strategy,
    user: User,
    dry_run: bool = False
) -> bool:
    """Generate and save README for a single build."""
    logger.info(f"Processing build {build.uuid} for strategy '{strategy.name}'...")

    # Get best iteration
    best_iter = await get_best_iteration(db, build.uuid)
    if not best_iter:
        logger.warning(f"  No completed iterations found for build {build.uuid}")
        return False

    logger.info(f"  Using iteration #{best_iter.iteration_number} (return: {(best_iter.backtest_results or {}).get('total_return', 0)}%)")

    # Extract data from iteration
    design = best_iter.llm_design or {}
    backtest_results = best_iter.backtest_results or {}
    best_model = best_iter.best_model or {}
    model_metrics = best_model.get("metrics", {}) if isinstance(best_model, dict) else {}

    # Build config from iteration data
    config = {
        "entry_rules": best_iter.entry_rules or [],
        "exit_rules": best_iter.exit_rules or [],
        "features": best_iter.features or {},
    }

    if dry_run:
        logger.info(f"  [DRY RUN] Would generate README for build {build.uuid}")
        return True

    # Generate README using BuildOrchestrator
    try:
        # Create BuildOrchestrator with db, user, and build
        orchestrator = BuildOrchestrator(db, user, build)

        readme = await orchestrator.generate_strategy_readme(
            strategy_name=strategy.name,
            symbols=strategy.symbols or [],
            design=design,
            backtest_results=backtest_results,
            model_metrics=model_metrics,
            config=config,
            strategy_type=strategy.strategy_type or "",
        )

        # Save to database
        build.readme = readme
        await db.commit()

        logger.info(f"  ✓ Generated and saved README ({len(readme)} chars)")
        return True

    except Exception as e:
        logger.error(f"  ✗ Failed to generate README: {e}")
        return False


async def main(build_id: str | None = None, dry_run: bool = False):
    """Main function to generate missing READMEs."""
    async with AsyncSessionLocal() as db:
        # Find builds without READMEs, joining with Strategy and User
        query = select(StrategyBuild, Strategy, User).join(
            Strategy, StrategyBuild.strategy_id == Strategy.uuid
        ).join(
            User, StrategyBuild.user_id == User.uuid
        ).where(
            StrategyBuild.status == "complete",
            StrategyBuild.readme.is_(None)
        )

        if build_id:
            query = query.where(StrategyBuild.uuid == build_id)

        result = await db.execute(query)
        builds_strategies_users = result.all()

        if not builds_strategies_users:
            if build_id:
                logger.info(f"Build {build_id} not found or already has a README")
            else:
                logger.info("No builds found that need READMEs")
            return

        logger.info(f"Found {len(builds_strategies_users)} build(s) without READMEs")

        success_count = 0
        fail_count = 0

        for build, strategy, user in builds_strategies_users:
            try:
                success = await generate_readme_for_build(db, build, strategy, user, dry_run)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"Unexpected error processing build {build.uuid}: {e}")
                fail_count += 1

        logger.info(f"\nSummary: {success_count} successful, {fail_count} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate READMEs for existing builds that are missing them"
    )
    parser.add_argument(
        "--build-id",
        type=str,
        help="Generate README for a specific build only"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    asyncio.run(main(build_id=args.build_id, dry_run=args.dry_run))

