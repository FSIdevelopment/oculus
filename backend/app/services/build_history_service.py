"""Service for managing build history and feature tracking in PostgreSQL."""
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.build_history import BuildHistory, FeatureTracker


class BuildHistoryService:
    """Service for managing build history records."""
    
    @staticmethod
    async def add_build(
        db: AsyncSession,
        user_id: str,
        strategy_name: str,
        asset_class: str,
        timeframe: str,
        target: float,
        symbols: List[str],
        success: bool,
        backtest_results: Optional[Dict] = None,
        model_info: Optional[Dict] = None,
        risk_params: Optional[Dict] = None,
        features: Optional[Dict] = None,
        iterations: Optional[Dict] = None,
    ) -> BuildHistory:
        """Add a completed build record to history."""
        # Remove existing entry with same strategy_name for this user
        await db.execute(
            select(BuildHistory).where(
                (BuildHistory.user_id == user_id) &
                (BuildHistory.strategy_name == strategy_name)
            )
        )
        
        build = BuildHistory(
            strategy_name=strategy_name,
            asset_class=asset_class,
            timeframe=timeframe,
            target=target,
            symbols=symbols,
            success=success,
            backtest_results=backtest_results or {},
            model_info=model_info or {},
            risk_params=risk_params or {},
            features=features or {},
            iterations=iterations or {},
            user_id=user_id,
        )
        
        db.add(build)
        await db.flush()
        return build
    
    @staticmethod
    async def get_relevant_builds(
        db: AsyncSession,
        user_id: str,
        symbols: Optional[List[str]] = None,
        asset_class: Optional[str] = None,
        max_results: int = 5,
    ) -> List[BuildHistory]:
        """Get most relevant past builds for current design task."""
        # Query all builds for this user
        result = await db.execute(
            select(BuildHistory).where(BuildHistory.user_id == user_id)
        )
        builds = result.scalars().all()
        
        if not builds:
            return []
        
        # Score builds
        symbol_set = set(s.upper() for s in (symbols or []))
        scored = []
        
        for build in builds:
            score = 0
            build_symbols = set(s.upper() for s in build.symbols)
            
            # Asset class match
            if asset_class and build.asset_class == asset_class:
                score += 10
            
            # Symbol overlap
            overlap = len(symbol_set & build_symbols)
            score += overlap * 3
            
            # Success bonus
            if build.success:
                score += 5
            
            # Recency bonus
            days_ago = (datetime.utcnow() - build.build_date).days
            score += max(0, 2 - days_ago / 30)
            
            scored.append((score, build))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [build for _, build in scored[:max_results]]


class FeatureTrackerService:
    """Service for managing feature effectiveness tracking."""
    
    @staticmethod
    async def update_feature_stats(
        db: AsyncSession,
        feature_name: str,
        times_used: int = 1,
        performance_data: Optional[Dict] = None,
        entry_rule_stats: Optional[Dict] = None,
        exit_rule_stats: Optional[Dict] = None,
    ) -> FeatureTracker:
        """Update or create feature tracker entry."""
        result = await db.execute(
            select(FeatureTracker).where(FeatureTracker.feature_name == feature_name)
        )
        tracker = result.scalar_one_or_none()
        
        if tracker:
            tracker.times_used += times_used
            if performance_data:
                tracker.performance_data = performance_data
            if entry_rule_stats:
                tracker.entry_rule_stats = entry_rule_stats
            if exit_rule_stats:
                tracker.exit_rule_stats = exit_rule_stats
            tracker.updated_at = datetime.utcnow()
        else:
            tracker = FeatureTracker(
                feature_name=feature_name,
                times_used=times_used,
                performance_data=performance_data or {},
                entry_rule_stats=entry_rule_stats or {},
                exit_rule_stats=exit_rule_stats or {},
            )
            db.add(tracker)
        
        await db.flush()
        return tracker
    
    @staticmethod
    async def get_feature_stats(
        db: AsyncSession,
        feature_name: str,
    ) -> Optional[FeatureTracker]:
        """Get feature tracker entry."""
        result = await db.execute(
            select(FeatureTracker).where(FeatureTracker.feature_name == feature_name)
        )
        return result.scalar_one_or_none()

