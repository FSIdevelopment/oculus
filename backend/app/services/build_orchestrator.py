"""Build orchestrator service for strategy design and training."""
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import anthropic
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.strategy_build import StrategyBuild
from app.models.user import User
from app.services.balance import deduct_tokens
from app.services.build_history_service import BuildHistoryService, FeatureTrackerService


class BuildOrchestrator:
    """Orchestrates strategy build process: LLM design, training, and iteration."""
    
    def __init__(self, db: AsyncSession, user: User, build: StrategyBuild):
        self.db = db
        self.user = user
        self.build = build
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.redis_client: Optional[redis.Redis] = None
        self.logs: List[str] = []
    
    async def initialize(self):
        """Initialize Redis connection via Ngrok tunnel."""
        self.redis_client = await redis.from_url(settings.REDIS_URL)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.redis_client:
            await self.redis_client.close()
    
    def _log(self, message: str):
        """Add message to logs."""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
    
    async def design_strategy(
        self,
        strategy_name: str,
        symbols: List[str],
        description: str,
        timeframe: str,
        target_return: float,
    ) -> Dict[str, Any]:
        """Call Claude API to design strategy.

        Returns a dict with keys:
        - "design": the parsed strategy design JSON
        - "thinking": concatenated thinking block text
        """
        self._log(f"Starting LLM strategy design for {strategy_name}")

        # Get relevant past builds for context
        asset_class = self._infer_asset_class(symbols)
        relevant_builds = await BuildHistoryService.get_relevant_builds(
            self.db, self.user.uuid, symbols, asset_class, max_results=3
        )

        # Build context from past builds
        context = self._format_build_context(relevant_builds)

        # Create prompt
        prompt = self._build_design_prompt(
            strategy_name, symbols, description, timeframe, target_return, context
        )

        try:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract design and thinking from response
            result = self._parse_design_response(response)
            desc_preview = result["design"].get("strategy_description", "N/A")[:100]
            self._log(f"Strategy design completed: {desc_preview}")
            return result

        except anthropic.APIError as e:
            self._log(f"Claude API error: {e}")
            raise
    
    async def dispatch_training_job(
        self,
        design: Dict[str, Any],
        symbols: List[str],
        timeframe: str,
    ) -> str:
        """Dispatch training job to Redis via REDIS_URL."""
        self._log("Dispatching training job to Redis")
        
        job_payload = {
            "build_id": self.build.uuid,
            "user_id": self.user.uuid,
            "design": design,
            "symbols": symbols,
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Push to Redis queue
        job_id = f"build:{self.build.uuid}"
        await self.redis_client.set(job_id, json.dumps(job_payload), ex=86400)  # 24h TTL
        await self.redis_client.lpush("training_queue", job_id)
        
        self._log(f"Job dispatched with ID: {job_id}")
        return job_id
    
    async def listen_for_results(self, job_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Listen for training results from Redis."""
        self._log(f"Listening for results on {job_id}")
        
        result_key = f"result:{job_id}"
        start_time = datetime.utcnow()
        
        while True:
            result_data = await self.redis_client.get(result_key)
            if result_data:
                result = json.loads(result_data)
                self._log(f"Results received: {result.get('status', 'unknown')}")
                return result
            
            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > timeout:
                raise TimeoutError(f"Training job {job_id} timed out after {timeout}s")
            
            await asyncio.sleep(5)  # Poll every 5 seconds
    
    async def refine_strategy(
        self,
        previous_design: Dict[str, Any],
        backtest_results: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """Refine strategy based on backtest results.

        Returns a dict with keys:
        - "design": the parsed strategy design JSON
        - "thinking": concatenated thinking block text
        """
        self._log(f"Refining strategy (iteration {iteration})")

        prompt = self._build_refinement_prompt(
            previous_design, backtest_results, iteration
        )

        try:
            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=settings.CLAUDE_MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )

            result = self._parse_design_response(response)
            self._log(f"Strategy refined (iteration {iteration})")
            return result

        except anthropic.APIError as e:
            self._log(f"Claude API error during refinement: {e}")
            raise
    
    async def deduct_tokens(self, amount: float, description: str):
        """Deduct tokens from user balance."""
        await deduct_tokens(self.db, self.user.uuid, amount, description)
        self._log(f"Deducted {amount} tokens: {description}")
    
    def _infer_asset_class(self, symbols: List[str]) -> str:
        """Infer asset class from symbols."""
        asset_class_map = {
            "semiconductors": {"NVDA", "AMD", "AVGO", "TSM", "QCOM"},
            "tech": {"AAPL", "MSFT", "GOOGL", "META", "AMZN"},
            "energy": {"XLE", "XOP", "CVX", "XOM"},
            "index": {"SPY", "QQQ", "IWM", "DIA"},
        }
        
        symbol_set = set(s.upper() for s in symbols)
        for asset_class, class_symbols in asset_class_map.items():
            if symbol_set & class_symbols:
                return asset_class
        return "mixed"
    
    def _format_build_context(self, builds: List) -> str:
        """Format past builds as context for Claude."""
        if not builds:
            return "No previous builds available."
        
        lines = []
        for b in builds:
            lines.append(f"- {b.strategy_name}: {b.asset_class}, symbols={b.symbols}")
            if b.backtest_results:
                lines.append(f"  Return: {b.backtest_results.get('total_return', 'N/A')}%")
        
        return "\n".join(lines)
    
    def _build_design_prompt(
        self,
        strategy_name: str,
        symbols: List[str],
        description: str,
        timeframe: str,
        target_return: float,
        context: str,
    ) -> str:
        """Build initial strategy design prompt."""
        return f"""Design a trading strategy with these specifications:
- Name: {strategy_name}
- Symbols: {', '.join(symbols)}
- Description: {description}
- Timeframe: {timeframe}
- Target Return: {target_return}%

Previous successful builds:
{context}

Provide a JSON response with:
- strategy_description
- strategy_rationale
- priority_features (list of technical indicators)
- entry_rules (list of dicts with feature, operator, threshold)
- exit_rules (list of dicts with feature, operator, threshold)
- recommended_n_iter_search (int)
"""
    
    def _build_refinement_prompt(
        self,
        previous_design: Dict[str, Any],
        backtest_results: Dict[str, Any],
        iteration: int,
    ) -> str:
        """Build strategy refinement prompt."""
        return f"""Refine the strategy based on backtest results.

Previous design:
{json.dumps(previous_design, indent=2)}

Backtest results (iteration {iteration}):
{json.dumps(backtest_results, indent=2)}

Provide refined JSON with same structure as before."""
    
    def _parse_design_response(self, response) -> Dict[str, Any]:
        """Parse Claude response to extract design JSON and thinking content.

        Returns a dict with keys:
        - "design": the parsed strategy design JSON
        - "thinking": concatenated thinking block text (empty string if none)
        """
        # Extract text and thinking blocks from response
        text = ""
        thinking_parts: List[str] = []
        for block in response.content:
            if block.type == "thinking" and hasattr(block, "thinking"):
                thinking_parts.append(block.thinking)
            elif hasattr(block, "text"):
                text += block.text

        thinking_text = "\n".join(thinking_parts)

        # Find JSON in response
        import re
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                design = json.loads(json_match.group())
                return {"design": design, "thinking": thinking_text}
            except json.JSONDecodeError:
                pass

        # Return default design if parsing fails
        return {
            "design": {
                "strategy_description": "Default design",
                "strategy_rationale": "Could not parse Claude response",
                "priority_features": [],
                "entry_rules": [],
                "exit_rules": [],
                "recommended_n_iter_search": 50,
            },
            "thinking": thinking_text,
        }

