"""
Risk Manager Module

Provides comprehensive risk management for trading strategies including:
- Stop Loss: Exit if position loss exceeds threshold
- Take Profit: Exit if position gain exceeds threshold  
- Trailing Stop: Track highest price, exit if drops X% from peak
- Max Daily Loss: Pause new entries if daily loss exceeds threshold
- Max Drawdown: Circuit breaker if portfolio drawdown exceeds threshold

Author: SignalSynk
Copyright © 2026 Forefront Solutions Inc. All rights reserved.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger('RiskManager')


@dataclass
class PositionRisk:
    """Tracks risk metrics for a single position"""
    symbol: str
    entry_price: float
    entry_date: datetime
    quantity: int
    highest_price: float  # For trailing stop
    lowest_price: float   # For tracking
    
    def update_price(self, current_price: float):
        """Update high/low watermarks"""
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
    
    def get_pnl_pct(self, current_price: float) -> float:
        """Get current P/L percentage"""
        if self.entry_price <= 0:
            return 0.0
        return ((current_price - self.entry_price) / self.entry_price) * 100
    
    def get_trailing_drawdown_pct(self, current_price: float) -> float:
        """Get drawdown from highest price (for trailing stop)"""
        if self.highest_price <= 0:
            return 0.0
        return ((self.highest_price - current_price) / self.highest_price) * 100


class RiskManager:
    """
    Manages risk across all positions and portfolio level.
    
    Integrates with backtest engine to enforce risk rules.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RiskManager with risk configuration.
        
        Args:
            config: Risk management configuration dict with keys:
                - stop_loss_pct: Exit if position loss exceeds this %
                - take_profit_pct: Exit if position gain exceeds this %
                - trailing_stop_pct: Exit if price drops this % from peak
                - max_daily_loss_pct: Pause trading if daily loss exceeds this %
                - max_drawdown_pct: Circuit breaker if drawdown exceeds this %
                - cooldown_days: Days to wait after hitting limits
        """
        self.stop_loss_pct = config.get('stop_loss_pct', 5.0)
        self.take_profit_pct = config.get('take_profit_pct', 10.0)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 3.0)
        self.max_daily_loss_pct = config.get('max_daily_loss_pct', 3.0)
        self.max_drawdown_pct = config.get('max_drawdown_pct', 15.0)
        self.cooldown_days = config.get('cooldown_days', 1)
        
        # State tracking
        self.position_risks: Dict[str, PositionRisk] = {}
        self.daily_pnl = 0.0
        self.daily_start_value = 0.0
        self.current_date: Optional[datetime] = None
        self.peak_portfolio_value = 0.0
        self.trading_paused = False
        self.pause_reason = ""
        self.cooldown_until: Optional[datetime] = None
        
        # Statistics
        self.stop_loss_exits = 0
        self.take_profit_exits = 0
        self.trailing_stop_exits = 0
        self.daily_limit_hits = 0
        
        logger.info(f"RiskManager initialized: SL={self.stop_loss_pct}%, TP={self.take_profit_pct}%, "
                   f"Trailing={self.trailing_stop_pct}%, MaxDailyLoss={self.max_daily_loss_pct}%")
    
    def start_new_day(self, date: datetime, portfolio_value: float):
        """Reset daily tracking at start of each trading day"""
        self.current_date = date
        self.daily_start_value = portfolio_value
        self.daily_pnl = 0.0
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
        
        # Check if cooldown period has ended
        if self.cooldown_until and date >= self.cooldown_until:
            self.trading_paused = False
            self.pause_reason = ""
            self.cooldown_until = None
            logger.info(f"📗 Trading resumed after cooldown period")
    
    def register_position(self, symbol: str, entry_price: float, entry_date: datetime, quantity: int):
        """Register a new position for risk tracking"""
        self.position_risks[symbol] = PositionRisk(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=entry_date,
            quantity=quantity,
            highest_price=entry_price,
            lowest_price=entry_price
        )
        logger.debug(f"Registered position risk tracking for {symbol} @ ${entry_price:.2f}")
    
    def unregister_position(self, symbol: str):
        """Remove position from risk tracking"""
        if symbol in self.position_risks:
            del self.position_risks[symbol]
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update position's high/low watermarks"""
        if symbol in self.position_risks:
            self.position_risks[symbol].update_price(current_price)

    def check_position_risk(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if position should be closed due to risk rules.

        Returns:
            'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', or None
        """
        if symbol not in self.position_risks:
            return None

        pos = self.position_risks[symbol]
        pos.update_price(current_price)

        pnl_pct = pos.get_pnl_pct(current_price)
        trailing_dd = pos.get_trailing_drawdown_pct(current_price)

        # Check stop loss (negative P/L exceeds threshold)
        if pnl_pct <= -self.stop_loss_pct:
            self.stop_loss_exits += 1
            logger.info(f"🛑 {symbol}: STOP LOSS triggered at {pnl_pct:.1f}% loss")
            return 'STOP_LOSS'

        # Check take profit (positive P/L exceeds threshold)
        if pnl_pct >= self.take_profit_pct:
            self.take_profit_exits += 1
            logger.info(f"🎯 {symbol}: TAKE PROFIT triggered at {pnl_pct:.1f}% gain")
            return 'TAKE_PROFIT'

        # Check trailing stop (only if position is profitable)
        # Trailing stop activates when we've had a gain but price drops from peak
        if pos.highest_price > pos.entry_price and trailing_dd >= self.trailing_stop_pct:
            self.trailing_stop_exits += 1
            logger.info(f"📉 {symbol}: TRAILING STOP triggered at {trailing_dd:.1f}% from peak ${pos.highest_price:.2f}")
            return 'TRAILING_STOP'

        return None

    def check_daily_loss_limit(self, current_portfolio_value: float) -> bool:
        """
        Check if daily loss limit has been breached.

        Returns:
            True if trading should be paused
        """
        if self.daily_start_value <= 0:
            return False

        daily_loss_pct = ((self.daily_start_value - current_portfolio_value) / self.daily_start_value) * 100

        if daily_loss_pct >= self.max_daily_loss_pct:
            if not self.trading_paused:
                self.trading_paused = True
                self.pause_reason = f"Daily loss limit hit: {daily_loss_pct:.1f}%"
                self.daily_limit_hits += 1
                logger.warning(f"⚠️ DAILY LOSS LIMIT: {daily_loss_pct:.1f}% - Trading paused")
            return True

        return False

    def check_drawdown_limit(self, current_portfolio_value: float) -> bool:
        """
        Check if max drawdown circuit breaker should trigger.

        Returns:
            True if trading should be paused
        """
        if self.peak_portfolio_value <= 0:
            return False

        drawdown_pct = ((self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value) * 100

        if drawdown_pct >= self.max_drawdown_pct:
            if not self.trading_paused:
                self.trading_paused = True
                self.pause_reason = f"Max drawdown breached: {drawdown_pct:.1f}%"
                logger.warning(f"🚨 MAX DRAWDOWN BREACHED: {drawdown_pct:.1f}% - Trading paused")
            return True

        return False

    def can_open_new_position(self) -> bool:
        """Check if new positions are allowed based on risk state"""
        return not self.trading_paused

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk management status"""
        return {
            'trading_paused': self.trading_paused,
            'pause_reason': self.pause_reason,
            'stop_loss_exits': self.stop_loss_exits,
            'take_profit_exits': self.take_profit_exits,
            'trailing_stop_exits': self.trailing_stop_exits,
            'daily_limit_hits': self.daily_limit_hits,
            'positions_tracked': len(self.position_risks),
            'peak_portfolio_value': self.peak_portfolio_value
        }

    def get_exit_reason_for_trade(self, exit_type: str) -> str:
        """Get human-readable exit reason for trade logging"""
        reasons = {
            'STOP_LOSS': 'Stop loss triggered',
            'TAKE_PROFIT': 'Take profit target reached',
            'TRAILING_STOP': 'Trailing stop triggered',
            'SIGNAL': 'Strategy signal'
        }
        return reasons.get(exit_type, 'Unknown')

