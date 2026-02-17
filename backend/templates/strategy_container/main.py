#!/usr/bin/env python3
"""
Signal Synk Strategy Container - Main Entry Point

This is the main entry point for a Signal Synk trading strategy.
It handles:
1. WebSocket connection to Signal Synk
2. Authentication with the platform
3. Heartbeat/keepalive messaging
4. Strategy execution loop
5. Signal transmission to the platform

Author: Signal Synk Team
Copyright © 2026 Forefront Solutions Inc. All rights reserved.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo

import websockets

from strategy import TradingStrategy

# =============================================================================
# Market Hours Configuration
# Trading is only allowed during market hours: 9:30 AM - 6:00 PM Eastern Time
# =============================================================================

# US Eastern timezone
ET = ZoneInfo('America/New_York')

# Trading hours (9:30 AM - 6:00 PM ET)
TRADING_OPEN = time(9, 30)   # 9:30 AM ET
TRADING_CLOSE = time(18, 0)  # 6:00 PM ET

# US Market Holidays (Full closures) through 2028
MARKET_HOLIDAYS = [
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    # 2028
    "2028-01-01", "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29",
    "2028-06-19", "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25",
]

# Early close days (market closes at 1:00 PM ET - no extended hours)
HALF_DAYS = [
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-07-02", "2026-11-27", "2026-12-24",
    "2027-11-26",
    "2028-07-03", "2028-11-24",
]


def get_current_et_time() -> datetime:
    """Get the current time in Eastern timezone."""
    return datetime.now(ET)


def is_trading_hours() -> tuple[bool, str]:
    """
    Check if we're within allowed trading hours (9:30 AM - 6:00 PM ET).

    Returns:
        Tuple of (can_trade: bool, reason: str)
    """
    now = get_current_et_time()
    date_str = now.strftime("%Y-%m-%d")
    current_time = now.time()

    # Check if weekend
    if now.weekday() >= 5:
        return False, "Weekend - trading not allowed"

    # Check if holiday
    if date_str in MARKET_HOLIDAYS:
        return False, f"Holiday - trading not allowed ({date_str})"

    # Check if half day (early close - trading ends at 4:00 PM on half days)
    close_time = TRADING_CLOSE
    if date_str in HALF_DAYS:
        close_time = time(16, 0)  # 4:00 PM ET - no extended hours on half days

    # Check if within trading hours
    if current_time < TRADING_OPEN:
        return False, f"Pre-market - trading starts at {TRADING_OPEN.strftime('%I:%M %p')} ET"

    if current_time >= close_time:
        return False, f"After hours - trading closed at {close_time.strftime('%I:%M %p')} ET"

    return True, "Trading hours active"


def get_seconds_until_trading_opens() -> int:
    """
    Get the number of seconds until trading hours begin.

    Returns:
        Number of seconds until trading opens (0 if already open)
    """
    can_trade, _ = is_trading_hours()
    if can_trade:
        return 0

    now = get_current_et_time()

    # Start from today at market open
    check_date = now.replace(hour=9, minute=30, second=0, microsecond=0)

    # If we're past market open today, start from tomorrow
    if now.time() >= TRADING_OPEN:
        check_date = check_date + timedelta(days=1)

    # Find next trading day (skip weekends and holidays)
    for _ in range(10):  # Safety limit
        date_str = check_date.strftime("%Y-%m-%d")
        if check_date.weekday() < 5 and date_str not in MARKET_HOLIDAYS:
            break
        check_date = check_date + timedelta(days=1)

    delta = check_date - now
    return max(0, int(delta.total_seconds()))

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('SignalSynkStrategy')

# Environment variables
STRATEGY_PASSPHRASE = os.getenv('STRATEGY_PASSPHRASE')
SIGNALSYNK_WS_URL = os.getenv('SIGNALSYNK_WS_URL', 'wss://app.signalsynk.com/ws/strategy')
HEARTBEAT_INTERVAL = int(os.getenv('HEARTBEAT_INTERVAL', '30'))


class StrategyRunner:
    """Manages WebSocket connection and strategy execution."""

    def __init__(self, passphrase: str, ws_url: str):
        self.passphrase = passphrase
        self.ws_url = f"{ws_url}/{passphrase}"
        self.websocket = None
        self.strategy = TradingStrategy()
        self.connected = False
        self.authenticated = False
        self.running = True
        self.signals_sent = 0
        # Lock to prevent concurrent recv() calls on the websocket
        self._recv_lock = asyncio.Lock()

    async def connect(self):
        """Establish WebSocket connection to Signal Synk."""
        logger.info("Connecting to Signal Synk WebSocket...")

        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            self.connected = True
            logger.info("✅ Connected to Signal Synk!")

            # Send authentication message
            auth_msg = {
                "type": "AUTH",
                "passphrase": self.passphrase,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.websocket.send(json.dumps(auth_msg))
            logger.info("Sent authentication message")

            # Wait for auth response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10)
            response_data = json.loads(response)

            if response_data.get("type") == "AUTH_SUCCESS":
                self.authenticated = True
                logger.info(f"✅ Authenticated! Strategy ID: {response_data.get('strategy_id')}")
            else:
                logger.error(f"Authentication failed: {response_data}")
                raise Exception("Authentication failed")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected = False
            self.authenticated = False
            raise

    async def send_signal(self, signal: dict) -> bool:
        """Send a trade signal to Signal Synk.

        Note: We don't wait for acknowledgment here because message_handler()
        is running concurrently and will receive the response. Calling recv()
        here would conflict with message_handler's recv() call.
        """
        if not self.connected or not self.authenticated or not self.websocket:
            logger.error("Cannot send signal - not connected/authenticated")
            return False

        message = {
            "type": "TRADE_SIGNAL",
            "passphrase": self.passphrase,
            "ticker": signal['ticker'],
            "action": signal['action'],
            "quantity": signal['quantity'],
            "order_type": signal.get('order_type', 'MARKET'),
            "price": signal.get('price'),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "indicators": signal.get('indicators', {})
        }

        try:
            await self.websocket.send(json.dumps(message))
            self.signals_sent += 1
            logger.info(f"📤 Signal sent: {signal['action']} {signal['quantity']} {signal['ticker']}")
            # Acknowledgment will be received by message_handler()
            return True

        except Exception as e:
            logger.error(f"Failed to send signal: {e}")
            return False

    async def heartbeat_loop(self):
        """Send periodic heartbeats to keep connection alive."""
        while self.running and self.connected:
            try:
                heartbeat = {
                    "type": "HEARTBEAT",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "signals_sent": self.signals_sent
                }
                await self.websocket.send(json.dumps(heartbeat))
                logger.debug("💓 Heartbeat sent")
                await asyncio.sleep(HEARTBEAT_INTERVAL)
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                self.connected = False
                break

    async def strategy_loop(self):
        """Execute strategy analysis at regular intervals.

        Trading is only executed during market hours (9:30 AM - 6:00 PM ET).
        Outside of trading hours, the strategy will sleep until the next
        trading window opens.
        """
        interval = self.strategy.get_interval()
        logger.info(f"Starting strategy loop (interval: {interval}s)")

        while self.running and self.connected and self.authenticated:
            try:
                # Check if we're within trading hours
                can_trade, reason = is_trading_hours()

                if not can_trade:
                    # Calculate time until trading opens
                    wait_seconds = get_seconds_until_trading_opens()

                    if wait_seconds > 0:
                        # Log the wait (but not too frequently)
                        wait_hours = wait_seconds / 3600
                        logger.info(f"⏸️  Outside trading hours: {reason}")
                        logger.info(f"   Next trading window in {wait_hours:.1f} hours")

                        # Sleep for a shorter interval during off-hours
                        # Wake up periodically to check and send heartbeats
                        sleep_time = min(wait_seconds, 300)  # Max 5 min sleep
                        await asyncio.sleep(sleep_time)
                        continue

                # Run strategy analysis during trading hours
                signals = await self.strategy.analyze()

                # Send any generated signals
                for signal in signals:
                    await self.send_signal(signal)
                    await asyncio.sleep(0.5)  # Small delay between signals

                # Wait for next interval
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Strategy loop error: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    async def message_handler(self):
        """Handle incoming messages from Signal Synk.

        This is the ONLY coroutine that should call recv() on the websocket
        to avoid "cannot call recv while another coroutine is already running" errors.
        """
        while self.running and self.connected:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=60)
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "HEARTBEAT_ACK":
                    logger.debug("💓 Heartbeat acknowledged")
                elif msg_type == "SIGNAL_ACK":
                    # Signal was acknowledged by the server
                    executed = data.get("executed", 0)
                    failed = data.get("failed", 0)
                    logger.info(f"✅ Signal acknowledged - {executed} trades executed, {failed} failed")
                elif msg_type == "SIGNAL_ERROR":
                    # Signal processing failed
                    logger.error(f"❌ Signal error: {data.get('error')}")
                elif msg_type == "COMMAND":
                    await self.handle_command(data)
                elif msg_type == "ERROR":
                    logger.error(f"Server error: {data.get('error')}")
                else:
                    logger.debug(f"Received: {msg_type}")

            except asyncio.TimeoutError:
                continue  # No message, that's fine
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed by server")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"Message handler error: {e}")

    async def handle_command(self, data: dict):
        """Handle commands from Signal Synk server."""
        command = data.get("command", "")

        if command == "stop":
            logger.info("Received STOP command")
            self.running = False
        elif command == "restart":
            logger.info("Received RESTART command")
            self.running = False  # Will reconnect in outer loop
        elif command == "config":
            # Update configuration
            new_config = data.get("config", {})
            logger.info(f"Received config update: {new_config}")
        else:
            logger.warning(f"Unknown command: {command}")

    async def run(self):
        """Main run loop - connect and execute strategy."""
        await self.connect()

        # Run all tasks concurrently
        await asyncio.gather(
            self.heartbeat_loop(),
            self.strategy_loop(),
            self.message_handler(),
            return_exceptions=True
        )

    async def shutdown(self):
        """Graceful shutdown."""
        self.running = False
        if self.websocket:
            try:
                # Send status update before closing
                await self.websocket.send(json.dumps({
                    "type": "STATUS",
                    "status": "stopping",
                    "signals_sent": self.signals_sent
                }))
                await self.websocket.close()
            except:
                pass
        logger.info("Strategy shutdown complete")


async def main():
    """Main entry point."""
    if not STRATEGY_PASSPHRASE:
        logger.error("❌ STRATEGY_PASSPHRASE environment variable not set!")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🚀 Signal Synk Strategy Container Starting")
    logger.info(f"   Passphrase: {STRATEGY_PASSPHRASE[:8]}...")
    logger.info(f"   WebSocket: {SIGNALSYNK_WS_URL}")
    logger.info(f"   Heartbeat: {HEARTBEAT_INTERVAL}s")
    logger.info("=" * 60)

    runner = StrategyRunner(STRATEGY_PASSPHRASE, SIGNALSYNK_WS_URL)

    # Reconnection loop with exponential backoff
    retry_delay = 5
    max_retry_delay = 300

    while True:
        try:
            await runner.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            await runner.shutdown()
            break
        except Exception as e:
            logger.error(f"Runner error: {e}")
            logger.info(f"Reconnecting in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)

            # Reset runner for reconnection
            runner = StrategyRunner(STRATEGY_PASSPHRASE, SIGNALSYNK_WS_URL)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")

