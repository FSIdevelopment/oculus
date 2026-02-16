"""Strategy container entrypoint."""
import os
import sys
import json
import logging
import asyncio
import httpx
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Display terms of use warning
TERMS_OF_USE = """
================================================================================
                    OCULUS STRATEGY CONTAINER - TERMS OF USE
================================================================================
This container runs a proprietary trading strategy. By running this container,
you agree to:
1. Use this strategy only for authorized trading purposes
2. Not reverse-engineer, modify, or redistribute this strategy
3. Maintain confidentiality of strategy logic and parameters
4. Comply with all applicable financial regulations
5. Accept all trading risks and losses

For full terms, visit: https://oculusalgorithms.com/terms
================================================================================
"""

print(TERMS_OF_USE)


async def validate_license(license_id: str, api_url: str) -> dict:
    """Validate license with Oculus API."""
    logger.info(f"Validating license: {license_id}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{api_url}/api/licenses/{license_id}/validate",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            logger.info("License validation successful")
            return data
    except Exception as e:
        logger.error(f"License validation failed: {str(e)}")
        raise


async def stream_to_webhook(webhook_url: str, data: dict) -> None:
    """Stream strategy output to webhook URL."""
    if not webhook_url:
        logger.warning("No webhook URL configured, skipping output streaming")
        return

    logger.info(f"Streaming output to webhook: {webhook_url}")
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "strategy_output": data,
            }
            response = await client.post(webhook_url, json=payload, timeout=10.0)
            response.raise_for_status()
            logger.info("Output streamed successfully")
    except Exception as e:
        logger.error(f"Failed to stream output: {str(e)}")


async def run_strategy() -> dict:
    """Run the strategy and return output data."""
    logger.info("Starting strategy execution")

    # Get environment variables
    license_id = os.getenv("LICENSE_ID")
    webhook_url = os.getenv("WEBHOOK_URL")
    api_url = os.getenv("OCULUS_API_URL", "https://api.oculusalgorithms.com")

    if not license_id:
        raise ValueError("LICENSE_ID environment variable not set")

    # Validate license
    license_data = await validate_license(license_id, api_url)

    # Extract data provider keys from license validation response
    data_providers = license_data.get("data_providers", {})
    logger.info(f"Retrieved data provider keys: {list(data_providers.keys())}")

    # TODO: Import and run actual strategy logic here
    # For now, return placeholder output
    strategy_output = {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        "signals": [],
        "metrics": {
            "total_return": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
        },
    }

    # Stream output to webhook if configured
    if webhook_url:
        await stream_to_webhook(webhook_url, strategy_output)

    logger.info("Strategy execution completed")
    return strategy_output


async def main():
    """Main entrypoint."""
    try:
        output = await run_strategy()
        logger.info(f"Strategy output: {json.dumps(output, indent=2)}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Strategy execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

