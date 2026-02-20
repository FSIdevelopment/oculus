"""Dynamic license pricing based on strategy backtest performance.

Pricing formula
---------------
1. Extract ``total_return_pct`` (or ``total_return`` as fallback) and
   ``sharpe_ratio`` from the strategy's backtest_results JSON.
2. Normalise each metric to [0, 1] using the configured min/max thresholds
   (values outside the range are clamped).
3. Combine the two normalised scores using a weighted average.
4. Apply a power-curve transformation to shape the pricing distribution.
5. Interpolate between ``LICENSE_MONTHLY_MIN_PRICE`` and
   ``LICENSE_MONTHLY_MAX_PRICE`` and round to the nearest whole dollar.
6. Calculate the annual price as ``monthly * LICENSE_ANNUAL_MULTIPLIER``.
"""

from __future__ import annotations

from typing import Any

from app.config import settings


def _normalise(value: float, min_val: float, max_val: float) -> float:
    """Normalise *value* to [0, 1] clamped at the boundaries."""
    if max_val <= min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def calculate_license_price(
    backtest_results: dict[str, Any],
) -> tuple[int, int, float]:
    """Calculate monthly and annual license prices from backtest results.

    Parameters
    ----------
    backtest_results:
        The raw backtest results dict stored on the ``Strategy`` model.

    Returns
    -------
    tuple[int, int, float]
        ``(monthly_price, annual_price, performance_score)`` where
        ``performance_score`` is the final curved score in [0, 1].
    """
    # ── 1. Extract metrics ────────────────────────────────────────────────────
    # The field name differs between the container template (total_return_pct)
    # and the legacy worker results (total_return).  Check both.
    total_return: float = float(
        backtest_results.get("total_return_pct")
        or backtest_results.get("total_return")
        or 0.0
    )
    sharpe_ratio: float = float(backtest_results.get("sharpe_ratio") or 0.0)

    # ── 2. Normalise each metric to [0, 1] ────────────────────────────────────
    norm_return = _normalise(
        total_return,
        settings.PERF_MIN_RETURN,
        settings.PERF_MAX_RETURN,
    )
    norm_sharpe = _normalise(
        sharpe_ratio,
        settings.PERF_MIN_SHARPE,
        settings.PERF_MAX_SHARPE,
    )

    # ── 3. Weighted combination ───────────────────────────────────────────────
    total_weight = settings.PERF_RETURN_WEIGHT + settings.PERF_SHARPE_WEIGHT
    if total_weight <= 0:
        combined_score = 0.0
    else:
        combined_score = (
            norm_return * settings.PERF_RETURN_WEIGHT
            + norm_sharpe * settings.PERF_SHARPE_WEIGHT
        ) / total_weight

    # ── 4. Power-curve transformation ─────────────────────────────────────────
    # Exponent < 1 creates an ease-in curve: lower-performing strategies still
    # earn a meaningful price bump above the floor.
    curved_score = combined_score ** settings.PERF_CURVE_EXPONENT

    # ── 5. Map to price range ─────────────────────────────────────────────────
    price_range = settings.LICENSE_MONTHLY_MAX_PRICE - settings.LICENSE_MONTHLY_MIN_PRICE
    monthly_price = round(settings.LICENSE_MONTHLY_MIN_PRICE + curved_score * price_range)

    # ── 6. Annual price ───────────────────────────────────────────────────────
    annual_price = round(monthly_price * settings.LICENSE_ANNUAL_MULTIPLIER)

    return monthly_price, annual_price, round(curved_score, 4)

