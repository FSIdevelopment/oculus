"""Unit tests for config normalization helpers.

These tests protect against unit drift where percent-like values are emitted as
whole percent points (e.g. 5.0 meaning 5%) instead of decimal fractions (0.05).
"""

import pytest

from app.services.build_orchestrator import BuildOrchestrator


def test_normalize_config_percent_fractions_converts_known_pct_fields():
    cfg = {
        "trading": {"position_size_pct": 10},
        "risk_management": {
            "stop_loss_pct": 5,
            "take_profit_pct": "4%",
            "trailing_stop_pct": "1.5",
            "max_daily_loss_pct": 2.0,
            "max_drawdown_pct": 20,
        },
    }

    out = BuildOrchestrator.normalize_config_percent_fractions(cfg)

    assert out["trading"]["position_size_pct"] == pytest.approx(0.10)
    assert out["risk_management"]["stop_loss_pct"] == pytest.approx(0.05)
    assert out["risk_management"]["take_profit_pct"] == pytest.approx(0.04)
    assert out["risk_management"]["trailing_stop_pct"] == pytest.approx(0.015)
    assert out["risk_management"]["max_daily_loss_pct"] == pytest.approx(0.02)
    assert out["risk_management"]["max_drawdown_pct"] == pytest.approx(0.20)


def test_normalize_config_percent_fractions_converts_worker_style_keys():
    cfg = {
        "risk_management": {
            # Worker-style keys without *_pct suffix
            "max_daily_loss": 5.0,
            "max_drawdown": "20",
            "max_position_size": "10%",
            # Already-fractional values should remain unchanged
            "stop_loss_pct": 0.05,
        }
    }

    out = BuildOrchestrator.normalize_config_percent_fractions(cfg)

    assert out["risk_management"]["max_daily_loss"] == pytest.approx(0.05)
    assert out["risk_management"]["max_drawdown"] == pytest.approx(0.20)
    assert out["risk_management"]["max_position_size"] == pytest.approx(0.10)
    assert out["risk_management"]["stop_loss_pct"] == pytest.approx(0.05)

