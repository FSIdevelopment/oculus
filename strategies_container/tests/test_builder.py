"""Tests for builder.py — mock _run so no real Docker needed."""
import os
import sys

import pytest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import builder


def test_sanitize_tag():
    assert builder.sanitize_tag("My Strategy_V2") == "my-strategy-v2"
    assert builder.sanitize_tag("---test---") == "test"
    assert builder.sanitize_tag("Strategy@#$") == "strategy"
    assert builder.sanitize_tag("a" * 200) == "a" * 128


@pytest.mark.asyncio
@patch("builder._run", new_callable=AsyncMock)
@patch.dict(os.environ, {"DO_REGISTRY_TOKEN": "", "DO_REGISTRY_URL": "registry.digitalocean.com/test"})
async def test_build_and_push_success(mock_run):
    mock_run.return_value = (0, "sha256:abc", "")
    result = await builder.build_and_push(
        strategy_name="test-strat",
        version=2,
        files={"Dockerfile": "FROM python:3.11-slim\n"},
    )
    assert result["success"] is True
    assert result["image_tag"] == "registry.digitalocean.com/test/test-strat:2"
    assert result["error"] is None


@pytest.mark.asyncio
@patch("builder._run", new_callable=AsyncMock)
@patch.dict(os.environ, {"DO_REGISTRY_TOKEN": "", "DO_REGISTRY_URL": "registry.digitalocean.com/test"})
async def test_build_failure_returns_error(mock_run):
    mock_run.return_value = (1, "", "COPY failed: file not found")
    result = await builder.build_and_push(
        strategy_name="bad-strat",
        version=1,
        files={"Dockerfile": "FROM scratch\n"},
    )
    assert result["success"] is False
    assert result["image_tag"] is None
    assert "docker build failed" in result["error"]


@pytest.mark.asyncio
@patch("builder._run", new_callable=AsyncMock)
@patch.dict(os.environ, {"DO_REGISTRY_TOKEN": "mytoken", "DO_REGISTRY_URL": "registry.digitalocean.com/test"})
async def test_login_failure_short_circuits(mock_run):
    # First call is the login — make it fail; build should never be called
    mock_run.side_effect = [(1, "", "unauthorized"), (0, "", "")]
    result = await builder.build_and_push(
        strategy_name="strat",
        version=1,
        files={"Dockerfile": "FROM python:3.11-slim\n"},
    )
    assert result["success"] is False
    assert "Registry login failed" in result["error"]
    # Only 1 call (login), build was never attempted
    assert mock_run.call_count == 1


@pytest.mark.asyncio
@patch("builder._run", new_callable=AsyncMock)
@patch.dict(os.environ, {"DO_REGISTRY_TOKEN": "", "DO_REGISTRY_URL": "registry.digitalocean.com/test"})
async def test_path_traversal_files_skipped(mock_run):
    mock_run.return_value = (0, "", "")
    result = await builder.build_and_push(
        strategy_name="strat",
        version=1,
        files={
            "Dockerfile": "FROM python:3.11-slim\n",
            "../evil.py": "rm -rf /",
            "/etc/passwd": "bad",
        },
    )
    # Build should still succeed; malicious filenames are silently skipped
    assert result["success"] is True
