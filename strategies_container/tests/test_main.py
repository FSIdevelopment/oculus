"""Tests for main.py FastAPI endpoints."""
import os
import sys

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["BUILD_API_KEY"] = "test-secret"

import main

client = TestClient(main.app)

VALID_PAYLOAD = {
    "strategy_name": "momentum-spy",
    "version": 1,
    "files": {
        "Dockerfile": "FROM python:3.11-slim\nCMD [\"python\", \"main.py\"]",
        "strategy.py": "class TradingStrategy: pass",
    },
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_build_rejects_missing_key():
    response = client.post("/build", json=VALID_PAYLOAD)
    assert response.status_code == 422  # missing required header


def test_build_rejects_wrong_key():
    response = client.post(
        "/build",
        json=VALID_PAYLOAD,
        headers={"X-Build-Api-Key": "wrong-secret"},
    )
    assert response.status_code == 401


@patch("main.builder.build_and_push", new_callable=AsyncMock)
def test_build_success(mock_build):
    mock_build.return_value = {
        "success": True,
        "image_tag": "registry.digitalocean.com/test/momentum-spy:1",
        "error": None,
    }
    response = client.post(
        "/build",
        json=VALID_PAYLOAD,
        headers={"X-Build-Api-Key": "test-secret"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["image_tag"] == "registry.digitalocean.com/test/momentum-spy:1"
    mock_build.assert_awaited_once_with(
        strategy_name="momentum-spy",
        version=1,
        files=VALID_PAYLOAD["files"],
    )


@patch("main.builder.build_and_push", new_callable=AsyncMock)
def test_build_failure_propagated(mock_build):
    mock_build.return_value = {
        "success": False,
        "image_tag": None,
        "error": "docker build failed: syntax error",
    }
    response = client.post(
        "/build",
        json=VALID_PAYLOAD,
        headers={"X-Build-Api-Key": "test-secret"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "docker build failed" in data["error"]


def test_build_rejects_empty_files():
    payload = {**VALID_PAYLOAD, "files": {}}
    response = client.post(
        "/build",
        json=payload,
        headers={"X-Build-Api-Key": "test-secret"},
    )
    assert response.status_code == 422
