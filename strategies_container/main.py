"""Build and container-management agent FastAPI service.

Receives requests from the SignalSynk backend and executes Docker operations
on this Droplet where the Docker daemon lives.

Endpoints:
  GET  /health                       - liveness probe (no auth)
  POST /build                        - build + push a strategy image
  POST /start                        - start a strategy container
  POST /stop                         - stop + remove a strategy container
  POST /restart                      - restart a strategy container
  GET  /logs/{strategy_id}           - tail container logs
  GET  /status/{strategy_id}         - container status
  GET  /list                         - list all managed containers
  POST /pull                         - pull an image from a registry
  GET  /image-exists                 - check whether an image is present locally
"""
import logging
import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import builder
import containers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Strategies Container Agent")

_API_KEY = os.environ.get("BUILD_API_KEY", "")


def _check_key(key: str) -> None:
    if not _API_KEY:
        raise HTTPException(status_code=500, detail="BUILD_API_KEY not configured on agent")
    if key != _API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Build models (existing)
# ---------------------------------------------------------------------------

class BuildRequest(BaseModel):
    strategy_name: str
    version: int
    files: dict[str, str]


class BuildResponse(BaseModel):
    success: bool
    image_tag: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Container models (new)
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    strategy_id: int
    image_name: str
    env_vars: dict[str, str] = {}


class StartResponse(BaseModel):
    success: bool
    container_id: Optional[str] = None
    error: Optional[str] = None


class StopRequest(BaseModel):
    strategy_id: int


class StopResponse(BaseModel):
    success: bool
    error: Optional[str] = None


class RestartRequest(BaseModel):
    strategy_id: int
    image_name: str
    env_vars: dict[str, str] = {}


class RestartResponse(BaseModel):
    success: bool
    container_id: Optional[str] = None
    error: Optional[str] = None


class LogsResponse(BaseModel):
    success: bool
    logs: Optional[str] = None
    error: Optional[str] = None


class StatusResponse(BaseModel):
    success: bool
    status: Optional[str] = None
    container_id: Optional[str] = None
    error: Optional[str] = None


class ContainerItem(BaseModel):
    container_id: str
    name: str
    status: str
    image: str


class ListResponse(BaseModel):
    success: bool
    containers: list[ContainerItem] = []
    error: Optional[str] = None


class PullRequest(BaseModel):
    image_ref: str


class PullResponse(BaseModel):
    success: bool
    image: Optional[str] = None
    error: Optional[str] = None


class ImageExistsResponse(BaseModel):
    success: bool
    exists: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Existing endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/build", response_model=BuildResponse)
async def build(
    request: BuildRequest,
    x_build_api_key: str = Header(..., alias="X-Build-Api-Key"),
):
    _check_key(x_build_api_key)

    if not request.files:
        raise HTTPException(status_code=422, detail="files dict is empty")

    logger.info(
        "Build request: strategy=%s version=%d files=%d",
        request.strategy_name,
        request.version,
        len(request.files),
    )

    result = await builder.build_and_push(
        strategy_name=request.strategy_name,
        version=request.version,
        files=request.files,
    )
    return BuildResponse(**result)


# ---------------------------------------------------------------------------
# Container management endpoints
# ---------------------------------------------------------------------------

@app.post("/start", response_model=StartResponse)
async def start_container(
    request: StartRequest,
    x_build_api_key: str = Header(..., alias="X-Build-Api-Key"),
):
    """Start a strategy container on this droplet."""
    _check_key(x_build_api_key)
    logger.info(
        "Start request: strategy_id=%d image=%s",
        request.strategy_id,
        request.image_name,
    )
    result = await containers.start(
        strategy_id=request.strategy_id,
        image_name=request.image_name,
        env_vars=request.env_vars,
    )
    return StartResponse(**result)


@app.post("/stop", response_model=StopResponse)
async def stop_container(
    request: StopRequest,
    x_build_api_key: str = Header(..., alias="X-Build-Api-Key"),
):
    """Stop and remove a strategy container."""
    _check_key(x_build_api_key)
    logger.info("Stop request: strategy_id=%d", request.strategy_id)
    result = await containers.stop(strategy_id=request.strategy_id)
    return StopResponse(**result)


@app.post("/restart", response_model=RestartResponse)
async def restart_container(
    request: RestartRequest,
    x_build_api_key: str = Header(..., alias="X-Build-Api-Key"),
):
    """Stop an existing strategy container and start it again fresh."""
    _check_key(x_build_api_key)
    logger.info(
        "Restart request: strategy_id=%d image=%s",
        request.strategy_id,
        request.image_name,
    )
    result = await containers.restart(
        strategy_id=request.strategy_id,
        image_name=request.image_name,
        env_vars=request.env_vars,
    )
    return RestartResponse(**result)


@app.get("/logs/{strategy_id}", response_model=LogsResponse)
async def get_logs(
    strategy_id: int,
    tail: int = 100,
    x_build_api_key: str = Header(..., alias="X-Build-Api-Key"),
):
    """Retrieve recent logs from a strategy container."""
    _check_key(x_build_api_key)
    result = await containers.get_logs(strategy_id=strategy_id, tail=tail)
    return LogsResponse(**result)


@app.get("/status/{strategy_id}", response_model=StatusResponse)
async def get_status(
    strategy_id: int,
    x_build_api_key: str = Header(..., alias="X-Build-Api-Key"),
):
    """Get the current Docker status of a strategy container."""
    _check_key(x_build_api_key)
    result = await containers.get_status(strategy_id=strategy_id)
    return StatusResponse(**result)


@app.get("/list", response_model=ListResponse)
async def list_containers(
    x_build_api_key: str = Header(..., alias="X-Build-Api-Key"),
):
    """List all SignalSynk-managed containers on this droplet."""
    _check_key(x_build_api_key)
    result = await containers.list_all()
    return ListResponse(**result)


@app.post("/pull", response_model=PullResponse)
async def pull_image(
    request: PullRequest,
    x_build_api_key: str = Header(..., alias="X-Build-Api-Key"),
):
    """Pull a Docker image from a registry onto this droplet."""
    _check_key(x_build_api_key)
    logger.info("Pull request: image_ref=%s", request.image_ref)
    result = await containers.pull(image_ref=request.image_ref)
    return PullResponse(**result)


@app.get("/image-exists", response_model=ImageExistsResponse)
async def image_exists(
    image_ref: str,
    x_build_api_key: str = Header(..., alias="X-Build-Api-Key"),
):
    """Check whether a Docker image is already present on this droplet."""
    _check_key(x_build_api_key)
    result = await containers.image_exists(image_ref=image_ref)
    return ImageExistsResponse(**result)
