"""Build agent FastAPI service.

Receives strategy files from the Oculus backend and runs
docker build + push on this Droplet where Docker is available.
"""
import logging
import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import builder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Strategies Container Build Agent")

_API_KEY = os.environ.get("BUILD_API_KEY", "")


def _check_key(key: str) -> None:
    if not _API_KEY:
        raise HTTPException(status_code=500, detail="BUILD_API_KEY not configured on agent")
    if key != _API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


class BuildRequest(BaseModel):
    strategy_name: str
    version: int
    files: dict[str, str]


class BuildResponse(BaseModel):
    success: bool
    image_tag: Optional[str] = None
    error: Optional[str] = None


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
