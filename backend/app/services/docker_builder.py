"""Docker builder service for building and pushing strategy images."""
import asyncio
import re
import logging
from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild

logger = logging.getLogger(__name__)


class DockerBuilder:
    """Service for building and pushing Docker images to Docker Hub."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.docker_hub_username = settings.DOCKER_HUB_USERNAME
        self.docker_hub_pat = settings.DOCKER_HUB_PAT

    def _sanitize_docker_tag(self, name: str) -> str:
        """Sanitize strategy name for Docker tag (lowercase, alphanumeric, hyphens)."""
        # Convert to lowercase and replace spaces/underscores with hyphens
        tag = name.lower().replace("_", "-").replace(" ", "-")
        # Remove non-alphanumeric except hyphens
        tag = re.sub(r"[^a-z0-9-]", "", tag)
        # Remove leading/trailing hyphens
        tag = tag.strip("-")
        # Limit to 128 chars (Docker tag limit)
        return tag[:128] or "strategy"

    async def build_and_push(
        self,
        strategy: Strategy,
        build: StrategyBuild,
        strategy_output_dir: str,
    ) -> bool:
        """
        Build Docker image and push to Docker Hub.

        Args:
            strategy: Strategy model instance
            build: StrategyBuild model instance
            strategy_output_dir: Path to generated strategy directory

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting Docker build for strategy {strategy.name}")

            # Sanitize strategy name for Docker tag
            tag_name = self._sanitize_docker_tag(strategy.name)
            version = strategy.version or 1
            image_tag = f"{self.docker_hub_username}/{tag_name}:{version}"
            latest_tag = f"{self.docker_hub_username}/{tag_name}:latest"

            # Build image
            logger.info(f"Building image: {image_tag}")
            build_start = datetime.utcnow()
            build_cmd = f"docker build -t {image_tag} -t {latest_tag} {strategy_output_dir}"
            result = await self._run_command(build_cmd, timeout=600)
            build_duration = (datetime.utcnow() - build_start).total_seconds()
            logger.info(f"Docker build completed in {build_duration:.1f}s")
            if result != 0:
                raise RuntimeError(f"Docker build failed with code {result}")

            # Login to Docker Hub
            if self.docker_hub_pat:
                logger.info("Logging in to Docker Hub")
                login_start = datetime.utcnow()
                login_cmd = f"echo '{self.docker_hub_pat}' | docker login -u {self.docker_hub_username} --password-stdin"
                result = await self._run_command(login_cmd, timeout=300)
                login_duration = (datetime.utcnow() - login_start).total_seconds()
                logger.info(f"Docker login completed in {login_duration:.1f}s")
                if result != 0:
                    raise RuntimeError("Docker login failed")

            # Push image
            logger.info(f"Pushing image: {image_tag}")
            push_start = datetime.utcnow()
            push_cmd = f"docker push {image_tag}"
            result = await self._run_command(push_cmd, timeout=300)
            push_duration = (datetime.utcnow() - push_start).total_seconds()
            logger.info(f"Docker push completed in {push_duration:.1f}s")
            if result != 0:
                raise RuntimeError(f"Docker push failed with code {result}")

            # Push latest tag
            push_latest_start = datetime.utcnow()
            push_latest_cmd = f"docker push {latest_tag}"
            result = await self._run_command(push_latest_cmd, timeout=300)
            push_latest_duration = (datetime.utcnow() - push_latest_start).total_seconds()
            logger.info(f"Docker push latest completed in {push_latest_duration:.1f}s")
            if result != 0:
                logger.warning(f"Failed to push latest tag, but versioned tag succeeded")

            # Update strategy record
            strategy.docker_registry = self.docker_hub_username
            strategy.docker_image_url = image_tag
            build.status = "complete"
            build.completed_at = datetime.utcnow()

            await self.db.commit()
            logger.info(f"Docker build successful: {image_tag}")
            return True

        except Exception as e:
            logger.error(f"Docker build failed: {str(e)}")
            build.status = "failed"
            build.logs = f"{build.logs or ''}\n\nDocker build error: {str(e)}"
            build.completed_at = datetime.utcnow()
            await self.db.commit()
            return False

    async def _run_command(self, cmd: str, timeout: int = 600) -> int:
        """
        Run shell command asynchronously with timeout.

        Args:
            cmd: Shell command to execute
            timeout: Timeout in seconds (default 600s)

        Returns:
            Process return code

        Raises:
            RuntimeError: If command times out
        """
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            # Log stdout at debug level
            if stdout:
                logger.debug(f"Command stdout: {stdout.decode()}")

            # Log stderr at debug level
            if stderr:
                logger.debug(f"Command stderr: {stderr.decode()}")

            return process.returncode

        except asyncio.TimeoutError:
            # Kill the process if it times out
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"Command timed out after {timeout}s: {cmd[:100]}..."
            )

