"""Docker builder service for building and pushing strategy images."""
import asyncio
import json
import os
import re
import logging
import tempfile
import uuid as _uuid_module
from typing import Any, Dict, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild

logger = logging.getLogger(__name__)


class DockerBuilder:
    """Service for building and pushing Docker images to Digital Ocean Container Registry."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.registry_url = settings.DO_REGISTRY_URL
        self.registry_token = settings.DO_REGISTRY_TOKEN

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

    async def build_image(
        self,
        strategy: Strategy,
        build: StrategyBuild,
        strategy_output_dir: str,
    ) -> Optional[str]:
        """
        Build Docker image only (does not push).

        Args:
            strategy: Strategy model instance
            build: StrategyBuild model instance
            strategy_output_dir: Path to generated strategy directory

        Returns:
            Image tag on success, None on failure
        """
        try:
            logger.info(f"Building Docker image for strategy {strategy.name}")

            # Sanitize strategy name for Docker tag
            tag_name = self._sanitize_docker_tag(strategy.name)
            version = strategy.version or 1
            image_tag = f"{self.registry_url}/{tag_name}:{version}"
            latest_tag = f"{self.registry_url}/{tag_name}:latest"

            # Build image
            logger.info(f"Building image: {image_tag}")
            build_start = datetime.utcnow()
            build_cmd = f"docker build -t {image_tag} -t {latest_tag} {strategy_output_dir}"
            result = await self._run_command(build_cmd, timeout=600)
            build_duration = (datetime.utcnow() - build_start).total_seconds()
            logger.info(f"Docker build completed in {build_duration:.1f}s")

            if result != 0:
                error_msg = f"Docker build failed with code {result}"
                logger.error(error_msg)
                build.logs = f"{build.logs or ''}\n\n{error_msg}"
                await self.db.commit()
                return None

            logger.info(f"Docker build successful: {image_tag}")
            return image_tag

        except Exception as e:
            logger.error(f"Docker build failed: {str(e)}")
            build.logs = f"{build.logs or ''}\n\nDocker build error: {str(e)}"
            await self.db.commit()
            return None

    async def test_container(self, image_tag: str, strategy_output_dir: str) -> dict:
        """
        Test the Docker container by running a simple import test.

        Args:
            image_tag: Docker image tag to test
            strategy_output_dir: Path to strategy directory (for context)

        Returns:
            Dict with keys: passed (bool), output (str), error (str)
        """
        try:
            logger.info(f"Testing container: {image_tag}")

            # Run a simple test: import the strategy and instantiate it
            test_cmd = (
                f'docker run --rm '
                f'-e BACKTEST_MODE=true '
                f'-e LOG_LEVEL=INFO '
                f'{image_tag} '
                f'python -c "from strategy import TradingStrategy; s = TradingStrategy(); print(\'OK\')"'
            )

            process = await asyncio.create_subprocess_shell(
                test_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=60
                )

                stdout_str = stdout.decode() if stdout else ""
                stderr_str = stderr.decode() if stderr else ""

                # Check if test passed
                passed = (
                    process.returncode == 0
                    and "Error" not in stderr_str
                    and "Traceback" not in stderr_str
                )

                logger.info(f"Container test {'passed' if passed else 'failed'}")

                return {
                    "passed": passed,
                    "output": stdout_str,
                    "error": stderr_str if not passed else "",
                }

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.error("Container test timed out")
                return {
                    "passed": False,
                    "output": "",
                    "error": "Container test timed out after 60 seconds",
                }

        except Exception as e:
            logger.error(f"Container test failed: {str(e)}")
            return {
                "passed": False,
                "output": "",
                "error": str(e),
            }

    async def push_image(self, image_tag: str, build: StrategyBuild) -> bool:
        """
        Push Docker image to Docker Hub.

        Args:
            image_tag: Docker image tag to push
            build: StrategyBuild model instance

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pushing image to Digital Ocean Container Registry: {image_tag}")

            # Login to Digital Ocean Container Registry
            if self.registry_token:
                logger.info("Logging in to Digital Ocean Container Registry")
                login_start = datetime.utcnow()
                login_cmd = f"echo '{self.registry_token}' | docker login registry.digitalocean.com --username unused --password-stdin"
                result = await self._run_command(login_cmd, timeout=300)
                login_duration = (datetime.utcnow() - login_start).total_seconds()
                logger.info(f"Docker login completed in {login_duration:.1f}s")
                if result != 0:
                    error_msg = "Digital Ocean Container Registry login failed"
                    logger.error(error_msg)
                    build.logs = f"{build.logs or ''}\n\n{error_msg}"
                    await self.db.commit()
                    return False

            # Push versioned tag
            logger.info(f"Pushing image: {image_tag}")
            push_start = datetime.utcnow()
            push_cmd = f"docker push {image_tag}"
            result = await self._run_command(push_cmd, timeout=300)
            push_duration = (datetime.utcnow() - push_start).total_seconds()
            logger.info(f"Docker push completed in {push_duration:.1f}s")
            if result != 0:
                error_msg = f"Docker push failed with code {result}"
                logger.error(error_msg)
                build.logs = f"{build.logs or ''}\n\n{error_msg}"
                await self.db.commit()
                return False

            # Push latest tag
            latest_tag = image_tag.rsplit(":", 1)[0] + ":latest"
            push_latest_start = datetime.utcnow()
            push_latest_cmd = f"docker push {latest_tag}"
            result = await self._run_command(push_latest_cmd, timeout=300)
            push_latest_duration = (datetime.utcnow() - push_latest_start).total_seconds()
            logger.info(f"Docker push latest completed in {push_latest_duration:.1f}s")
            if result != 0:
                logger.warning("Failed to push latest tag, but versioned tag succeeded")

            logger.info(f"Docker push successful: {image_tag}")
            return True

        except Exception as e:
            logger.error(f"Docker push failed: {str(e)}")
            build.logs = f"{build.logs or ''}\n\nDocker push error: {str(e)}"
            await self.db.commit()
            return False

    async def build_and_push(
        self,
        strategy: Strategy,
        build: StrategyBuild,
        strategy_output_dir: str,
    ) -> bool:
        """
        Build Docker image and push to Docker Hub (backward compatibility).

        Args:
            strategy: Strategy model instance
            build: StrategyBuild model instance
            strategy_output_dir: Path to generated strategy directory

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build the image
            image_tag = await self.build_image(strategy, build, strategy_output_dir)
            if not image_tag:
                build.status = "failed"
                build.completed_at = datetime.utcnow()
                await self.db.commit()
                return False

            # Push the image
            push_success = await self.push_image(image_tag, build)
            if not push_success:
                build.status = "failed"
                build.completed_at = datetime.utcnow()
                await self.db.commit()
                return False

            # Update strategy record
            strategy.docker_registry = self.registry_url
            strategy.docker_image_url = image_tag
            build.status = "complete"
            build.completed_at = datetime.utcnow()

            await self.db.commit()
            logger.info(f"Docker build and push successful: {image_tag}")
            return True

        except Exception as e:
            logger.error(f"Docker build and push failed: {str(e)}")
            build.status = "failed"
            build.logs = f"{build.logs or ''}\n\nDocker build and push error: {str(e)}"
            build.completed_at = datetime.utcnow()
            await self.db.commit()
            return False

    async def remote_build_and_push(
        self,
        strategy: Strategy,
        build: StrategyBuild,
        files: dict,
    ) -> bool:
        """
        Send all strategy files to the remote build agent on the
        signalSynk-Strategies Droplet for Docker build and push.

        Args:
            strategy: Strategy model instance
            build: StrategyBuild model instance
            files: Dict of {filename: content} — all 10 strategy files from
                   BuildIteration.strategy_files

        Returns:
            True if the image was built and pushed successfully, False otherwise
        """
        import httpx

        if not settings.BUILD_AGENT_URL:
            logger.error("BUILD_AGENT_URL not configured — cannot build remotely")
            build.logs = f"{build.logs or ''}\n\nRemote build agent not configured (BUILD_AGENT_URL missing)"
            await self.db.commit()
            return False

        if not files:
            logger.error("No strategy files provided for remote build")
            build.logs = f"{build.logs or ''}\n\nNo strategy files available for Docker build"
            await self.db.commit()
            return False

        version = strategy.version or 1
        payload = {
            "strategy_name": strategy.name,
            "version": version,
            "files": files,
        }

        logger.info(
            "Sending build request to agent: strategy=%s version=%d files=%d",
            strategy.name,
            version,
            len(files),
        )

        try:
            async with httpx.AsyncClient(timeout=900.0) as client:
                response = await client.post(
                    f"{settings.BUILD_AGENT_URL}/build",
                    json=payload,
                    headers={"X-Build-Api-Key": settings.BUILD_AGENT_API_KEY},
                )

            if response.status_code != 200:
                error_msg = f"Build agent HTTP {response.status_code}: {response.text[:500]}"
                logger.error(error_msg)
                build.logs = f"{build.logs or ''}\n\n{error_msg}"
                await self.db.commit()
                return False

            result = response.json()

            if not result.get("success"):
                error_msg = f"Remote build failed: {result.get('error', 'unknown error')}"
                logger.error(error_msg)
                build.logs = f"{build.logs or ''}\n\n{error_msg}"
                await self.db.commit()
                return False

            image_tag = result["image_tag"]
            strategy.docker_registry = self.registry_url
            strategy.docker_image_url = image_tag
            build.status = "complete"
            build.completed_at = datetime.utcnow()
            await self.db.commit()

            logger.info("Remote build successful: %s", image_tag)
            return True

        except Exception as e:
            error_msg = f"Remote build agent error: {str(e)}"
            logger.error(error_msg)
            build.logs = f"{build.logs or ''}\n\n{error_msg}"
            await self.db.commit()
            return False

    async def run_backtest_verification(
        self,
        image_tag: str,
        strategy_id: str,
        backtest_period: str = "1Y",
    ) -> Optional[Dict[str, Any]]:
        """
        Run a full backtest inside the built container and return the results.

        Executes backtest.py inside the container, copies the output JSON file
        back to the host, and returns the parsed results for database storage.
        The container is NOT given DATABASE_URL — builds.py persists the results
        via SQLAlchemy after this method returns.

        Args:
            image_tag: Docker image tag to run backtest against
            strategy_id: Strategy UUID (passed as STRATEGY_ID env var)
            backtest_period: Historical period to backtest over (default: 1Y)

        Returns:
            Parsed backtest results dict, or None on failure
        """
        container_name = f"backtest-verify-{_uuid_module.uuid4().hex[:8]}"

        try:
            logger.info(
                f"Starting backtest verification: {image_tag} (period={backtest_period})"
            )

            # Run backtest.py inside the container.
            # DATABASE_URL is intentionally omitted so the container skips its own
            # DB update — builds.py will persist the results via SQLAlchemy instead.
            run_cmd = (
                f"docker run --name {container_name} "
                f"-e BACKTEST_MODE=true "
                f"-e BACKTEST_PERIOD={backtest_period} "
                f"-e STRATEGY_ID={strategy_id} "
                f"-e ALPHAVANTAGE_API_KEY={settings.ALPHAVANTAGE_API_KEY} "
                f"-e POLYGON_API_KEY={settings.POLYGON_API_KEY} "
                f"-e LOG_LEVEL=INFO "
                f"{image_tag} "
                f"python backtest.py"
            )

            process = await asyncio.create_subprocess_shell(
                run_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=600,  # 10-minute hard limit
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                await self._run_command(f"docker rm -f {container_name}", timeout=30)
                logger.error("Backtest verification timed out after 10 minutes")
                return None

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            if process.returncode != 0:
                logger.error(
                    f"Backtest container exited with code {process.returncode}. "
                    f"stderr: {stderr_str[:2000]}"
                )
                await self._run_command(f"docker rm -f {container_name}", timeout=30)
                return None

            logger.info(
                f"Backtest container finished. Output preview: {stdout_str[:500]}"
            )

            # Copy results file from the stopped container to a temp directory
            results: Optional[Dict[str, Any]] = None
            with tempfile.TemporaryDirectory() as tmp_dir:
                host_results_path = os.path.join(tmp_dir, "backtest_results.json")
                cp_result = await self._run_command(
                    f"docker cp {container_name}:/app/backtest_results.json {host_results_path}",
                    timeout=30,
                )

                if cp_result != 0:
                    logger.error(
                        "docker cp failed — backtest_results.json not found in container"
                    )
                elif not os.path.exists(host_results_path):
                    logger.error("backtest_results.json missing after docker cp")
                else:
                    with open(host_results_path, "r") as fh:
                        results = json.load(fh)

            # Always clean up the stopped container
            await self._run_command(f"docker rm {container_name}", timeout=30)

            if results is None:
                return None

            # A results dict with only an "error" key means the backtest itself failed
            if "error" in results and not results.get("total_return_pct"):
                logger.error(f"Backtest returned error: {results.get('error')}")
                return None

            logger.info(
                f"Backtest verification complete — "
                f"return={results.get('total_return_pct', 'N/A')}%, "
                f"trades={results.get('total_trades', 'N/A')}"
            )
            return results

        except Exception as exc:
            logger.error(f"Backtest verification failed: {exc}", exc_info=True)
            # Best-effort container cleanup
            try:
                await self._run_command(f"docker rm -f {container_name}", timeout=30)
            except Exception:
                pass
            return None

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

