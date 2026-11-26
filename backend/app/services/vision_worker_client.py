"""HTTP client for the loopback-only standalone vision worker."""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit
from typing import Any

import httpx
import loguru


class VisionWorkerError(RuntimeError):
    """Base error for communication or configuration failures."""


class VisionWorkerUnavailable(VisionWorkerError):
    """The worker could not be reached."""


class VisionWorkerConfigurationError(VisionWorkerError):
    """The local worker URL or shared control credential is invalid."""


class VisionWorkerResponseError(VisionWorkerError):
    """The worker returned an HTTP error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code


class VisionWorkerProtocolError(VisionWorkerError):
    """The worker returned a response that was not valid JSON."""


def _validate_worker_host(value: str) -> str:
    host = value.strip()
    if not host:
        raise VisionWorkerConfigurationError(
            "VISION_WORKER_HOST must be a non-empty hostname."
        )
    if (
        "://" in host
        or any(character in host for character in "/?#@")
        or ":" in host
        or not re.fullmatch(r"[A-Za-z0-9._-]+", host)
    ):
        raise VisionWorkerConfigurationError(
            "VISION_WORKER_HOST must be a hostname, not a URL."
        )
    if host.lower() not in ("localhost", "127.0.0.1", "::1"):
        raise VisionWorkerConfigurationError(
            "VISION_WORKER_HOST must use a loopback hostname."
        )
    return "127.0.0.1" if host.lower() == "localhost" else host


def _validate_worker_post(value: str) -> int:
    try:
        port = int(value.strip())
    except (AttributeError, TypeError, ValueError) as error:
        raise VisionWorkerConfigurationError(
            "VISION_WORKER_PORT must be an integer from 1 to 65535."
        ) from error
    if not 1 <= port <= 65535:
        raise VisionWorkerConfigurationError(
            "VISION_WORKER_PORT must be an integer from 1 to 65535."
        )
    return port


def _validate_base_url(value: str) -> str:
    if "://" not in value:
        raise VisionWorkerConfigurationError(
            "VISION_WORKER_HOST must be configured as a hostname."
        )
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise VisionWorkerConfigurationError(
            "Worker base URL is malformed."
        ) from error
    if (
        parsed.scheme != "http"
        or hostname not in ("localhost", "127.0.0.1", "::1")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or port is None
        or not 1 <= port <= 65535
    ):
        raise VisionWorkerConfigurationError(
            "Worker base URL must be a loopback HTTP URL with a valid port."
        )
    return value.rstrip("/")


class VisionWorkerClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
    ) -> None:
        if base_url is not None:
            self._base_url = _validate_base_url(base_url)
            self._token = token if token is not None else os.getenv("VISION_WORKER_TOKEN", "")
            self._transport = transport
            self._timeout = timeout
            return

        _vision_host = os.getenv("VISION_WORKER_HOST")
        _vision_post = os.getenv("VISION_WORKER_PORT")
        if not _vision_host:
            raise VisionWorkerConfigurationError(
                "VISION_WORKER_HOST is not configured."
            )
        if not _vision_post:
            raise VisionWorkerConfigurationError(
                "VISION_WORKER_PORT is not configured."
            )
        host = _validate_worker_host(_vision_host)
        port = _validate_worker_post(_vision_post)
        self._base_url = f"http://{host}:{port}"
        self._token = token if token is not None else os.getenv("VISION_WORKER_TOKEN", "")
        self._transport = transport
        self._timeout = timeout

    async def request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        if not self._token:
            raise VisionWorkerConfigurationError(
                "VISION_WORKER_TOKEN must be configured for worker control."
            )

        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=self._timeout,
            ) as client:
                response = await client.request(
                    method,
                    f"{self._base_url}/{path.lstrip('/')}",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._token}"},
                )
        except httpx.RequestError as error:
            raise VisionWorkerUnavailable("The vision worker is unavailable.") from error

        if not response.is_success:
            try:
                data = response.json()
                detail = (
                    data.get("detail", "Vision worker request failed.")
                    if isinstance(data, dict)
                    else "Vision worker request failed."
                )
            except ValueError:
                detail = "Vision worker request failed."
            raise VisionWorkerResponseError(response.status_code, str(detail))

        try:
            return response.json()
        except ValueError as error:
            raise VisionWorkerProtocolError(
                "The vision worker returned an invalid response."
            ) from error

    async def get_inference_status(self) -> dict[str, Any]:
        return await self.request_json("GET", "/api/v1/inference/status")

    async def get_latest_detections(self, camera_id: int) -> dict[str, Any]:
        return await self.request_json(
            "GET",
            f"/api/v1/cameras/{camera_id}/detections/latest",
        )
