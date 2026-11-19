"""HTTP client for the loopback-only standalone vision worker."""

from __future__ import annotations

import os
from urllib.parse import urlsplit
from typing import Any

import httpx


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


def _validate_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        parsed.port
    except ValueError as error:
        raise VisionWorkerConfigurationError(
            "VISION_WORKER_URL is malformed."
        ) from error
    if (
        parsed.scheme != "http"
        or hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise VisionWorkerConfigurationError(
            "VISION_WORKER_URL must use http://127.0.0.1 without a path."
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
        self._base_url = _validate_base_url(
            base_url or os.getenv("VISION_WORKER_URL", "http://127.0.0.1:8001")
        )
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
