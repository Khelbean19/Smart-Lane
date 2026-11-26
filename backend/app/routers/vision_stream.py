"""Same-origin proxy for vision-worker MJPEG camera streams.

The vision worker listens only on loopback (``VISION_WORKER_HOST``), so a
browser cannot reach it when the app runs in a container where a single port
is published (for example Hugging Face Spaces on port 7860).  This router
exposes the worker's ``GET /camera/{camera_id}/stream`` on the main backend
and forwards the multipart MJPEG bytes unchanged.

Because the route is public-facing, it requires a signed-in, active user.
``<img>`` tags cannot send an ``Authorization`` header, so the access token is
read from the ``access_token`` query parameter instead and validated exactly
like a bearer token (see ``get_current_user``).
"""

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from ..services.vision_worker_client import VisionWorkerClient
from .auth import get_current_user
from .vision import get_vision_worker_client


router = APIRouter()

# Tests may replace this with an ``httpx.MockTransport``.
_transport: httpx.AsyncBaseTransport | None = None

_DEFAULT_MEDIA_TYPE = "multipart/x-mixed-replace; boundary=frame"


def get_stream_user(
    access_token: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    """Resolve the ``access_token`` query parameter to an active user."""
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to view camera streams.",
        )
    return get_current_user(access_token)


@router.get("/camera/{camera_id}/stream", include_in_schema=False)
async def proxy_camera_stream(
    camera_id: int,
    current_user: Annotated[dict[str, Any], Depends(get_stream_user)],
    client: Annotated[VisionWorkerClient, Depends(get_vision_worker_client)],
) -> StreamingResponse:
    """Forward one camera's MJPEG stream from the loopback vision worker."""
    # The read timeout is disabled because an MJPEG stream never finishes.
    http = httpx.AsyncClient(
        transport=_transport,
        timeout=httpx.Timeout(10.0, read=None),
    )
    try:
        response = await http.send(
            http.build_request("GET", f"{client._base_url}/camera/{camera_id}/stream"),
            stream=True,
        )
    except httpx.RequestError as error:
        await http.aclose()
        raise HTTPException(
            status_code=503, detail="The vision worker is unavailable."
        ) from error

    if response.status_code != 200:
        try:
            await response.aread()
            data = response.json()
            detail = data.get("detail") if isinstance(data, dict) else None
        except ValueError:
            detail = None
        finally:
            await response.aclose()
            await http.aclose()
        status_code = response.status_code if response.status_code in (404, 503) else 502
        raise HTTPException(
            status_code=status_code,
            detail=str(detail or "Vision worker stream request failed."),
        )

    async def body():
        # Runs on normal completion and when the browser disconnects.
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
            await http.aclose()

    return StreamingResponse(
        body(),
        media_type=response.headers.get("content-type", _DEFAULT_MEDIA_TYPE),
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Accel-Buffering": "no",
        },
    )
