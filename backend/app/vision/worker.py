"""Standalone HTTP worker for the database-configured camera registry.

Run independently from the main FastAPI application with
``python -m backend.app.vision.worker`` from the repository root.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .camera import CameraCapture, CameraManager, CameraUnavailable
from ..schemas.vision import (
    CameraConfigUpdate,
    CameraReconcileRequest,
    CameraStatusResponse,
)


logger = logging.getLogger(__name__)
WORKER_PORT = 8001


def _worker_token() -> str:
    return os.getenv("VISION_WORKER_TOKEN", "")


async def require_worker_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = _worker_token()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="VISION_WORKER_TOKEN is not configured.",
        )
    scheme, separator, supplied = (authorization or "").partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied, expected)
    ):
        raise HTTPException(
            status_code=401,
            detail="Worker control authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _get_camera(request: Request, camera_id: int) -> CameraCapture:
    try:
        return request.app.state.camera_manager.get(camera_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error


def _camera_unavailable(error: CameraUnavailable) -> HTTPException:
    return HTTPException(status_code=503, detail=str(error))


def create_worker_app(camera_manager: CameraManager | None = None) -> FastAPI:
    manager = camera_manager if camera_manager is not None else CameraManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            try:
                await asyncio.to_thread(manager.stop_all)
            except CameraUnavailable as error:
                logger.error("Camera shutdown did not complete cleanly: %s", error)

    app = FastAPI(title="SmartLane Vision Worker", lifespan=lifespan)
    app.state.camera_manager = manager

    @app.get(
        "/api/v1/cameras",
        response_model=list[CameraStatusResponse],
        dependencies=[Depends(require_worker_token)],
    )
    async def list_cameras(request: Request) -> list[dict[str, Any]]:
        return request.app.state.camera_manager.statuses()

    @app.put(
        "/api/v1/cameras/reconcile",
        response_model=list[CameraStatusResponse],
        dependencies=[Depends(require_worker_token)],
    )
    async def reconcile_cameras(
        request: Request,
        update: CameraReconcileRequest,
    ) -> list[dict[str, Any]]:
        try:
            return await asyncio.to_thread(
                request.app.state.camera_manager.reconcile,
                [camera.model_dump() for camera in update.cameras],
                update.active_camera_ids,
                update.retry_failed,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.patch(
        "/api/v1/cameras/{camera_id}/config",
        response_model=CameraStatusResponse,
        dependencies=[Depends(require_worker_token)],
    )
    async def configure_camera(
        request: Request,
        camera_id: int,
        update: CameraConfigUpdate,
    ) -> dict[str, Any]:
        camera = _get_camera(request, camera_id)
        try:
            return await asyncio.to_thread(
                camera.configure,
                update.model_dump(exclude_unset=True),
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except CameraUnavailable as error:
            raise _camera_unavailable(error) from error

    @app.get("/camera/{camera_id}/stream")
    async def stream_camera(request: Request, camera_id: int) -> StreamingResponse:
        camera = _get_camera(request, camera_id)
        if camera.status()["state"] != "running":
            raise HTTPException(status_code=503, detail="Camera is not running.")

        async def frames():
            sequence = 0
            while True:
                frame = await asyncio.to_thread(camera.next_frame, sequence)
                if frame is None:
                    if camera.status()["state"] != "running":
                        break
                    continue
                sequence, jpeg = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    return app


app = create_worker_app()


def main() -> None:
    token = _worker_token()
    if not token:
        raise RuntimeError("Set VISION_WORKER_TOKEN before starting the vision worker.")
    uvicorn.run(app, host="127.0.0.1", port=WORKER_PORT)


if __name__ == "__main__":
    main()
