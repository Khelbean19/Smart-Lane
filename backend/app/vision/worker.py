"""Standalone HTTP worker for the database-configured camera registry.

Run independently from the main FastAPI application with
``python -m backend.app.vision.worker`` from the repository root.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Annotated, Any

import uvicorn
from loguru import logger
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .camera import CameraCapture, CameraManager, CameraUnavailable
from ..schemas.vision import (
    CameraConfigUpdate,
    CameraReconcileRequest,
    CameraStatusResponse,
    CameraInferenceResult,
    InferenceStatusResponse,
)
from .detector import YOLODetector, get_detector_config
from .inference import CameraInferenceRunner
from dotenv import load_dotenv


load_dotenv()


def _validate_worker_host(value: str | None) -> str:
    if value is None or not value.strip():
        raise RuntimeError("Set VISION_WORKER_HOST before starting the vision worker.")
    host = value.strip()
    if (
        "://" in host
        or any(character in host for character in "/?#@:")
        or not re.fullmatch(r"[A-Za-z0-9._-]+", host)
    ):
        raise RuntimeError("VISION_WORKER_HOST must be a hostname, not a URL.")
    return host


def _validate_worker_post(value: str | None) -> int:
    if value is None or not value.strip():
        raise RuntimeError("Set VISION_WORKER_PORT before starting the vision worker.")
    try:
        port = int(value.strip())
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "VISION_WORKER_PORT must be an integer from 1 to 65535."
        ) from error
    if not 1 <= port <= 65535:
        raise RuntimeError(
            "VISION_WORKER_PORT must be an integer from 1 to 65535."
        )
    return port


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


def create_worker_app(
    camera_manager: CameraManager | None = None,
    detector: YOLODetector | None = None,
    inference_runner: CameraInferenceRunner | None = None,
) -> FastAPI:
    manager = camera_manager if camera_manager is not None else CameraManager()
    config = get_detector_config()
    detector = detector or YOLODetector(
        model_path=config["model_path"],
        confidence=config["confidence"],
        iou=config["iou"],
        image_size=config["image_size"],
        device=config["device"],
        classes=config["classes"],
    )
    runner = inference_runner or CameraInferenceRunner(
        detector,
        inference_fps=config["inference_fps"],
    )

    @asynccontextmanager
    async def lifespan(worker_app: FastAPI):
        try:
            detector.warmup()
            yield
        finally:
            runner.stop_all()
            detector.unload()
            try:
                await asyncio.to_thread(manager.stop_all)
            except CameraUnavailable as error:
                logger.error("Camera shutdown did not complete cleanly: %s", error)

    app = FastAPI(
        title="SmartLane Vision Worker",
        description="Internal control and camera-stream API for SmartLane vision processing.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.camera_manager = manager
    app.state.detector = detector
    app.state.inference_runner = runner
    

    @app.get(
        "/api/v1/cameras",
        response_model=list[CameraStatusResponse],
        dependencies=[Depends(require_worker_token)],
        tags=["Cameras"],
        summary="List registered cameras",
        responses={
            401: {"description": "Missing or invalid worker bearer token"},
            503: {"description": "Worker bearer token is not configured"},
        },
    )
    async def list_cameras(request: Request) -> list[dict[str, Any]]:
        """Return the status and active configuration of every registered camera."""
        return request.app.state.camera_manager.statuses()

    @app.put(
        "/api/v1/cameras/reconcile",
        response_model=list[CameraStatusResponse],
        dependencies=[Depends(require_worker_token)],
        tags=["Cameras"],
        summary="Reconcile the camera registry",
        responses={
            401: {"description": "Missing or invalid worker bearer token"},
            422: {"description": "Invalid registry or active-camera configuration"},
            503: {"description": "Worker bearer token is not configured"},
        },
    )
    async def reconcile_cameras(
        request: Request,
        update: CameraReconcileRequest,
    ) -> list[dict[str, Any]]:
        """Register cameras, update their settings, and reconcile active streams."""
        try:
            statuses = await asyncio.to_thread(
                request.app.state.camera_manager.reconcile,
                [camera.model_dump() for camera in update.cameras],
                update.active_camera_ids,
                update.retry_failed,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        active_ids = {
            status["camera_id"]
            for status in statuses
            if status["state"] == "running"
        }
        cameras = {
            camera_id: request.app.state.camera_manager.get(camera_id)
            for camera_id in active_ids
        }
        request.app.state.inference_runner.sync(active_ids, cameras)
        return statuses

    @app.get(
        "/api/v1/inference/status",
        response_model=InferenceStatusResponse,
        dependencies=[Depends(require_worker_token)],
        tags=["Inference"],
        summary="Get detector and per-camera inference status",
    )
    async def inference_status(request: Request) -> dict[str, Any]:
        return request.app.state.inference_runner.status()

    @app.get(
        "/api/v1/cameras/{camera_id}/detections/latest",
        response_model=CameraInferenceResult,
        dependencies=[Depends(require_worker_token)],
        tags=["Inference"],
        summary="Get the latest detections for a camera",
        responses={
            404: {"description": "Unknown or inactive camera"},
        },
    )
    async def latest_detections(
        request: Request,
        camera_id: int,
    ) -> dict[str, Any]:
        try:
            request.app.state.camera_manager.get(camera_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error.args[0])) from error

        result = request.app.state.inference_runner.latest(camera_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Inference is not running for this camera.",
            )
        return result

    @app.patch(
        "/api/v1/cameras/{camera_id}/config",
        response_model=CameraStatusResponse,
        dependencies=[Depends(require_worker_token)],
        tags=["Cameras"],
        summary="Update camera configuration",
        responses={
            401: {"description": "Missing or invalid worker bearer token"},
            404: {"description": "Unknown camera ID"},
            422: {"description": "Invalid or empty camera configuration"},
            503: {"description": "Camera unavailable or worker bearer token is not configured"},
        },
    )
    async def configure_camera(
        request: Request,
        camera_id: int,
        update: CameraConfigUpdate,
    ) -> dict[str, Any]:
        """Apply non-null or explicitly requested settings to one registered camera."""
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

    @app.get(
        "/camera/{camera_id}/stream",
        tags=["Streams"],
        summary="Stream MJPEG camera frames",
        responses={
            200: {
                "description": "Multipart MJPEG frame stream",
                "content": {
                    "multipart/x-mixed-replace": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            },
            404: {"description": "Unknown camera ID"},
            503: {"description": "The camera is not running"},
        },
    )
    async def stream_camera(request: Request, camera_id: int) -> StreamingResponse:
        """Stream JPEG frames for an active camera as a multipart MJPEG response."""
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
    host = _validate_worker_host(os.getenv("VISION_WORKER_HOST"))
    port = _validate_worker_post(os.getenv("VISION_WORKER_PORT"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
