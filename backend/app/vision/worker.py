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
from threading import RLock
from typing import Annotated, Any

from dotenv import load_dotenv
from loguru import logger
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from psycopg2.extras import RealDictCursor

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
from .pipeline import get_global_frame_cap, map_detection_bbox_to_preview
from ..services.camera_repository import CameraRepository


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


def _camera_unavailable(error: CameraUnavailable) -> HTTPException:
    return HTTPException(status_code=503, detail=str(error))


def reconcile_camera_resolutions_on_startup(
    repository: CameraRepository,
) -> None:
    max_width, max_height = get_global_frame_cap()
    changes = repository.reconcile_global_frame_cap(max_width, max_height)
    for change in changes:
        logger.info(
            "Camera {} detection resolution recalculated from {}x{} to {}x{} "
            "for global frame cap {}x{}.",
            change["camera_id"],
            change["previous_width"],
            change["previous_height"],
            change["detection_width"],
            change["detection_height"],
            max_width,
            max_height,
        )


class VisionWorker:
    def __init__(
        self,
        camera_manager: CameraManager | None = None,
        detector: YOLODetector | None = None,
        inference_runner: CameraInferenceRunner | None = None,
    ) -> None:
        self.camera_manager = (
            camera_manager if camera_manager is not None else CameraManager()
        )
        self.config = get_detector_config()
        self.detector = detector or YOLODetector(
            model_path=self.config["model_path"],
            confidence=self.config["confidence"],
            iou=self.config["iou"],
            image_size=self.config["image_size"],
            device=self.config["device"],
            classes=self.config["classes"],
        )
        self.inference_runner = inference_runner or CameraInferenceRunner(
            self.detector,
            inference_fps=self.config["inference_fps"],
        )

    def start(self) -> None:
        self.detector.warmup()

    async def stop(self) -> None:
        self.inference_runner.stop_all()
        self.detector.unload()
        try:
            await asyncio.to_thread(self.camera_manager.stop_all)
        except CameraUnavailable as error:
            logger.error("Camera shutdown did not complete cleanly: %s", error)

    def camera_statuses(self) -> list[dict[str, Any]]:
        return self.camera_manager.statuses()

    def get_camera(self, camera_id: int) -> CameraCapture:
        return self.camera_manager.get(camera_id)

    async def configure_camera(
        self,
        camera_id: int,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        camera = self.get_camera(camera_id)
        status = await asyncio.to_thread(camera.configure, updates)
        if status["state"] == "running":
            await asyncio.to_thread(
                self.inference_runner.restart_camera,
                camera_id,
                camera,
            )
        else:
            await asyncio.to_thread(self.inference_runner.stop, camera_id)
        return status

    async def reconcile_cameras(
        self,
        cameras: list[dict[str, Any]],
        active_camera_ids: list[int],
        retry_failed: bool,
    ) -> list[dict[str, Any]]:
        previous_statuses = await asyncio.to_thread(self.camera_manager.statuses)
        previous_configurations = {
            status["camera_id"]: (status["source"], status["config"])
            for status in previous_statuses
        }
        statuses = await asyncio.to_thread(
            self.camera_manager.reconcile,
            cameras,
            active_camera_ids,
            retry_failed,
        )
        changed_camera_ids: set[int] = set()
        for status in statuses:
            previous = previous_configurations.get(status["camera_id"])
            if previous is not None and previous != (
                status["source"],
                status["config"],
            ):
                changed_camera_ids.add(status["camera_id"])

        active_ids: set[int] = {
            int(status["camera_id"])
            for status in statuses
            if status["state"] == "running"
        }
        active_cameras = {
            camera_id: self.camera_manager.get(camera_id)
            for camera_id in active_ids
        }
        self.inference_runner.sync(
            active_ids,
            active_cameras,
            restart_camera_ids=changed_camera_ids,
        )
        return statuses

    def inference_status(self) -> dict[str, Any]:
        status = self.inference_runner.status()
        for camera_id, result in list(status["camera_results"].items()):
            try:
                self._add_preview_bboxes(int(camera_id), result)
            except KeyError:
                del status["camera_results"][camera_id]
        return status

    def latest_detections(self, camera_id: int) -> dict[str, Any] | None:
        result = self.inference_runner.latest(camera_id)
        if result is None:
            return None

        self._add_preview_bboxes(camera_id, result)
        return result

    def _add_preview_bboxes(
        self,
        camera_id: int,
        result: dict[str, Any],
    ) -> None:
        camera = self.camera_manager.get(camera_id)
        camera_status = camera.status()
        preview_resolution = camera_status["preview_resolution"]
        for detection in result["detections"]:
            detection["preview_bbox"] = (
                map_detection_bbox_to_preview(
                    detection["bbox"],
                    preview_resolution["width"],
                    preview_resolution["height"],
                    camera_status["config"],
                )
                if preview_resolution is not None
                else None
            )


def get_vision_worker(request: Request) -> VisionWorker:
    worker = getattr(request.app.state, "vision_worker", None)
    if not isinstance(worker, VisionWorker):
        raise RuntimeError("VisionWorker is not configured on the FastAPI app.")
    return worker


def create_worker_router() -> APIRouter:
    router = APIRouter()

    @router.get(
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
    async def list_cameras(
        worker: Annotated[VisionWorker, Depends(get_vision_worker)],
    ) -> list[dict[str, Any]]:
        """Return the status and active configuration of every registered camera."""
        return worker.camera_statuses()

    @router.put(
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
        worker: Annotated[VisionWorker, Depends(get_vision_worker)],
        update: CameraReconcileRequest,
    ) -> list[dict[str, Any]]:
        """Register cameras, update their settings, and reconcile active streams."""
        try:
            return await worker.reconcile_cameras(
                [camera.model_dump() for camera in update.cameras],
                update.active_camera_ids,
                update.retry_failed,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get(
        "/api/v1/inference/status",
        response_model=InferenceStatusResponse,
        dependencies=[Depends(require_worker_token)],
        tags=["Inference"],
        summary="Get detector and per-camera inference status",
    )
    async def inference_status(
        worker: Annotated[VisionWorker, Depends(get_vision_worker)],
    ) -> dict[str, Any]:
        return worker.inference_status()

    @router.get(
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
        worker: Annotated[VisionWorker, Depends(get_vision_worker)],
        camera_id: int,
    ) -> dict[str, Any]:
        try:
            result = worker.latest_detections(camera_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error.args[0])) from error

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Inference is not running for this camera.",
            )
        return result

    @router.patch(
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
        worker: Annotated[VisionWorker, Depends(get_vision_worker)],
        camera_id: int,
        update: CameraConfigUpdate,
    ) -> dict[str, Any]:
        """Apply non-null or explicitly requested settings to one registered camera."""
        try:
            return await worker.configure_camera(
                camera_id,
                update.model_dump(exclude_unset=True),
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=error.args[0]) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except CameraUnavailable as error:
            raise _camera_unavailable(error) from error

    @router.get(
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
    async def stream_camera(
        worker: Annotated[VisionWorker, Depends(get_vision_worker)],
        camera_id: int,
    ) -> StreamingResponse:
        """Stream JPEG frames for an active camera as a multipart MJPEG response."""
        try:
            camera = worker.get_camera(camera_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=error.args[0]) from error
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

    return router


def create_worker_app(
    camera_manager: CameraManager | None = None,
    detector: YOLODetector | None = None,
    inference_runner: CameraInferenceRunner | None = None,
) -> FastAPI:
    worker = VisionWorker(camera_manager, detector, inference_runner)

    @asynccontextmanager
    async def lifespan(worker_app: FastAPI):
        try:
            worker.start()
            yield
        finally:
            await worker.stop()

    app = FastAPI(
        title="SmartLane Vision Worker",
        description="Internal control and camera-stream API for SmartLane vision processing.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.vision_worker = worker
    app.include_router(create_worker_router())
    return app


app = create_worker_app()


def main() -> None:
    token = _worker_token()
    if not token:
        raise RuntimeError("Set VISION_WORKER_TOKEN before starting the vision worker.")
    host = _validate_worker_host(os.getenv("VISION_WORKER_HOST"))
    port = _validate_worker_post(os.getenv("VISION_WORKER_PORT"))

    from .. import database

    connection = database.main()
    if connection is None:
        raise RuntimeError(
            "The vision worker cannot start without synchronizing camera settings "
            "with the database."
        )
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        try:
            repository = CameraRepository(connection, cursor, RLock())
            reconcile_camera_resolutions_on_startup(repository)
        finally:
            cursor.close()
    finally:
        connection.close()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
