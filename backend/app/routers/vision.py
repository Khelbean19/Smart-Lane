"""Vision-camera assignment and preprocessing API routes.

This module is the coordination boundary between PostgreSQL, the FastAPI
process, and the separate vision worker.  Database operations describe the
configured camera state; reconciliation asks the worker to bring actual
camera processes in line with that state.

Admin-only writes:

* ``PUT /api/v1/vision/camera-slots`` replaces slot assignments.
* ``PUT /api/v1/vision/cameras/{camera_id}/settings`` changes preprocessing.

Every authenticated role can read ``GET /api/v1/vision/camera-slots``.  The
response includes private source and worker details only for administrators.

Example assignment payload::

    curl -X PUT http://localhost:8000/api/v1/vision/camera-slots \
      -H "Authorization: Bearer $ADMIN_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"assignments":[{"slot_id":1,"camera_id":3}]}'

Assignment and settings writes are persisted first, then reconciled with the
vision worker.  HTTP 202 means the worker was unavailable and reconciliation
is pending rather than that the database write failed.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..schemas.camera import ApplyCameraAssignments
from ..schemas.vision import (
    CameraAssignmentsResponse,
    CameraPreprocessingResponse,
    CameraPreprocessingSettingsUpdate,
    CameraSlotsResponse,
)
from ..services.camera_repository import (
    CameraDatabaseUnavailable,
    CameraRepository,
    InvalidCameraAssignment,
)
from ..services.vision_worker_client import (
    VisionWorkerClient,
    VisionWorkerConfigurationError,
    VisionWorkerError,
)
from ..vision.pipeline import get_global_frame_cap, validate_preprocessing_settings
from .auth import get_admin_user, get_current_user


router = APIRouter()


def get_vision_worker_client() -> VisionWorkerClient:
    """Build a vision-worker HTTP client or report configuration failure.

    FastAPI calls this factory as a dependency.  A configuration problem is
    converted to HTTP 503 so route handlers do not leak client internals.
    """
    try:
        return VisionWorkerClient()
    except VisionWorkerConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def get_camera_repository() -> CameraRepository:
    """Build a repository from the application's shared database resources.

    The repository serializes operations through ``database_lock`` and reads
    the connection/cursor created in ``backend.app.main``.
    """
    from .. import main as backend_main

    return CameraRepository(
        backend_main.connection,
        backend_main.cursor,
        backend_main.database_lock,
    )


async def reconcile_camera_state(
    repository: CameraRepository,
    client: VisionWorkerClient,
    cameras: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    *,
    retry_failed: bool,
) -> tuple[dict[int, str], str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Synchronize the database-assigned cameras with the vision worker.

    Args:
        repository: Database repository for status updates and later reads.
        client: HTTP client used to call the vision worker.
        cameras: Configured camera records.
        slots: Camera-to-slot assignments.  Assigned slot cameras are active.
        retry_failed: Whether the worker should retry previously failed cameras.

    Returns:
        A tuple of status values by camera id, optional worker error text,
        activation errors, and worker camera state records.

    Worker communication failure marks currently assigned cameras as FAULT and
    unassigned cameras as OFFLINE.  Successful communication derives final
    statuses from worker state and assigned slot membership.
    """
    active_camera_ids = [slot["camera_id"] for slot in slots if slot["camera_id"] is not None]
    try:
        worker_cameras = await client.request_json(
            "PUT",
            "/api/v1/cameras/reconcile",
            {
                "cameras": [
                    {
                        "camera_id": camera["camera_id"],
                        "source": str(camera["source"]),
                        "detection_width": camera.get("detection_width", 640),
                        "detection_height": camera.get("detection_height", 480),
                        "preprocessing_mode": camera.get("preprocessing_mode", "fit"),
                        "roi_x": camera.get("roi_x", 0.0),
                        "roi_y": camera.get("roi_y", 0.0),
                        "roi_width": camera.get("roi_width", 1.0),
                        "roi_height": camera.get("roi_height", 1.0),
                    }
                    for camera in cameras
                ],
                "active_camera_ids": active_camera_ids,
                "retry_failed": retry_failed,
            },
        )
    except VisionWorkerError as error:
        statuses = {
            camera["camera_id"]: (
                "FAULT" if camera["camera_id"] in active_camera_ids else "OFFLINE"
            )
            for camera in cameras
        }
        repository.update_statuses(statuses)
        return statuses, str(error), [], []

    states_by_id = {camera["camera_id"]: camera for camera in worker_cameras}
    statuses: dict[int, str] = {}
    activation_errors: list[dict[str, Any]] = []
    for camera in cameras:
        camera_id = camera["camera_id"]
        worker_camera = states_by_id.get(camera_id)
        if worker_camera is not None and worker_camera["state"] == "error":
            statuses[camera_id] = "FAULT"
            if camera_id in active_camera_ids:
                activation_errors.append({
                    "camera_id": camera_id,
                    "error": worker_camera.get("error") or "Camera failed to activate.",
                })
        elif camera_id in active_camera_ids:
            if worker_camera is not None and worker_camera["state"] == "running":
                statuses[camera_id] = "ONLINE"
            else:
                statuses[camera_id] = "FAULT"
                activation_errors.append({
                    "camera_id": camera_id,
                    "error": "Camera was not reported as running by the worker.",
                })
        else:
            statuses[camera_id] = "OFFLINE"
    repository.update_statuses(statuses)
    return statuses, None, activation_errors, worker_cameras


def build_camera_slots_response(
    cameras: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    role: str,
    worker_error: str | None,
    activation_errors: list[dict[str, Any]],
    worker_cameras: list[dict[str, Any]],
    global_max_resolution: dict[str, int],
) -> dict[str, Any]:
    """Convert repository and worker state into a role-aware API payload.

    Non-administrators receive slot status and identity fields, but source
    paths and worker details are omitted.  Administrators receive full
    configuration and diagnostics.
    """
    is_admin = role == "ADMINISTRATOR"
    worker_cameras_by_id = {camera["camera_id"]: camera for camera in worker_cameras}
    response_slots = []
    for slot in slots:
        response_slot = {
            "slot_id": slot["slot_id"],
            "region": slot["region"],
            "camera_id": slot["camera_id"],
            "camera_name": slot["camera_name"],
            "status": slot["status"] or "OFFLINE",
        }
        slot = {key: value for key, value in slot.items() if value is not None}
        if is_admin:
            response_slot["source"] = slot.get("source")
            worker_camera = worker_cameras_by_id.get(slot.get("camera_id"))
            response_slot["source_resolution"] = worker_camera.get("source_resolution") if worker_camera else None
            response_slot["preview_resolution"] = worker_camera.get("preview_resolution") if worker_camera else None
            response_slot["max_frame_resolution"] = global_max_resolution
            response_slot["detection_resolution"] = {
                "width": slot.get("detection_width", 640),
                "height": slot.get("detection_height", 480),
            }
            response_slot["preprocessing_mode"] = slot.get("preprocessing_mode", "fit")
            response_slot["roi"] = {
                "x": slot.get("roi_x", 0.0),
                "y": slot.get("roi_y", 0.0),
                "width": slot.get("roi_width", 1.0),
                "height": slot.get("roi_height", 1.0),
            }
        response_slots.append(response_slot)

    response_cameras = []
    if is_admin:
        for camera in cameras:
            worker_camera = worker_cameras_by_id.get(camera["camera_id"], {})
            response_cameras.append({
                "camera_id": camera["camera_id"],
                "camera_name": camera["camera_name"],
                "region": camera["region"],
                "status": camera["status"],
                "source": camera["source"],
                "detection_width": camera.get("detection_width", 640),
                "detection_height": camera.get("detection_height", 480),
                "preprocessing_mode": camera.get("preprocessing_mode", "fit"),
                "roi_x": camera.get("roi_x", 0.0),
                "roi_y": camera.get("roi_y", 0.0),
                "roi_width": camera.get("roi_width", 1.0),
                "roi_height": camera.get("roi_height", 1.0),
                "max_frame_resolution": global_max_resolution,
                "source_resolution": worker_camera.get("source_resolution"),
                "preview_resolution": worker_camera.get("preview_resolution"),
            })

    return {
        "slots": response_slots,
        "cameras": response_cameras,
        "global_max_resolution": global_max_resolution if is_admin else None,
        "worker_available": worker_error is None,
        "worker_error": worker_error if is_admin else None,
        "activation_errors": activation_errors if is_admin else [],
    }


@router.get(
    "/api/v1/vision/camera-slots",
    response_model=CameraSlotsResponse,
    response_model_exclude_unset=True,
    tags=["Vision"],
)
async def get_vision_camera_slots(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    repository: Annotated[CameraRepository, Depends(get_camera_repository)],
    client: Annotated[VisionWorkerClient, Depends(get_vision_worker_client)],
) -> dict[str, Any]:
    """Read current camera assignments and worker status.

    The operation reconciles state but does not retry known failed cameras.
    A non-administrator receives a reduced payload that excludes source and
    activation diagnostics.

    Example:
        GET /api/v1/vision/camera-slots
        Authorization: Bearer <token>
    """
    try:
        cameras, slots = repository.read_state()
        _, worker_error, activation_errors, worker_cameras = await reconcile_camera_state(
            repository, client, cameras, slots, retry_failed=False
        )
        cameras, slots = repository.read_state()
        global_width, global_height = get_global_frame_cap()
    except CameraDatabaseUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return build_camera_slots_response(
        cameras, slots, current_user["role"], worker_error, activation_errors,
        worker_cameras, {"width": global_width, "height": global_height},
    )


@router.put(
    "/api/v1/vision/camera-slots",
    response_model=CameraAssignmentsResponse,
    response_model_exclude_unset=True,
    tags=["Vision"],
)
async def update_vision_camera_slots(
    update: ApplyCameraAssignments,
    current_user: Annotated[dict[str, Any], Depends(get_admin_user)],
    repository: Annotated[CameraRepository, Depends(get_camera_repository)],
    client: Annotated[VisionWorkerClient, Depends(get_vision_worker_client)],
) -> JSONResponse:
    """Replace camera-slot assignments and reconcile the vision worker.

    Example payload:
        ``{"assignments":[{"slot_id":1,"camera_id":3}]}``

    The database mutation is accepted before worker reconciliation.  HTTP 202
    is used with ``pending_reconciliation: true`` when the worker is down.
    """
    assignments = [assignment.model_dump() for assignment in update.assignments]
    try:
        repository.replace_assignments(assignments, current_user["user_id"])
        cameras, slots = repository.read_state()
    except CameraDatabaseUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except InvalidCameraAssignment as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        _, worker_error, activation_errors, worker_cameras = await reconcile_camera_state(
            repository, client, cameras, slots, retry_failed=True
        )
        cameras, slots = repository.read_state()
        global_width, global_height = get_global_frame_cap()
    except CameraDatabaseUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    payload = build_camera_slots_response(
        cameras, slots, current_user["role"], worker_error, activation_errors,
        worker_cameras, {"width": global_width, "height": global_height},
    )
    payload["assignments_saved"] = True
    payload["pending_reconciliation"] = worker_error is not None
    return JSONResponse(
        status_code=202 if worker_error is not None else 200,
        content=payload,
    )


@router.put(
    "/api/v1/vision/cameras/{camera_id}/settings",
    response_model=CameraPreprocessingResponse,
    response_model_exclude_unset=True,
    tags=["Vision"],
)
async def update_camera_preprocessing_settings(
    camera_id: int,
    update: CameraPreprocessingSettingsUpdate,
    current_user: Annotated[dict[str, Any], Depends(get_admin_user)],
    repository: Annotated[CameraRepository, Depends(get_camera_repository)],
    client: Annotated[VisionWorkerClient, Depends(get_vision_worker_client)],
) -> JSONResponse:
    """Validate and save preprocessing settings for one camera.

    Supported values and bounds are determined by
    ``validate_preprocessing_settings`` and the global frame cap.  Saved values
    are sent to the vision worker through reconciliation; if reconciliation
    fails, HTTP 202 reports that the settings are saved but still pending.
    """
    settings = update.model_dump()
    try:
        max_width, max_height = get_global_frame_cap()
        validate_preprocessing_settings(settings, max_width, max_height)
        repository.update_camera_preprocessing(camera_id, settings)
        cameras, slots = repository.read_state()
        _, worker_error, activation_errors, worker_cameras = await reconcile_camera_state(
            repository, client, cameras, slots, retry_failed=True
        )
        cameras, slots = repository.read_state()
    except CameraDatabaseUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    payload = build_camera_slots_response(
        cameras, slots, current_user["role"], worker_error, activation_errors,
        worker_cameras, {"width": max_width, "height": max_height},
    )
    camera = next(
        (item for item in payload["cameras"] if item["camera_id"] == camera_id),
        None,
    )
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera ID {camera_id}.")
    payload["camera"] = camera
    payload["settings_saved"] = True
    payload["pending_reconciliation"] = worker_error is not None
    return JSONResponse(
        status_code=202 if worker_error is not None else 200,
        content=payload,
    )
