import sys
from pathlib import Path
from threading import RLock
from typing import Annotated, Any

import psycopg2
from loguru import logger
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from . import database
from .core.security import (
    ACCESS_TOKEN_TTL_SECONDS,
    REMEMBER_ACCESS_TOKEN_TTL_SECONDS,
    create_access_token,
    get_access_token_subject,
    hash_password,
    password_hash_needs_upgrade,
    verify_password,
)
from .models import UserIn, UserAuth
from .schemas.camera import ApplyCameraAssignments
from .schemas.vision import CameraPreprocessingSettingsUpdate
from .schemas.user import UserProfile
from .services.camera_repository import (
    CameraDatabaseUnavailable,
    CameraRepository,
    InvalidCameraAssignment,
    InvalidCameraSettings,
)
from .vision.pipeline import get_global_frame_cap, validate_preprocessing_settings
from .services.vision_worker_client import (
    VisionWorkerClient,
    VisionWorkerConfigurationError,
    VisionWorkerError,
)
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor


load_dotenv()
logger.remove()
logger.add(sys.stdout, level="DEBUG")

app = FastAPI()  # initialize backend
connection = database.main()  # initialize database connection
FRONTEND_PUBLIC_DIR = Path(__file__).resolve().parents[2] / "frontend" / "public"
# OAuth2 clients obtain bearer tokens from this form-encoded endpoint.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
cursor = None
database_lock = RLock()

if connection:
    cursor = connection.cursor(cursor_factory=RealDictCursor)

# directory is relative to project root
app.frontend("/", directory=FRONTEND_PUBLIC_DIR, fallback="404.html")


@app.get("/login", include_in_schema=False)
async def login_page() -> FileResponse:
    """Serve the login page."""
    return FileResponse(FRONTEND_PUBLIC_DIR / "index.html")


@app.get("/dashboard", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    """Serve the dashboard page."""
    return FileResponse(FRONTEND_PUBLIC_DIR / "dashboard.html")


@app.get("/live", include_in_schema=False)
async def live_page() -> FileResponse:
    """Serve the live-control page."""
    return FileResponse(FRONTEND_PUBLIC_DIR / "live-control.html")


@app.get("/reports", include_in_schema=False)
async def reports_page() -> FileResponse:
    """Serve the reports page."""
    return FileResponse(FRONTEND_PUBLIC_DIR / "reports.html")


@app.get("/users", include_in_schema=False)
async def users_page() -> FileResponse:
    """Serve the user-management page."""
    return FileResponse(FRONTEND_PUBLIC_DIR / "users.html")


@app.get("/settings", include_in_schema=False)
async def settings_page() -> FileResponse:
    """Serve the system-settings page."""
    return FileResponse(FRONTEND_PUBLIC_DIR / "settings.html")



# Server status page
@app.get("/status", include_in_schema=False)
async def status_page() -> FileResponse:
    """Serve the styled server-status page."""
    return FileResponse(FRONTEND_PUBLIC_DIR / "status.html")


def authenticate_user(
    username: str,
    password: str,
    remember_me: bool = False,
) -> tuple[dict[str, Any], str, int]:
    """Verify credentials for an active account and issue a short-lived token.

    Username may be either the account username or email address. Legacy
    SHA-256 password digests are upgraded to Argon2 only after verification.
    Credentials are intentionally reported with the same 401 response whether
    the account is missing, disabled, or the password is wrong.

    Raises:
        HTTPException: 401 for invalid credentials; 503 if the database is
            unavailable; 500 if token signing cannot be configured.
    """
    if not connection or not cursor:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    with database_lock:
        cursor.execute(
            "SELECT user_id, password_hash, role, status FROM users "
            "WHERE username=%s OR email=%s LIMIT 1",
            (username, username),
        )
        user = cursor.fetchone()
    invalid_credentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Account credentials are incorrect or the requested account is inactive.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not user:
        raise invalid_credentials

    if not verify_password(user["password_hash"], password) or user["status"] != "ACTIVE":
        raise invalid_credentials

    if password_hash_needs_upgrade(user["password_hash"]):
        upgraded_hash = hash_password(password)
        with database_lock:
            cursor.execute(
                "UPDATE users SET password_hash=%s WHERE user_id=%s",
                (upgraded_hash, user["user_id"]),
            )
            connection.commit()

    expires_in = (
        REMEMBER_ACCESS_TOKEN_TTL_SECONDS if remember_me else ACCESS_TOKEN_TTL_SECONDS
    )
    try:
        access_token = create_access_token(str(user["user_id"]), remember_me=remember_me)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return user, access_token, expires_in


# ──────────────────────────────────────────────────────────────────────────────────────────────── #
#   POST Authenticate user (called by frontend)
# ──────────────────────────────────────────────────────────────────────────────────────────────── #
@app.post("/api/v1/auth/", responses={
    401: {"description": "Incorrect Credentials"},
    500: {}
})
async def auth_user(credentials: Annotated[UserAuth, ""]):
    """Authenticate using the existing JSON login contract and return a token.

    This endpoint does not require an existing bearer token. Clients should
    retain ``access_token`` securely and send it as a Bearer authorization
    header when requesting protected user-management endpoints.
    """
    user, access_token, expires_in = authenticate_user(
        credentials.username,
        credentials.password,
        remember_me=credentials.remember,
    )
    return {
        "message": "Account credentials are correct.",
        "user_id": user["user_id"],
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_in,
    }


# ──────────────────────────────────────────────────────────────────────────────────────────────── #
#   POST Issue access token (for authentication)
# ──────────────────────────────────────────────────────────────────────────────────────────────── #
@app.post("/api/v1/auth/token")
async def issue_access_token(
    credentials: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    """OAuth2 password-flow token endpoint using form-encoded credentials.

    This is the ``tokenUrl`` advertised by ``oauth2_scheme``. The response is
    compatible with OAuth2 clients; account validation and token lifetime are
    enforced by ``authenticate_user``.
    """
    _, access_token, expires_in = authenticate_user(
        credentials.username,
        credentials.password,
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_in,
    }


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> dict[str, Any]:
    """Resolve the bearer token to its active database user.

    The token is used only to identify the user. Role and account status are
    reloaded on every request so disabling an account or changing its role
    takes effect immediately, rather than waiting for token expiry. This
    dependency authenticates any active role; use ``get_admin_user`` when an
    operation also requires administrator authorization.

    Raises:
        HTTPException: 401 for invalid/expired tokens or unavailable users,
            503 if the database is unavailable, or 500 if token configuration
            is missing.
    """
    try:
        user_id = get_access_token_subject(token)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not connection or not cursor:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    with database_lock:
        cursor.execute(
            "SELECT user_id, full_name, username, email, phone, role, status, "
            "last_login, created_at, updated_at "
            "FROM users WHERE user_id=%s",
            (user_id,),
        )
        user = cursor.fetchone()
    if not user or user["status"] != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is unavailable.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


@app.get("/api/v1/me", response_model=UserProfile)
async def get_me(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    """Return the authenticated user's safe profile fields.

    OAuth2 bearer authentication is required. The response model deliberately
    excludes the password hash and other credential material.
    """
    return current_user


def get_admin_user(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    """Require that the authenticated active user has the administrator role.

    Raises:
        HTTPException: 403 if the user is authenticated but not an
            administrator. Authentication failures are raised by
            ``get_current_user``.
    """
    if current_user["role"] != "ADMINISTRATOR":
        raise HTTPException(status_code=403, detail="Administrator role required.")

    return current_user


# ──────────────────────────────────────────────────────────────────────────────────────────────── #
#   POST Create User
# ──────────────────────────────────────────────────────────────────────────────────────────────── #
@app.post("/api/v1/users/", responses={
    400: {"description": "Account Already Exists"},
    500: {}
})
async def create_user(
    user: Annotated[UserIn, ""],
    _: Annotated[dict[str, Any], Depends(get_admin_user)],
):
    """Create a user account; requires an active administrator bearer token.

    Passwords are stored as Argon2 hashes. The request's role and status are
    persisted as provided, so this privileged endpoint must never be exposed
    without its administrator dependency.
    """
    if connection and cursor:
        password_hash = hash_password(user.password)
        try:
            with database_lock:
                try:
                    cursor.execute("INSERT INTO users " \
                                   "(full_name, username, email, phone, password_hash, role, status) " \
                                   "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                                   (user.full_name,
                                    user.username,
                                    "null" if user.email is None else user.email,
                                    user.phone,
                                    password_hash,
                                    user.role,
                                    user.status))
                    connection.commit()
                except psycopg2.errors.UniqueViolation:
                    connection.rollback()
                    raise
            return {
                "message": "User created successfully"
            }
        except psycopg2.errors.UniqueViolation:
            raise HTTPException(400, "An account with the same username, email, or phone number already exists.")
    else:
        raise HTTPException(500)


# ──────────────────────────────────────────────────────────────────────────────────────────────── #
#   GET Users
# ──────────────────────────────────────────────────────────────────────────────────────────────── #
@app.get("/api/v1/users/")  # TODO (OPTIONAL): Add query parameters for pagination
async def get_users(_: Annotated[dict[str, Any], Depends(get_admin_user)]):
    """List user account metadata; requires an active administrator token.

    Password hashes are excluded from the selected columns and response. The
    bearer token itself is never returned in the response.
    """
    if connection and cursor:  # db connection is open
        with database_lock:
            cursor.execute("SELECT " \
                           "user_id, full_name, username, email, phone, role, status, last_login, created_at, updated_at " \
                           "FROM users LIMIT 10")
            data = cursor.fetchall()
        return {"data": data}
    else:
        raise HTTPException(status_code=503, detail="Database unavailable.")


def get_vision_worker_client() -> VisionWorkerClient:
    try:
        return VisionWorkerClient()
    except VisionWorkerConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def get_camera_repository() -> CameraRepository:
    return CameraRepository(connection, cursor, database_lock)


async def reconcile_camera_state(
    repository: CameraRepository,
    client: VisionWorkerClient,
    cameras: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    *,
    retry_failed: bool,
) -> tuple[dict[int, str], str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    active_camera_ids = [
        slot["camera_id"] for slot in slots if slot["camera_id"] is not None
    ]
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
                        "preprocessing_mode": camera.get(
                            "preprocessing_mode",
                            "fit",
                        ),
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

    states_by_id = {
        camera["camera_id"]: camera
        for camera in worker_cameras
    }
    statuses: dict[int, str] = {}
    activation_errors: list[dict[str, Any]] = []
    for camera in cameras:
        camera_id = camera["camera_id"]
        worker_camera = states_by_id.get(camera_id)
        if worker_camera is not None and worker_camera["state"] == "error":
            statuses[camera_id] = "FAULT"
            if camera_id in active_camera_ids:
                activation_errors.append(
                    {
                        "camera_id": camera_id,
                        "error": worker_camera.get("error")
                        or "Camera failed to activate.",
                    }
                )
        elif camera_id in active_camera_ids:
            if worker_camera is not None and worker_camera["state"] == "running":
                statuses[camera_id] = "ONLINE"
            else:
                statuses[camera_id] = "FAULT"
                activation_errors.append(
                    {
                        "camera_id": camera_id,
                        "error": "Camera was not reported as running by the worker.",
                    }
                )
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
    is_admin = role == "ADMINISTRATOR"
    worker_cameras_by_id = {
        camera["camera_id"]: camera for camera in worker_cameras
    }
    response_slots = []
    for slot in slots:
        response_slot = {
            "slot_id": slot["slot_id"],
            "region": slot["region"],
            "camera_id": slot["camera_id"],
            "camera_name": slot["camera_name"],
            "status": slot["status"] or "OFFLINE",
        }
        if is_admin:
            response_slot["source"] = slot["source"]
            worker_camera = worker_cameras_by_id.get(slot["camera_id"])
            response_slot["source_resolution"] = (
                worker_camera.get("source_resolution") if worker_camera else None
            )
            response_slot["preview_resolution"] = (
                worker_camera.get("preview_resolution") if worker_camera else None
            )
            response_slot["max_frame_resolution"] = global_max_resolution
            response_slot["detection_resolution"] = {
                "width": slot.get("detection_width", 640),
                "height": slot.get("detection_height", 480),
            }
            response_slot["preprocessing_mode"] = slot.get(
                "preprocessing_mode",
                "fit",
            )
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
            response_cameras.append(
                {
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
                    "source_resolution": (
                        worker_cameras_by_id.get(camera["camera_id"], {})
                        .get("source_resolution")
                    ),
                    "preview_resolution": (
                        worker_cameras_by_id.get(camera["camera_id"], {})
                        .get("preview_resolution")
                    ),
                }
            )

    return {
        "slots": response_slots,
        "cameras": response_cameras,
        "global_max_resolution": global_max_resolution if is_admin else None,
        "worker_available": worker_error is None,
        "worker_error": worker_error if is_admin else None,
        "activation_errors": activation_errors if is_admin else [],
    }


@app.get("/api/v1/vision/camera-slots")
async def get_vision_camera_slots(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    repository: Annotated[CameraRepository, Depends(get_camera_repository)],
    client: Annotated[VisionWorkerClient, Depends(get_vision_worker_client)],
) -> dict[str, Any]:
    try:
        cameras, slots = repository.read_state()
        _, worker_error, activation_errors, worker_cameras = await reconcile_camera_state(
            repository,
            client,
            cameras,
            slots,
            retry_failed=False,
        )
        cameras, slots = repository.read_state()
        global_width, global_height = get_global_frame_cap()
    except CameraDatabaseUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return build_camera_slots_response(
        cameras,
        slots,
        current_user["role"],
        worker_error,
        activation_errors,
        worker_cameras,
        {"width": global_width, "height": global_height},
    )


@app.put("/api/v1/vision/camera-slots")
async def update_vision_camera_slots(
    update: ApplyCameraAssignments,
    current_user: Annotated[dict[str, Any], Depends(get_admin_user)],
    repository: Annotated[CameraRepository, Depends(get_camera_repository)],
    client: Annotated[VisionWorkerClient, Depends(get_vision_worker_client)],
) -> JSONResponse:
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
            repository,
            client,
            cameras,
            slots,
            retry_failed=True,
        )
        cameras, slots = repository.read_state()
        global_width, global_height = get_global_frame_cap()
    except CameraDatabaseUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    payload = build_camera_slots_response(
        cameras,
        slots,
        current_user["role"],
        worker_error,
        activation_errors,
        worker_cameras,
        {"width": global_width, "height": global_height},
    )
    payload["assignments_saved"] = True
    payload["pending_reconciliation"] = worker_error is not None
    return JSONResponse(
        status_code=202 if worker_error is not None else 200,
        content=payload,
    )


@app.put("/api/v1/vision/cameras/{camera_id}/settings")
async def update_camera_preprocessing_settings(
    camera_id: int,
    update: CameraPreprocessingSettingsUpdate,
    current_user: Annotated[dict[str, Any], Depends(get_admin_user)],
    repository: Annotated[CameraRepository, Depends(get_camera_repository)],
    client: Annotated[VisionWorkerClient, Depends(get_vision_worker_client)],
) -> JSONResponse:
    settings = update.model_dump()
    try:
        max_width, max_height = get_global_frame_cap()
        validate_preprocessing_settings(settings, max_width, max_height)
        repository.update_camera_preprocessing(camera_id, settings)
        cameras, slots = repository.read_state()
        _, worker_error, activation_errors, worker_cameras = await reconcile_camera_state(
            repository,
            client,
            cameras,
            slots,
            retry_failed=True,
        )
        cameras, slots = repository.read_state()
    except CameraDatabaseUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    payload = build_camera_slots_response(
        cameras,
        slots,
        current_user["role"],
        worker_error,
        activation_errors,
        worker_cameras,
        {"width": max_width, "height": max_height},
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
