import sys
from pathlib import Path
from threading import RLock
from typing import Annotated, Any

import psycopg2
from loguru import logger
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
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
from .schemas.user import UserProfile
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
