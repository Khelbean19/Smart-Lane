"""Authentication and authorization dependencies.

The module exposes two login representations:

* ``POST /api/v1/auth/`` accepts SmartLane's JSON ``UserAuth`` model and
  returns both the bearer token and user id.
* ``POST /api/v1/auth/token`` implements the OAuth2 password form contract,
  allowing Swagger UI and OAuth2-compatible clients to obtain a token.

The returned JWT is consumed by ``get_current_user``.  That dependency loads
the live database record for every protected request, so deleted, inactive, or
disabled accounts stop working even if an old token has not expired.

Example JSON login::

    curl -X POST http://localhost:8000/api/v1/auth/ \
      -H "Content-Type: application/json" \
      -d '{"username":"operator","password":"secret","remember":false}'

Use the response token as ``Authorization: Bearer <access_token>``.  Protected
dependencies raise 401 for invalid credentials and 403 for a missing role.
"""

from typing import Annotated, Any
from pathlib import Path

from fastapi import Depends, APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from ..core.security import (
    ACCESS_TOKEN_TTL_SECONDS,
    REMEMBER_ACCESS_TOKEN_TTL_SECONDS,
    create_access_token,
    get_access_token_subject,
    hash_password,
    password_hash_needs_upgrade,
    verify_password,
)
from ..schemas.auth import AuthResponse, TokenResponse
from ..models import UserAuth


router = APIRouter()
# Page navigation target separate from static frontend mounting.
FRONTEND_PUBLIC_DIR = Path(__file__).resolve().parents[3] / "frontend" / "public"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


@router.get("/login", include_in_schema=False)
async def login_page() -> FileResponse:
    """Serve the public login page.

    This route is excluded from OpenAPI because it returns HTML rather than a
    JSON API resource.
    """
    return FileResponse(FRONTEND_PUBLIC_DIR / "index.html")


def authenticate_user(
    username: str,
    password: str,
    remember_me: bool = False,
) -> tuple[dict[str, Any], str, int]:
    """Verify credentials and issue a bearer token.

    Args:
        username: Account username or email address.
        password: Plain-text password supplied by the caller.
        remember_me: If true, select the longer configured token lifetime.

    Returns:
        A tuple containing the database user row, JWT string, and lifetime in
        seconds.

    Raises:
        HTTPException: 503 when PostgreSQL is unavailable, 401 for invalid or
            inactive credentials, or 500 for missing signing configuration.

    Password hashes are upgraded transparently after a successful login when
    the current security configuration indicates that a stronger encoding is
    available.
    """
    from .. import main as backend_main

    if not backend_main.connection or not backend_main.cursor:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    with backend_main.database_lock:
        backend_main.cursor.execute(
            "SELECT user_id, password_hash, role, status FROM users "
            "WHERE username=%s OR email=%s LIMIT 1",
            (username, username),
        )
        user = backend_main.cursor.fetchone()

    invalid_credentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Account credentials are incorrect or the requested account is inactive.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not user or not verify_password(user["password_hash"], password) or user["status"] != "ACTIVE":
        raise invalid_credentials

    if password_hash_needs_upgrade(user["password_hash"]):
        upgraded_hash = hash_password(password)
        with backend_main.database_lock:
            backend_main.cursor.execute(
                "UPDATE users SET password_hash=%s WHERE user_id=%s",
                (upgraded_hash, user["user_id"]),
            )
            backend_main.connection.commit()

    expires_in = REMEMBER_ACCESS_TOKEN_TTL_SECONDS if remember_me else ACCESS_TOKEN_TTL_SECONDS
    try:
        access_token = create_access_token(str(user["user_id"]), remember_me=remember_me)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return user, access_token, expires_in


@router.post(
    "/api/v1/auth/",
    response_model=AuthResponse,
    tags=["Authentication"],
    summary="Authenticate a user",
)
async def auth_user(credentials: UserAuth) -> dict[str, Any]:
    """Authenticate with the SmartLane JSON login model.

    Example:
        POST /api/v1/auth/ with ``{"username":"admin","password":"...","remember":false}``
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


@router.post(
    "/api/v1/auth/token",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Issue an OAuth2 bearer token",
)
async def issue_access_token(
    credentials: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> dict[str, Any]:
    """Issue a token using an OAuth2-compatible password form.

    Swagger UI's OAuth2 authorization control submits the ``username`` and
    ``password`` fields to this endpoint.  It does not support ``remember_me``;
    use ``POST /api/v1/auth/`` when that option is required.
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
    """Resolve a bearer token to an active database user.

    This FastAPI dependency is used by protected JSON routes.  It validates the
    JWT subject and then reloads the user row to enforce current role/status
    state.
    """
    from .. import main as backend_main

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
    if not backend_main.connection or not backend_main.cursor:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    with backend_main.database_lock:
        backend_main.cursor.execute(
            "SELECT user_id, full_name, username, email, phone, role, status, "
            "last_login, created_at, updated_at "
            "FROM users WHERE user_id=%s",
            (user_id,),
        )
        user = backend_main.cursor.fetchone()

    if not user or user["status"] != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is unavailable.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_admin_user(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    """Require the ``ADMINISTRATOR`` role.

    Use this dependency for administrative mutations such as creating users,
    changing camera assignments, or modifying system settings.
    """
    if current_user["role"] != "ADMINISTRATOR":
        raise HTTPException(status_code=403, detail="Administrator role required.")
    return current_user