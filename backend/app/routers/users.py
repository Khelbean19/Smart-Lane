"""User profile and administration routes.

The module separates identity data from mutating operations:

* ``GET /api/v1/me`` returns the authenticated caller.
* ``POST /api/v1/users/`` creates an account and requires administrator role.
* ``GET /api/v1/users/`` lists the first ten accounts and requires
  administrator role.

Example administrator requests::

    curl -H "Authorization: Bearer $TOKEN" \
      http://localhost:8000/api/v1/me

    curl -X POST http://localhost:8000/api/v1/users/ \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"full_name":"Lane Operator","username":"operator","email":"operator@example.com","phone":"...","password":"...","role":"OPERATOR","status":"ACTIVE"}'
"""

from pathlib import Path

from typing import Annotated, Any

import psycopg2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..models import UserIn
from ..schemas.auth import MessageResponse, UserListResponse
from ..schemas.user import UserProfile
from ..core.security import hash_password
from .auth import get_admin_user, get_current_user


router = APIRouter()
# Shared frontend location used for the hidden navigation page.
FRONTEND_PUBLIC_DIR = Path(__file__).resolve().parents[3] / "frontend" / "public"


@router.get("/users", include_in_schema=False)
async def users_page() -> FileResponse:
    """Serve the user-administration HTML page."""
    return FileResponse(FRONTEND_PUBLIC_DIR / "users.html")


@router.get(
    "/api/v1/me",
    response_model=UserProfile,
    tags=["Users"],
    summary="Get the authenticated user",
)
async def get_me(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    """Return the profile for the bearer-authenticated caller.

    The dependency reloads the user row for each request; no separate
    authorization role is required.
    """
    return current_user


@router.post(
    "/api/v1/users/",
    response_model=MessageResponse,
    tags=["Users"],
    summary="Create a user",
)
async def create_user(
    user: UserIn,
    _: Annotated[dict[str, Any], Depends(get_admin_user)],
) -> dict[str, str]:
    """Create a user account.

    Args:
        user: Validated account fields from ``UserIn``.
        _: An administrator identity; it is only used as an authorization gate.

    Raises:
        HTTPException: 400 for duplicate account fields, 500 for a missing
            database connection, or 403 from ``get_admin_user``.

    The password is hashed before insertion; the submitted plain-text password
    is never stored or returned.
    """
    from .. import main as backend_main

    if not backend_main.connection or not backend_main.cursor:
        raise HTTPException(status_code=500)

    password_hash = hash_password(user.password)
    try:
        with backend_main.database_lock:
            try:
                backend_main.cursor.execute(
                    "INSERT INTO users "
                    "(full_name, username, email, phone, password_hash, role, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        user.full_name,
                        user.username,
                        "null" if user.email is None else user.email,
                        user.phone,
                        password_hash,
                        user.role,
                        user.status,
                    ),
                )
                backend_main.connection.commit()
            except psycopg2.errors.UniqueViolation:
                backend_main.connection.rollback()
                raise
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(
            400,
            "An account with the same username, email, or phone number already exists.",
        )
    return {"message": "User created successfully"}


@router.get(
    "/api/v1/users/",
    response_model=UserListResponse,
    tags=["Users"],
    summary="List users",
)
async def get_users(
    _: Annotated[dict[str, Any], Depends(get_admin_user)],
) -> dict[str, Any]:
    """List up to ten user records for administrator clients.

    Example:
        GET /api/v1/users/
        Authorization: Bearer <administrator-token>

    The response shape is ``{"data": [...]}``.  Database availability failure
    is reported as HTTP 503 instead of an empty successful list.
    """
    from .. import main as backend_main

    if backend_main.connection and backend_main.cursor:
        with backend_main.database_lock:
            backend_main.cursor.execute(
                "SELECT user_id, full_name, username, email, phone, role, status, "
                "last_login, created_at, updated_at FROM users LIMIT 10"
            )
            data = backend_main.cursor.fetchall()
        return {"data": data}
    raise HTTPException(status_code=503, detail="Database unavailable.")