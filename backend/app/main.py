"""Application entry point for the SmartLane backend.

This module creates the FastAPI application, opens the PostgreSQL connection,
registers API and page routers, and serves the static frontend.  It is loaded
by ASGI servers with ``backend.app.main:app`` (for example, ``fastapi dev
backend/app/main.py``).

Startup and request flow
------------------------
1. Environment variables are loaded from ``.env``.
2. Loguru is configured to write debug and higher-severity logs to stdout.
3. ``database.main()`` establishes the shared database connection.
4. Routers from :mod:`backend.app.routers` are attached to ``app``.
5. The built frontend is mounted last, allowing explicit API and page routes
   to take precedence over static files.

Example
-------
Run the development server from the repository root::

    fastapi dev backend/app/main.py

The interactive OpenAPI description is then available at
``http://127.0.0.1:8000/docs``.
"""

import sys
from pathlib import Path
from threading import RLock
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger
from psycopg2.extras import RealDictCursor

from . import database
from .routers.auth import (
    authenticate_user,
    get_admin_user,
    get_current_user,
    router as auth_router,
)
from .routers.dashboard import router as dashboard_router
from .routers.live_control import router as live_control_router
from .routers.reports import router as reports_router
from .routers.settings import router as settings_router
from .routers.users import router as users_router
from .routers.vision import (
    build_camera_slots_response,
    get_camera_repository,
    get_vision_worker_client,
    reconcile_camera_state,
    router as vision_router,
)
from .routers.vision_stream import router as vision_stream_router


# Loading environment variables before creating services ensures that
# configuration read by database and security helpers sees .env values.
load_dotenv()
logger.remove()
logger.add(sys.stdout, level="DEBUG")

app = FastAPI(
    title="SmartLane Backend API",
    description="Administrative, authentication, and vision-control API for SmartLane.",
    version="1.0.0",
)

# Shared resources are intentionally created once in this module and reused by
# FastAPI dependencies and authentication helpers.  A lock protects cursor
# operations because psycopg2 connections and cursors are not thread-safe.
connection = database.main()
FRONTEND_PUBLIC_DIR: Path = Path(__file__).resolve().parents[2] / "frontend" / "public"
cursor: Any | None = None
database_lock = RLock()

if connection:
    cursor = connection.cursor(cursor_factory=RealDictCursor)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(vision_router)
app.include_router(vision_stream_router)
app.include_router(dashboard_router)
app.include_router(live_control_router)
app.include_router(reports_router)
app.include_router(settings_router)

app.mount("/", StaticFiles(directory=FRONTEND_PUBLIC_DIR, html=True), name="frontend")
