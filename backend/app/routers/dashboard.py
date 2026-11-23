"""HTML page routes for the dashboard and traffic status views.

These routes are browser-navigation endpoints rather than JSON API endpoints.
They serve static HTML while authentication state is enforced by the frontend
and protected API dependencies.

Example:
    GET /dashboard
    GET /status
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


router = APIRouter()
# Resolved once so FileResponse uses the frontend's absolute public directory.
FRONTEND_PUBLIC_DIR = Path(__file__).resolve().parents[3] / "frontend" / "public"


@router.get("/dashboard", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    """Serve the main operations dashboard HTML.

    Example response:
        GET /dashboard -> 200 text/html (dashboard.html)
    """
    return FileResponse(FRONTEND_PUBLIC_DIR / "dashboard.html")


@router.get("/status", include_in_schema=False)
async def status_page() -> FileResponse:
    """Serve the read-only traffic status page HTML.

    Example response:
        GET /status -> 200 text/html (status.html)
    """
    return FileResponse(FRONTEND_PUBLIC_DIR / "status.html")