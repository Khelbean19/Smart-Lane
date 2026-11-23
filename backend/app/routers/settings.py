"""HTML route for the system-settings interface.

This module is presentation-only.  Mutating settings should use documented
JSON API routes with an appropriate FastAPI dependency, normally
``get_admin_user``.

Example:
    GET /settings
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


router = APIRouter()
FRONTEND_PUBLIC_DIR = Path(__file__).resolve().parents[3] / "frontend" / "public"


@router.get("/settings", include_in_schema=False)
async def settings_page() -> FileResponse:
    """Serve the settings frontend.

    Example:
        GET /settings -> 200 text/html (settings.html)
    """
    return FileResponse(FRONTEND_PUBLIC_DIR / "settings.html")