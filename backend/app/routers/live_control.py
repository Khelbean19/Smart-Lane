"""HTML route for the live intersection-control interface.

The live-control page uses API routes in this backend to request snapshots,
assignments, and vision updates.  This module itself only serves the page.

Example:
    GET /live
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


router = APIRouter()
FRONTEND_PUBLIC_DIR = Path(__file__).resolve().parents[3] / "frontend" / "public"


@router.get("/live", include_in_schema=False)
async def live_page() -> FileResponse:
    """Serve the live-control frontend.

    Example:
        GET /live -> 200 text/html (live-control.html)
    """
    return FileResponse(FRONTEND_PUBLIC_DIR / "live-control.html")