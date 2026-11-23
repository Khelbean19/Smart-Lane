"""HTML route for the reporting interface.

Report-generation endpoints and report exports are feature modules elsewhere;
this router serves the browser page that invokes them.

Example:
    GET /reports
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


router = APIRouter()
FRONTEND_PUBLIC_DIR = Path(__file__).resolve().parents[3] / "frontend" / "public"


@router.get("/reports", include_in_schema=False)
async def reports_page() -> FileResponse:
    """Serve the reports frontend.

    Example:
        GET /reports -> 200 text/html (reports.html)
    """
    return FileResponse(FRONTEND_PUBLIC_DIR / "reports.html")