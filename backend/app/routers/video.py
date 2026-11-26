"""Reserved router for future dedicated video endpoints.

Camera MJPEG/thumbnail streams and related media endpoints can be implemented
as an ``APIRouter`` in this module.  Register the resulting ``router`` in
``backend.app.main`` before testing or deploying a stream.

When adding a route, keep binary responses and resource cleanup explicit:
use a streaming response type, never hold an entire video file in memory, and
close camera or network resources in a ``finally`` or generator cleanup block.
"""