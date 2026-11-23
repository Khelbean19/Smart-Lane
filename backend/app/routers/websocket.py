"""Reserved router for real-time WebSocket communication.

Future live detections, signal updates, and operator notifications can be
implemented here with FastAPI's ``WebSocket`` and ``WebSocketDisconnect`` APIs.
The module is intentionally empty until a connection lifecycle and broadcast
protocol are specified.

Example protocol shape for a future implementation::

    @router.websocket("/api/v1/ws/events")
    async def events(websocket: WebSocket) -> None:
        await websocket.accept()
        # Authenticate, subscribe, and forward or yield server events.

Authenticated WebSocket handlers should validate the bearer credential before
``accept()`` and should always remove their subscriber during disconnection.
"""