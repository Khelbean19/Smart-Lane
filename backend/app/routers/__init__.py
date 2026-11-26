"""FastAPI router modules for the SmartLane HTTP and WebSocket API.

Routers in this package isolate pages and resources by feature.  Importing
``router`` from a module gives an application entry point or dependency module
to register those routes using ``app.include_router(router)``.

The package currently provides authentication, user administration, vision
camera management, dashboard/status pages, live control, reports, settings,
and reserved modules for future video and WebSocket functionality.
"""