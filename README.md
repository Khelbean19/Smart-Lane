# SmartLane

Smart traffic light system with vehicle classification for single-lane, bi-directional roads.
Team: Vision Vanguard (BSIT 3-3) | SDG 11 – Target 11.2

**Stack:** React.js (Vite) · FastAPI · PostgreSQL + SQLAlchemy · OpenCV + Ultralytics YOLO (YOLO26n candidate) · WebSocket + MJPEG · SMS API

## Setup
### Database
#### Generate or Write a Database Password
It is important to secure even the local development database with a password. Since this project's
database is set up in a Docker container, a password file inside that container must be accessible
for the database to read it later when run.

A randomly-generated password string can be written to a file inside the container using the
following command. Make sure to run this in the WSL Shell:
```bash
(umask 077; openssl rand -base64 24 > postgres-password.txt)
```

If a specific password is desired, simply run the following command to write it to a file:
```bash
echo "yourPasswordHere123!" > postgres-password.txt
```

Either way, when the postgres container is run later, it will reference the saved file and extract
the password content in it. To verify the actual password, run `cat postgres-password.txt`.

#### Running the Database
The postgres database runs on Docker with port 5432. Run the database using the following commands:
```bash
docker run --name postgres -d \
    -p 127.0.0.1:5432:5432 \
    -v postgres-data:/var/lib/postgresql \
    -v "$PWD/postgres-password.txt:/run/secrets/postgres-password:ro" \
    -e POSTGRES_PASSWORD_FILE=/run/secrets/postgres-password \
    -e POSTGRES_DB=smartlane-db
    postgres:18
```

If the port 5432 is already used, simply change the second argument in the `-p` flag with the
desired port number that is available. For example: `-p 127.0.0.1:5433:5432` which uses port 5433
to expose the postgres database in the container.

#### Connecting to pgAdmin
This is applicable if pgAdmin 4 is installed as an application in your computer. Use the following
connection settings when adding a new server connection:
- Host name/address: `localhost` or `127.0.0.1` (However, it is recommended to directly use
`127.0.0.1` for the address as it provides faster connection to the postgres server)
- Username: `postgres`
- Password: (Whatever password has been set in the `postgres-password.txt` file)


### Backend
Run the following commands to initialize the local project repository.
```bash
uv venv .venv --python 3.15     # Creates a virtual environment at Python version 3.15
.venv\Scripts\activate          # Activate the virtual environment
uv sync                         # Downloads and installs project dependencies (similar to pip install)
```

Before running the project, you will need to create an environment file with the following fields.
Name the file to something like `.env.dev`.
- POSTGRES_USERNAME - The default is `postgres`
- POSTGRES_PASSWORD - Use the password set in `postgres-password.txt`
- POSTGRES_DB - The name of the local database (e.g., `smartlane-db`)
- POSTGRES_HOST - Use `localhost` for local development
- POSTGRES_PORT - The port number exposed from the `docker run` command
- AUTH_SECRET_KEY - A private signing key for API bearer tokens. Generate one with
  `python -c "import secrets; print(secrets.token_urlsafe(32))"` and keep it out of source control.
- VISION_WORKER_URL - The worker control API URL; keep it on loopback (`http://127.0.0.1:8001`)
- VISION_WORKER_TOKEN - A private shared bearer token for worker control. Generate one with
  `python -c "import secrets; print(secrets.token_urlsafe(32))"` and keep it out of source control.
- VISION_MAX_FRAME_WIDTH and VISION_MAX_FRAME_HEIGHT - deployment-level maximum dimensions for
  frames produced by the worker. They default to `1280` and `720`; set both before deployment to
  tune the system-wide cap for the machine and camera workload. Values must be positive whole
  numbers and are limited to 7680x4320. These are not administrator-editable from the frontend.

The environment file should look like:
```
POSTGRES_USERNAME=postgres
POSTGRES_PASSWORD=mypassword123!
POSTGRES_DB=smartlane-db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
AUTH_SECRET_KEY=replace-with-a-random-secret
VISION_WORKER_URL=http://127.0.0.1:8001
VISION_WORKER_TOKEN=replace-with-another-random-secret
VISION_MAX_FRAME_WIDTH=1280
VISION_MAX_FRAME_HEIGHT=720
```

The values of these variables should point to your database configuration that were used during the
PostgreSQL database setup.

Use `uv run --env-file .env.dev fastapi dev` to run the backend server with the injected environment
variables. Omitting the environment file will likely cause the project to fail when it attempts to
connect to the postgres database.

For an existing database, apply `backend/scripts/upgrade_camera_assignments.sql` and
`backend/scripts/upgrade_camera_preprocessing.sql` using the PostgreSQL Query Tool. New databases
created from `backend/schema.sql` already include these tables and camera preprocessing columns.
Camera slots start unassigned; camera runtime health remains represented by the `camera_status`
enum (`ONLINE`, `OFFLINE`, or `FAULT`).

### Standalone camera worker
The camera worker is a separate process from FastAPI. It is the only process that opens camera
sources; FastAPI sends the registered camera list and selected slot assignments to it but never
opens devices or relays video. The worker starts with an empty registry and does not open any camera
until an administrator assigns it to a live camera slot.
Start it in a separate terminal from the repository root, using the same `.env.dev` file:
```bash
uv run --env-file .env.dev python -m backend.app.vision.worker
```

The worker binds only to `127.0.0.1:8001`. On worker restart, the next camera-slot request from the
backend restores the saved assignments. Camera startup failures are recorded as `FAULT` in the
database and do not prevent other assigned cameras from running. Stop the worker process
independently from FastAPI.

The browser connects directly to the MJPEG streams:
- Each assigned camera: `http://127.0.0.1:8001/camera/{camera_id}/stream`

The authenticated FastAPI camera-slot API provides:
- `GET /api/v1/vision/camera-slots` — camera slots and runtime state; administrators also receive
  the registered camera options and source details, while enforcers do not receive camera sources.
- `PUT /api/v1/vision/camera-slots` — administrator-only full slot assignment update. The Live
  Control page requires confirmation before sending this request and prevents selecting a camera
  more than once.

The worker accepts a database-provided camera registry through its authenticated
`PUT /api/v1/cameras/reconcile` endpoint. Camera IDs are database IDs; each camera source may be a
numeric device index or a stream URL.

FastAPI uses `VISION_WORKER_URL` and `VISION_WORKER_TOKEN` to call the worker's versioned control
API. The worker validates the shared bearer token. Keep the token secret and consistent in the
environment used by both processes. The worker's loopback binding is not a boundary against other
web pages or processes running on the same machine. The frontend and camera worker must run on the
same host because a browser's `127.0.0.1` refers to its own device.

The worker requests the configured capture size where supported, then enforces the maximum on each
decoded frame before JPEG encoding and MJPEG delivery. Source dimensions shown to administrators
are measured from actual captured frames. Per-camera detection dimensions, `fit`/`crop` mode, and
normalized crop ROI are stored on the camera record and edited in Settings; Live Control provides
a direct Configure link and shows the global cap, measured source size, detection size, and mode.
Both inference preprocessing modes preserve aspect ratio by letterboxing. The worker publishes a
fixed-size latest detection frame for a future model adapter, but YOLO inference is not yet run by
this iteration.

Software resizing limits the worker's outgoing stream and inference input. It cannot reduce
camera-to-worker network bandwidth or decode cost when an RTSP source continues to send its native
high-resolution stream; configure a camera-side low-resolution substream for that case. Before
choosing YOLO26 nano for CPU-only, dual-camera deployment, verify the installed Ultralytics version
supports it and benchmark sustained accuracy, throughput, and latency against a supported nano
baseline (such as YOLO11n) using representative traffic footage. OpenVINO or ONNX Runtime exports
are additional CPU options to benchmark on supported hardware.

The OAuth2 token endpoint is `POST /api/v1/auth/token` and accepts form fields `username` and
`password`. Send its `access_token` as `Authorization: Bearer <token>` when calling
`/api/v1/users/`. Both user-listing and user-creation operations require an active
`ADMINISTRATOR` account; provision the first administrator directly in the database.
The web login keeps an unchecked sign-in in the current browser session (15-minute token). With
**Remember me** checked, the browser persists the sign-in across restarts, but the signed token
expires after seven days and the client returns to the login page when it expires.


### Frontend
FastAPI serves the static React/Babel pages from `frontend/public/`. Shared authentication behavior
is in `session.js`, and `clock.js` formats live timestamps in Philippine Time.

### Tests
```bash
cd backend
pytest
```

## Git workflow
- `main` = stable, `develop` = integration, `feature/<name>` = work branches
- Merge via pull requests. Never commit `.env` or model weights.
