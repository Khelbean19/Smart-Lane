---
title: SmartLane
emoji: 🚦
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# SmartLane

SmartLane is a proof-of-concept smart traffic system for narrow, single-lane,
bi-directional roads. It uses two webcam streams, computer vision, virtual traffic
signals, and a browser-based operations dashboard. The project is developed by
**Vision Vanguard** for BSIT 3-3 and supports **UN SDG 11 - Target 11.2**:
safer and more sustainable transport systems.

The current milestone is a working backend that serves the frontend UI and runs
two simultaneously assigned USB or network camera streams through a separate
vision worker.

## Project background

On residential streets and narrow barangay or access roads, vehicles traveling
in opposite directions often have to stop, wait, or reverse when they meet.
SmartLane evaluates the approaching vehicles and can distinguish between
two-wheeled and four-wheeled traffic. Its traffic decision rules allow safe
opposing combinations while preventing unsafe combinations, and authorized
enforcers can monitor and control the system.

This course prototype:

- Uses two USB webcams or configurable network camera sources.
- Runs on one host computer.
- Displays virtual traffic lights in the browser rather than controlling physical
  traffic hardware.
- Assumes a predefined pairing model; actual road width and clearance are not
  measured by the current implementation.
- Targets browser access on desktop and mobile clients.

The repository documentation now supersedes older Flask, SQLite, and React build
tool proposals in the course PDFs. The implemented stack uses FastAPI,
PostgreSQL, psycopg2, and static browser-rendered React/Babel pages.

## Tech stack overview

### Implemented technologies

| Area | Technology | Repository use |
|---|---|---|
| Backend and API | Python 3.15, FastAPI, Uvicorn | REST API, authentication, static-page routing, camera coordination |
| Frontend | HTML, CSS, JavaScript, React 18 via CDN/Babel | Login and operations pages |
| Data | PostgreSQL 18, psycopg2, repository modules | Persistent users, cameras, assignments, and project records |
| Computer vision | OpenCV, Ultralytics, NumPy | Camera capture, frame preprocessing, and future detector integration |
| Real-time video | Separate FastAPI vision worker, MJPEG over HTTP | Two independent camera streams |
| Database administration | pgAdmin 4 | Optional development database browser and schema maintenance |
| Containers | Docker Desktop with Docker Compose | Runs PostgreSQL locally |
| Package/environment management | uv | Installs the locked Python environment and dependencies |
| Source control | Git and GitHub | Team collaboration and version control |
| Testing | pytest | Backend unit and integration tests |

### Prerequisite installations

Install these before starting local development:

1. **Git** - for cloning and updating the repository.
2. **Docker Desktop** - starts and manages the PostgreSQL container.
3. **uv** - manages Python and installs the dependencies in `pyproject.toml`.
   On Windows, uv can be installed from PowerShell:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
4. **Visual Studio Code** - recommended editor and integrated terminal.
5. **pgAdmin 4** - optional, but recommended for viewing and editing PostgreSQL.
   It is a development tool, not a runtime dependency.
6. **Two local webcam devices or accessible network camera URLs** - required for
   testing assigned dual camera streams.

The project requires **Python 3.15 or newer** according to `pyproject.toml`.
The current frontend is served as static files with React and Babel loaded from
CDN scripts, so Node.js and npm are **not required for the current milestone**.
They are only needed if the team later moves the frontend to a compiled build
pipeline.

Additional conceptual components such as Chart.js, SMS alert integration,
Roboflow dataset tooling, and a production YOLO model are planned or
development-support tools. They are not all implemented in the current backend.

## Initial project setup

### 1. Clone the repository

```powershell
git clone https://github.com/Khelbean19/Smart-Lane.git
cd Smart-Lane
```

If the repository is already open in VS Code, open an integrated terminal at
the repository root and continue from that location.

### 2. Create the Python environment

```powershell
uv python install 3.15
uv venv .venv --python 3.15
uv sync
```

For activation in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Alternatively, every command can use `uv run` without manually activating the
virtual environment.

`uv sync` reads `pyproject.toml` and `uv.lock` and installs FastAPI, SQLAlchemy,
PostgreSQL drivers, OpenCV, Ultralytics, report tooling, pytest, and the other
project dependencies.

### 3. Start PostgreSQL

Create one repository-root `.env` for both Docker Compose and the Python
processes. Compose reads `.env` automatically, and the development commands use
the same file explicitly.

```dotenv
POSTGRES_USER=postgres
POSTGRES_PASSWORD=replace-with-a-local-password
POSTGRES_DB=smartlane-db
POSTGRES_PORT=5440
POSTGRES_HOST=127.0.0.1

AUTH_SECRET_KEY=replace-with-a-long-random-secret
VISION_WORKER_HOST=127.0.0.1
VISION_WORKER_PORT=8001
VISION_WORKER_TOKEN=replace-with-a-different-long-random-secret
VISION_WORKER_AUTOSTART=true
VISION_MAX_FRAME_WIDTH=1280
VISION_MAX_FRAME_HEIGHT=720
```

`.env` is ignored by Git and must not be committed. Replace the example
passwords and secrets with local values. Validate the Compose configuration and
start PostgreSQL:

```powershell
docker compose config --quiet
docker compose up -d db
docker compose ps
```

PostgreSQL is exposed on the host port configured by `POSTGRES_PORT`.
The container stores its data in the `postgres-data` Docker volume. PostgreSQL
18 uses `/var/lib/postgresql` as its data-volume mount; this differs from the
`/var/lib/postgresql/data` path used by PostgreSQL 17 and older. Do not change
the volume target when using the `postgres:18` image. This path correction does
not migrate data from an existing volume created with the old target. If an
earlier attempt created data, back it up and migrate it before starting with
the corrected mount. Do not use `docker compose down -v` as routine cleanup
because it deletes the database volume. See the
[official Postgres image documentation](https://hub.docker.com/_/postgres/)
for version-specific storage details.

Use the Compose project from the repository root in Docker Desktop rather than
starting a bare `postgres:18` image with the **Run** button. The Compose service
supplies the required password, host port, and persistent volume. If startup
fails, inspect `docker compose ps -a` and `docker compose logs --tail 50 db`.
If host port `5440` is already in use, set `POSTGRES_PORT` in `.env` to a free
port and use that port in pgAdmin; PostgreSQL inside the container remains on
`5432`.

In pgAdmin 4, add a server connection with:

- Host: `127.0.0.1`
- Port: the value of `POSTGRES_PORT`, for example `5440`
- Username: `postgres`
- Password: the password used in `.env`
- Database: `smartlane-db`

These are the example `.env` values. If you change `POSTGRES_USER` or
`POSTGRES_DB`, use your selected values in pgAdmin and in the SQL commands
below.

### 4. Initialize the database

Open `backend/schema.sql` in pgAdmin's Query Tool while the `smartlane-db`
database is selected, then execute it. The script creates the enums and the
SmartLane tables and camera preprocessing columns.

For a first database setup, pgAdmin is the clearest option. A command-line
alternative is:

```powershell
Get-Content backend\schema.sql -Raw | docker compose exec -T db psql -U postgres -d smartlane-db
```

For an existing database, apply the upgrade scripts through pgAdmin or execute:

```powershell
Get-Content backend\scripts\upgrade_camera_assignments.sql -Raw | docker compose exec -T db psql -U postgres -d smartlane-db
Get-Content backend\scripts\upgrade_camera_preprocessing.sql -Raw | docker compose exec -T db psql -U postgres -d smartlane-db
```

The schema contains no guaranteed initial login account. Create the first
administrator through the database as Argon2-protected password data. Use the
application helper to generate a hash, for example:

```powershell
uv run python -c "from backend.app.core.security import hash_password; print(hash_password('change-this-password'))"
```

Insert the resulting hash and account fields into `users` through pgAdmin.
Treat the plaintext password as a temporary bootstrap credential and change it
after setup.

### 5. Keep local environment configuration consistent

Keep the database and application settings in the same root `.env`. Docker
Compose loads `.env` by default; `uv run --env-file .env` loads it for Python.
Compose does **not** load `.env.dev` automatically. If you already have an
`.env.dev`, merge any needed application-only settings into `.env` rather than
maintaining different copies of `POSTGRES_USER`, `POSTGRES_PASSWORD`,
`POSTGRES_DB`, or `POSTGRES_PORT`. Do not overwrite an existing `.env` when
merging. Values exported in the shell can override the corresponding `.env`
entries during Compose interpolation, so clear conflicting exported
`POSTGRES_*` variables or make them match the file.

Generate independent secrets with:

```powershell
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The vision and database variables are consumed directly by the backend and
worker processes. `VISION_WORKER_TOKEN` must be identical in every process.
Keep `AUTH_SECRET_KEY` private because it signs bearer tokens.

Optional placeholder variables in older local `.env` files may include
`SMS_API_URL`, `SMS_API_KEY`, `YOLO_WEIGHTS_PATH`, and camera sources. They are
not required by the current two-camera milestone.

## Development setup

### Run the database

```powershell
docker compose up -d db
```

Stop it with:

```powershell
docker compose down
```

Stopping with `docker compose down` preserves the `postgres-data` volume.

### Run the vision worker

Open a terminal at the repository root:

```powershell
uv run --env-file .env python -m backend.app.vision.worker
```

The worker listens only on `127.0.0.1:8001`. It opens no cameras until the
backend database assignment assigns a camera to a live slot. The worker keeps
an independent capture thread for each assigned camera and schedules inference
fairly through its shared detector.

The detector loads `backend/models_weights/yolo26n.pt` by default. The weight
can be changed or inference tuned with these worker environment variables:

- `VISION_MODEL_PATH` (defaults to the bundled `backend/models_weights/yolo26n.pt`)
- `VISION_MODEL_DEVICE` (defaults to `cpu`; accepts an accelerator/device ID)
- `VISION_CONFIDENCE_THRESHOLD` (0 through 1, default `0.4`)
- `VISION_IOU_THRESHOLD` (0 through 1, default `0.5`)
- `VISION_MODEL_IMAGE_SIZE` (positive integer, default `640`)
- `VISION_MODEL_CLASSES` (optional comma-separated, unique non-negative class IDs)
- `VISION_INFERENCE_FPS` (greater than 0 and at most 120, default `10`)

### YOLO26 benchmark and detection testing

Add test videos to `backend/tests/footage`, and keep model weights under
`backend/models_weights/` (`backend/model_weights/` is also supported). Run:

```powershell
uv run python backend/tests/benchmark_yolo.py
```

The benchmark is isolated from the server environment: it does not read or set
`VISION_MODEL_PATH`. It tests nano, small, and medium weights in 16:9 and 4:3
presentations, with both production `fit` and `crop` preprocessing modes. The default
model input is 640x480, matching the repository's 640-pixel-wide detection pipeline.

Useful options include `--model-size n s`, `--video footage_sample1`,
`--aspect 16:9 4:3`, `--mode fit crop`, `--duration 10`, `--device cpu`,
`--skip-videos`, and `--report custom_name.txt`.

Outputs are written to `backend/tests/benchmarks/`:

- `benchmark_results.txt`: human-readable speed, detection, confidence, and label-backed
  accuracy statistics.
- `benchmark_results.json`: the same machine-readable result data.
- `detection_videos/`: 10-second 640x480 MP4 clips with vehicle bounding boxes.

For label-backed precision, recall, and AP@0.5, place YOLO-format label text files in
either `labels/<video-stem>/` or `<video-stem>_labels/` beside the footage. Frame label
names should use six-digit frame numbers or plain frame indices. When no labels exist,
the report clearly marks accuracy as unavailable instead of estimating it.
The worker exposes authenticated inference data through:

- `GET /api/v1/inference/status`
- `GET /api/v1/cameras/{camera_id}/detections/latest`

Both endpoints require `VISION_WORKER_TOKEN`. The latest-result endpoint
reports one result per frame sequence and is not a detection queue. Each
vehicle detection includes a per-camera in-memory `track_id`, a `vehicle_class`
group, and its native model `class_name`. Track IDs survive brief missed
detections, expire after 1.5 seconds, and reset when a camera or worker
restarts; detection events are not persisted.

`bbox` uses detection-input pixel coordinates. `preview_bbox` uses normalized
coordinates relative to the capped camera preview and accounts for letterbox
padding and crop ROI clipping. `inference_fps` remains the configured
per-camera target for compatibility. `actual_inference_fps` reports achieved
worker-wide and per-camera rates, while `inference_duration_ms` reports the
latest model-call duration; `inferences_completed` counts completed model calls.
The authenticated main-backend endpoint
`GET /api/v1/vision/inference/status` proxies the all-camera snapshot for Live
Control, which continues to poll at the configured target rate.

### Multi-camera performance notes

The worker keeps capture independent per camera but uses one fair inference
scheduler because calls to the shared YOLO model are serialized. When inference
falls behind, each scheduled call uses the newest available camera frame rather
than building a stale-frame queue. Detection resizing/ROI preprocessing is
performed only when the scheduler requests a frame; preview capping and JPEG
encoding still run for each captured frame so MJPEG behavior is unchanged.

This removes repeated detection preprocessing and frame copies, but does not
make the model itself faster. For example, with a 30 FPS camera and the default
10 FPS inference target, preprocessing/copy work falls from up to 30 frames per
second to at most 10 (about two-thirds fewer operations on that path). This is
a workload-based estimate, not a measured end-to-end FPS improvement. The
worker status endpoint exposes achieved FPS to measure deployment results.

Measure with the actual model and camera sources on the deployment machine:

```powershell
uv run python -m backend.tests.benchmark_vision_worker --source 0 --duration 30
uv run python -m backend.tests.benchmark_vision_worker --source 0 --source 1 --duration 30
```

The benchmark reports capture and inference rates, model-call duration, and
configured targets. It requires the local model weights and working camera
sources, and does not include browser/network delivery; inspect the MJPEG
streams separately when measuring end-to-end preview load.

Software optimization can reduce redundant preprocessing, copies, scheduling
contention, and stale work. It cannot remove the shared model's compute limit:
if achieved aggregate inference is below the sum of the desired per-camera
targets, consider a smaller model or input size, a lower target rate, or
accelerated inference hardware. CPU capacity also affects camera decoding and
JPEG encoding; multiple USB cameras may be limited by their shared bus, while
network cameras depend on host/network throughput. No architecture overhaul is
indicated before measurements show these code-level changes are insufficient.

### Run the backend and frontend

Open another terminal at the repository root:

```powershell
uv run --env-file .env fastapi dev backend/app/main.py
```

The backend opens the PostgreSQL connection and serves both the API and the
static frontend. Open:

- UI login: http://127.0.0.1:8000/
- Dashboard: http://127.0.0.1:8000/dashboard
- Live Control: http://127.0.0.1:8000/live
- Reports: http://127.0.0.1:8000/reports
- Users: http://127.0.0.1:8000/users
- Settings: http://127.0.0.1:8000/settings
- Interactive API: http://127.0.0.1:8000/docs

Camera streams are served by the worker and can be viewed at:

```text
http://127.0.0.1:8001/camera/{camera_id}/stream
```

The worker and browser must run on the same host. In a browser, `127.0.0.1`
refers to the browser device, not an upstream development machine.

### Assign and test two cameras

1. Sign in with an active `ADMINISTRATOR` account.
2. Open **Live Control**.
3. Assign one database camera record to each live slot.
4. Confirm that the browser requests the assignment update.
5. Verify both MJPEG streams display under the worker's `/camera/{camera_id}/stream`
   endpoint.

Camera source values may be numeric OpenCV device indexes or network URLs.
The worker reports startup failures as `FAULT`; failure of one camera does not
prevent the other assigned camera from starting. Runtime status is represented
by `ONLINE`, `OFFLINE`, or `FAULT`.

The worker applies a global output/detection cap from
`VISION_MAX_FRAME_WIDTH` and `VISION_MAX_FRAME_HEIGHT`, with defaults of
1280x720. Detection preprocessing supports `fit` letterboxing and normalized
ROI `crop` modes. These settings are deployment limits and are not ordinary
administrator-editable frontend values. At worker startup, the last applied
cap is checked in `system_settings`; if it changed, camera detection
dimensions are resized to fit the new cap and saved before the worker starts.

### Authentication testing

Log in through the browser or obtain a bearer token:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/auth/token" ^
  -H "Content-Type: application/x-www-form-urlencoded" ^
  -d "username=admin&password=change-this-password"
```

Send the returned token to protected endpoints:

```powershell
curl.exe "http://127.0.0.1:8000/api/v1/me" ^
  -H "Authorization: Bearer <access_token>"
```

Ordinary browser sessions expire after 15 minutes. Selecting **Remember me**
extends the signed token lifetime to seven days. The server also reloads the
account on each protected request, so disabled or deleted accounts stop
working immediately.

### Run tests

Run the focused backend suite from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path backend).Path
$env:VISION_WORKER_HOST = "127.0.0.1"
$env:VISION_WORKER_PORT = "8001"
uv run pytest backend/tests
```

Useful categories include:

- Camera preprocessing and frame sizing.
- Camera repository and slot assignment behavior.
- Authentication and current-user authorization.
- Vision worker reconciliation and worker client behavior.
- Pairing, decision, and signal-controller logic tests.

## API and source code reference

### Interactive API reference

FastAPI provides an interactive OpenAPI page at:

**http://127.0.0.1:8000/docs**

Use **Authorize** to enter a bearer token, or use the `POST /api/v1/auth/token`
operation to obtain one. The OpenAPI schema is generated directly from route
definitions and Pydantic schemas.

Key route groups include:

- `/api/v1/auth/` and `/api/v1/auth/token` - JSON and OAuth2-compatible login.
- `/api/v1/me` - current authenticated profile.
- `/api/v1/users/` - administrator-only user listing and creation.
- `/api/v1/vision/camera-slots` - read or replace camera-slot assignments.
- `/api/v1/vision/cameras/{camera_id}/settings` - update camera preprocessing.
- `/login`, `/dashboard`, `/live`, `/reports`, `/users`, and `/settings` -
  browser page routes excluded from the interactive API schema.

The vision worker has its own authenticated control API on port `8001`, while
browser camera delivery uses `/camera/{camera_id}/stream`.

### Source layers

| Layer or concern | Location | Status |
|---|---|---|
| FastAPI application entry point | `backend/app/main.py` | Implemented |
| Environment and runtime configuration | `backend/app/config.py`, `.env`, `docker-compose.yml` | Shared local settings for database and vision processes |
| Database connection and cursor sharing | `backend/app/database.py` | Implemented; PostgreSQL through psycopg2 |
| Authentication and authorization | `backend/app/core/security.py`, `backend/app/core/dependencies.py`, `backend/app/routers/auth.py` | Implemented |
| User API and UI | `backend/app/routers/users.py`, `backend/app/schemas/user.py`, `frontend/public/users.html`, `frontend/src/pages/users.js` | Implemented/admin API |
| Camera registry and assignments | `backend/app/services/camera_repository.py`, `backend/app/routers/vision.py`, `backend/app/schemas/vision.py` | Implemented |
| Vision worker and MJPEG streams | `backend/app/vision/camera.py`, `backend/app/vision/pipeline.py`, `backend/app/vision/inference.py`, `backend/app/vision/worker.py` | Implemented; shared-model inference is scheduled across active cameras |
| Camera-to-FastAPI worker client | `backend/app/services/vision_worker_client.py` | Implemented |
| Frontend public pages | `frontend/public/index.html`, `dashboard.html`, `live-control.html`, `reports.html`, `users.html`, `settings.html`, `styles.css` | Implemented as static UI pages |
| Shared browser auth/session behavior | `frontend/public/session.js` | Implemented |
| Page route modules | `backend/app/routers/dashboard.py`, `live_control.py`, `reports.py`, `settings.py` | Page routes implemented |
| Data model DDL | `backend/schema.sql` | Implemented |
| Database migrations and upgrades | `backend/alembic.ini`, `backend/alembic/`, `backend/scripts/` | Partial/incremental upgrade scripts |
| Business pairing, queue, and signal logic | `backend/app/logic/` | Work in progress; test coverage exists for future decisions |
| YOLO detector and detection zones | `backend/app/vision/detector.py`, `backend/app/vision/zones.py` | Detector integration implemented; zones remain separate |
| Alert and audit services | `backend/app/services/alert_service.py`, `audit_service.py` | Work in progress |
| Report generation and exports | `backend/app/routers/reports.py`, `backend/app/services/report_service.py` | Reports page and data modules need implementation |
| Realtime WebSocket events | `backend/app/routers/websocket.py` | Reserved; no live WebSocket protocol yet |
| Dedicated FastAPI video routes | `backend/app/routers/video.py` | Reserved; MJPEG is currently provided by the separate worker |
| Scheduler | `backend/app/services/scheduler.py` | Work in progress |
| Trained datasets and notebooks | `ml/` | Development and training area |
| Model weights | `backend/models_weights/` | Local-only model assets; no weight file is committed |

The directory `frontend/src/` contains React source fragments. The running
frontend currently uses `frontend/public/` and is mounted directly by FastAPI;
it is not a Node/Vite build output yet.

### Backend refactoring candidates

The vision capture lifecycle and worker API/orchestration modules are the most
critical maintainability candidates; keep their responsibilities explicit as
future changes are made. The vision routes also combine database coordination,
worker communication, and response shaping, while the repository uses one
shared PostgreSQL cursor protected by a lock. Those boundaries may merit
separation if API concurrency or route complexity becomes a measured concern.
This performance pass changes only backend vision code and does not attempt a
broader route, database, or frontend rewrite.

## Git workflow

- `main` contains the stable branch.
- `develop` integrates completed team work.
- Use `feature/<short-name>` branches for new features.
- Open a pull request before merging.
- Never commit `.env`, `.env.dev`, database passwords, bearer secrets, generated
  logs, model weights, recordings, or the `.venv` directory.
