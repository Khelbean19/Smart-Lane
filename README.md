# SmartLane

Smart traffic light system with vehicle classification for single-lane, bi-directional roads.
Team: Vision Vanguard (BSIT 3-3) | SDG 11 – Target 11.2

**Stack:** React.js (Vite) · FastAPI · PostgreSQL + SQLAlchemy · OpenCV + YOLOv8 · WebSocket + MJPEG · SMS API

## Setup

### Database
```bash
docker compose up -d
```

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
cp ../.env.example ../.env   # then fill in the values
uvicorn app.main:app --reload
```

### Frontend
Folder structure only for now. The frontend dev will set up the React (Vite) app inside `frontend/`.

### Tests
```bash
cd backend
pytest
```

## Git workflow
- `main` = stable, `develop` = integration, `feature/<name>` = work branches
- Merge via pull requests. Never commit `.env` or model weights.
