#!/bin/sh
# Start the vision worker in the background, then the main backend on port 7860.
python -m backend.app.vision.worker &
sleep 5
exec fastapi run backend/app/main.py --host 0.0.0.0 --port 7860
