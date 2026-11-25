from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.auth import get_current_user
from app.routers.vision import (
    get_vision_worker_client,
    router as vision_router,
)
from app.services.vision_worker_client import VisionWorkerResponseError


class FakeVisionWorkerClient:
    def __init__(self, status=None, error=None):
        self.status = status
        self.error = error
        self.requested_status = False

    async def get_inference_status(self):
        self.requested_status = True
        if self.error is not None:
            raise self.error
        return self.status


def test_authenticated_backend_route_proxies_all_camera_inference_results():
    status = {
        "detector": {"loaded": True},
        "active_camera_ids": [7],
        "inference_fps": 10.0,
        "camera_results": {
            "7": {
                "camera_id": 7,
                "frame_sequence": 19,
                "timestamp": 1234.5,
                "status": "ok",
                "inference_fps": 10.0,
                "detections": [
                    {
                        "class_id": 2,
                        "class_name": "car",
                        "vehicle_class": "4W",
                        "track_id": 3,
                        "confidence": 0.9,
                        "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
                        "preview_bbox": {
                            "x1": 0.1,
                            "y1": 0.2,
                            "x2": 0.3,
                            "y2": 0.4,
                        },
                    }
                ],
                "error": None,
            }
        },
    }
    worker_client = FakeVisionWorkerClient(status)
    app = FastAPI()
    app.include_router(vision_router)
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": "test-user",
        "role": "USER",
    }
    app.dependency_overrides[get_vision_worker_client] = lambda: worker_client

    response = TestClient(app).get(
        "/api/v1/vision/inference/status",
    )

    assert response.status_code == 200
    assert worker_client.requested_status
    assert response.json()["inference_fps"] == 10.0
    assert response.json()["actual_inference_fps"] == 0
    assert response.json()["camera_results"]["7"]["actual_inference_fps"] == 0
    detection = response.json()["camera_results"]["7"]["detections"][0]
    assert detection["vehicle_class"] == "4W"
    assert detection["track_id"] == 3
    assert detection["preview_bbox"] == {
        "x1": 0.1,
        "y1": 0.2,
        "x2": 0.3,
        "y2": 0.4,
    }


def test_backend_route_returns_worker_error_detail():
    worker_client = FakeVisionWorkerClient(
        error=VisionWorkerResponseError(401, "Worker control authentication required.")
    )
    app = FastAPI()
    app.include_router(vision_router)
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": "test-user",
        "role": "USER",
    }
    app.dependency_overrides[get_vision_worker_client] = lambda: worker_client

    response = TestClient(app).get("/api/v1/vision/inference/status")

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The vision worker returned HTTP 401: Worker control authentication required."
    )
