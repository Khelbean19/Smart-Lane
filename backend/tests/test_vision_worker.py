import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.vision import camera as camera_module
from app.vision.camera import CameraManager, CameraUnavailable
from app.vision.worker import create_worker_app


WORKER_TOKEN = "test-worker-token"


class FakeCapture:
    def __init__(self, source, opened=True):
        self.source = source
        self.opened = opened
        self.released = False
        self.settings = {}

    def isOpened(self):
        return self.opened

    def set(self, property_id, value):
        self.settings[property_id] = value
        return True

    def read(self):
        time.sleep(0.005)
        if self.released:
            return False, None
        shade = int(self.source) * 80 if str(self.source).isdecimal() else 160
        return True, np.full((8, 8, 3), shade, dtype=np.uint8)

    def release(self):
        self.released = True


def worker_headers():
    return {"Authorization": f"Bearer {WORKER_TOKEN}"}


def test_worker_starts_with_an_empty_inactive_registry(monkeypatch):
    monkeypatch.setenv("VISION_WORKER_TOKEN", WORKER_TOKEN)
    opened_sources = []

    def capture_factory(source):
        opened_sources.append(source)
        return FakeCapture(source)

    manager = CameraManager(capture_factory=capture_factory)
    app = create_worker_app(manager)

    with TestClient(app) as client:
        assert client.get("/api/v1/cameras").status_code == 401
        assert client.get(
            "/api/v1/cameras",
            headers=worker_headers(),
        ).json() == []
        assert client.get("/camera/10/stream").status_code == 404
        assert opened_sources == []


def test_worker_reconciles_registered_cameras_and_streams_only_active(monkeypatch):
    monkeypatch.setenv("VISION_WORKER_TOKEN", WORKER_TOKEN)
    opened_sources = []
    captures = []

    def capture_factory(source):
        opened_sources.append(source)
        capture = FakeCapture(source)
        captures.append(capture)
        return capture

    manager = CameraManager(capture_factory=capture_factory)
    app = create_worker_app(manager)
    registry = [
        {"camera_id": 10, "source": "0"},
        {"camera_id": 22, "source": "rtsp://camera.local/live"},
    ]

    with TestClient(app) as client:
        response = client.put(
            "/api/v1/cameras/reconcile",
            headers=worker_headers(),
            json={
                "cameras": registry,
                "active_camera_ids": [10],
            },
        )
        assert response.status_code == 200
        statuses = response.json()
        assert [status["camera_id"] for status in statuses] == [10, 22]
        assert [status["state"] for status in statuses] == ["running", "stopped"]
        assert opened_sources == [0]

        frame = manager.get(10).next_frame(0, timeout=1)
        assert frame is not None
        with client.stream("GET", "/camera/10/stream") as stream:
            assert stream.status_code == 200
            assert b"Content-Type: image/jpeg" in next(stream.iter_raw())
        assert client.get("/camera/22/stream").status_code == 503

        configured = client.patch(
            "/api/v1/cameras/10/config",
            headers=worker_headers(),
            json={"width": 640, "height": 480, "jpeg_quality": 90},
        )
        assert configured.status_code == 200
        assert configured.json()["config"] == {
            "width": 640,
            "height": 480,
            "fps": None,
            "jpeg_quality": 90,
        }

        deactivated = client.put(
            "/api/v1/cameras/reconcile",
            headers=worker_headers(),
            json={
                "cameras": registry,
                "active_camera_ids": [22],
            },
        )
        assert [status["state"] for status in deactivated.json()] == [
            "stopped",
            "running",
        ]
        assert opened_sources == [0, 0, "rtsp://camera.local/live"]

    assert all(capture.released for capture in captures)


def test_worker_reports_camera_failure_and_rejects_invalid_registry(monkeypatch):
    monkeypatch.setenv("VISION_WORKER_TOKEN", WORKER_TOKEN)
    captures = []

    def capture_factory(source):
        capture = FakeCapture(source, opened=False)
        captures.append(capture)
        return capture

    manager = CameraManager(capture_factory=capture_factory)
    app = create_worker_app(manager)

    with TestClient(app) as client:
        failed = client.put(
            "/api/v1/cameras/reconcile",
            headers=worker_headers(),
            json={
                "cameras": [{"camera_id": 7, "source": "0"}],
                "active_camera_ids": [7],
            },
        )
        assert failed.status_code == 200
        assert failed.json()[0]["state"] == "error"
        assert "could not be opened" in failed.json()[0]["error"]

        assert client.put(
            "/api/v1/cameras/reconcile",
            headers=worker_headers(),
            json={
                "cameras": [{"camera_id": 7, "source": "0"}],
                "active_camera_ids": [8],
            },
        ).status_code == 422
        assert client.patch(
            "/api/v1/cameras/7/config",
            headers=worker_headers(),
            json={},
        ).status_code == 422
        assert client.get(
            "/api/v1/cameras",
            headers=worker_headers(),
        ).json()[0]["state"] == "error"
        assert captures[0].released


def test_camera_manager_construction_does_not_open_a_device(monkeypatch):
    def fail_if_opened(*args, **kwargs):
        raise AssertionError("Constructing camera management opened a camera.")

    monkeypatch.setattr(cv2, "VideoCapture", fail_if_opened)
    manager = CameraManager()

    assert manager.statuses() == []


def test_manager_does_not_replace_camera_when_old_capture_fails_to_stop(monkeypatch):
    instances = []

    class StopFailingCamera:
        def __init__(self, camera_id, source, _capture_factory):
            self.camera_id = camera_id
            self.source = source
            self.state = "stopped"
            instances.append(self)

        def start(self):
            self.state = "running"
            return self.status()

        def stop(self):
            self.state = "error"
            raise CameraUnavailable("capture thread is still running")

        def status(self):
            return {
                "camera_id": self.camera_id,
                "source": str(self.source),
                "state": self.state,
                "config": {
                    "width": None,
                    "height": None,
                    "fps": None,
                    "jpeg_quality": 80,
                },
                "frame_sequence": 0,
                "error": "capture thread is still running" if self.state == "error" else None,
            }

    monkeypatch.setattr(camera_module, "CameraCapture", StopFailingCamera)
    manager = CameraManager()

    manager.reconcile([{"camera_id": 8, "source": "0"}], [8])
    statuses = manager.reconcile([{"camera_id": 8, "source": "1"}], [8])

    assert len(instances) == 1
    assert statuses[0]["source"] == "0"
    assert statuses[0]["state"] == "error"
