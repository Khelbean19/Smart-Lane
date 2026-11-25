import asyncio
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.vision import camera as camera_module
from app.vision.camera import CameraManager, CameraUnavailable
from app.vision.worker import (
    VisionWorker,
    create_worker_app,
    reconcile_camera_resolutions_on_startup,
)


WORKER_TOKEN = "test-worker-token"


class FakeCapture:
    def __init__(self, source, opened=True, shape=(8, 8)):
        self.source = source
        self.opened = opened
        self.released = False
        self.settings = {}
        self.shape = shape

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
        return True, np.full((*self.shape, 3), shade, dtype=np.uint8)

    def release(self):
        self.released = True


def worker_headers():
    return {"Authorization": f"Bearer {WORKER_TOKEN}"}


class FakeDetector:
    def __init__(self, bbox=None):
        self.loaded = False
        self.predicted_frames = []
        self.bbox = bbox or {"x1": 1.0, "y1": 2.0, "x2": 8.0, "y2": 9.0}

    def warmup(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def predict(self, frame):
        self.predicted_frames.append(frame)
        return [
            {
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.95,
                "bbox": self.bbox,
            }
        ]

    def status(self):
        return {
            "loaded": self.loaded,
            "model_path": "fake.pt",
            "device": "cpu",
            "confidence": 0.4,
            "iou": 0.5,
            "image_size": 640,
            "classes": [0, 2],
        }


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


def test_reconcile_restarts_inference_when_detection_resolution_changes():
        class RecordingInferenceRunner:
            def __init__(self):
                self.restart_camera_ids = []

            def sync(self, camera_ids, cameras, restart_camera_ids=None):
                self.restart_camera_ids.append(set(restart_camera_ids or ()))

        manager = CameraManager(capture_factory=lambda source: FakeCapture(source))
        inference_runner = RecordingInferenceRunner()
        worker = VisionWorker(
            camera_manager=manager,
            detector=FakeDetector(),
            inference_runner=inference_runner,
        )
        camera_config = {"camera_id": 10, "source": "0"}

        asyncio.run(worker.reconcile_cameras([camera_config], [10], retry_failed=False))
        asyncio.run(
            worker.reconcile_cameras(
                [{**camera_config, "detection_width": 320}],
                [10],
                retry_failed=False,
            )
        )

        assert inference_runner.restart_camera_ids == [set(), {10}]
        manager.stop_all()


def test_startup_reconciles_dimensions_against_global_cap(monkeypatch):
    class RecordingRepository:
        def __init__(self):
            self.cap = None

        def reconcile_global_frame_cap(self, max_width, max_height):
            self.cap = (max_width, max_height)
            return [
                {
                    "camera_id": 10,
                    "previous_width": 640,
                    "previous_height": 480,
                    "detection_width": 320,
                    "detection_height": 240,
                }
            ]

    monkeypatch.setenv("VISION_MAX_FRAME_WIDTH", "320")
    monkeypatch.setenv("VISION_MAX_FRAME_HEIGHT", "240")
    repository = RecordingRepository()
    messages = []
    monkeypatch.setattr(
        "app.vision.worker.logger.info",
        lambda message, *args: messages.append(message.format(*args)),
    )

    reconcile_camera_resolutions_on_startup(repository)

    assert repository.cap == (320, 240)
    assert messages == [
        "Camera 10 detection resolution recalculated from 640x480 to 320x240 "
        "for global frame cap 320x240."
    ]


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
    app = create_worker_app(
        manager,
        FakeDetector(bbox={"x1": 100.0, "y1": 20.0, "x2": 120.0, "y2": 40.0}),
    )
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

        inference_status = None
        for _ in range(50):
            inference_status = client.get(
                "/api/v1/inference/status",
                headers=worker_headers(),
            )
            if inference_status.json()["camera_results"].get("10", {}).get("detections"):
                break
            time.sleep(0.01)
        assert inference_status.status_code == 200
        detected = inference_status.json()["camera_results"]["10"]["detections"][0]
        assert detected["vehicle_class"] == "4W"
        assert detected["track_id"] == 1
        assert detected["preview_bbox"] is not None

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
            "detection_width": 640,
            "detection_height": 480,
            "preprocessing_mode": "fit",
            "roi_x": 0.0,
            "roi_y": 0.0,
            "roi_width": 1.0,
            "roi_height": 1.0,
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


def test_worker_caps_stream_and_publishes_detection_sized_frames(monkeypatch):
    monkeypatch.setenv("VISION_WORKER_TOKEN", WORKER_TOKEN)

    def capture_factory(source):
        return FakeCapture(source, shape=(480, 640))

    manager = CameraManager(
        capture_factory=capture_factory,
        max_frame_width=320,
        max_frame_height=240,
    )
    app = create_worker_app(manager)

    with TestClient(app) as client:
        response = client.put(
            "/api/v1/cameras/reconcile",
            headers=worker_headers(),
            json={
                "cameras": [
                    {
                        "camera_id": 9,
                        "source": "0",
                        "detection_width": 160,
                        "detection_height": 120,
                    }
                ],
                "active_camera_ids": [9],
            },
        )
        assert response.status_code == 200
        status = response.json()[0]
        assert status["max_frame_resolution"] == {"width": 320, "height": 240}
        assert status["source_resolution"] == {"width": 640, "height": 480}
        assert status["config"]["detection_width"] == 160

        encoded_frame = manager.get(9).next_frame(0, timeout=1)
        detection_frame = manager.get(9).next_detection_frame(0, timeout=1)
        assert encoded_frame is not None
        assert detection_frame is not None
        preview = cv2.imdecode(
            np.frombuffer(encoded_frame[1], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert preview.shape == (240, 320, 3)
        assert detection_frame[1].shape == (120, 160, 3)


def test_manager_does_not_replace_camera_when_old_capture_fails_to_stop(monkeypatch):
    instances = []

    class StopFailingCamera:
        def __init__(
            self,
            camera_id,
            source,
            _capture_factory,
            _max_frame_width,
            _max_frame_height,
        ):
            self.camera_id = camera_id
            self.source = source
            self.state = "stopped"
            self.config = {
                "detection_width": 640,
                "detection_height": 480,
                "preprocessing_mode": "fit",
                "roi_x": 0.0,
                "roi_y": 0.0,
                "roi_width": 1.0,
                "roi_height": 1.0,
            }
            instances.append(self)

        def configure(self, updates):
            self.config.update(updates)
            return self.status()

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
                    **self.config,
                },
                "max_frame_resolution": {"width": 1280, "height": 720},
                "source_resolution": None,
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


def test_inference_status_and_latest_detections_are_authenticated(monkeypatch):
    monkeypatch.setenv("VISION_WORKER_TOKEN", WORKER_TOKEN)

    def capture_factory(source):
        return FakeCapture(source, shape=(480, 640))

    manager = CameraManager(capture_factory=capture_factory)
    detector = FakeDetector()
    app = create_worker_app(
        manager,
        detector=detector,
        inference_runner=None,
    )
    worker = app.state.vision_worker
    assert isinstance(worker, VisionWorker)
    assert worker.camera_manager is manager
    assert worker.detector is detector

    with TestClient(app) as client:
        assert client.get("/api/v1/inference/status").status_code == 401

        reconcile = client.put(
            "/api/v1/cameras/reconcile",
            headers=worker_headers(),
            json={
                "cameras": [{"camera_id": 12, "source": "0"}],
                "active_camera_ids": [12],
            },
        )
        assert reconcile.status_code == 200

        status = client.get(
            "/api/v1/inference/status",
            headers=worker_headers(),
        )
        assert status.status_code == 200
        assert status.json()["detector"]["loaded"] is True
        assert status.json()["active_camera_ids"] == [12]
        assert status.json()["inference_fps"] == 10.0
        assert status.json()["camera_results"]["12"]["status"] in {"running", "ok"}
        assert isinstance(
            status.json()["camera_results"]["12"]["detections"],
            list,
        )

        for _ in range(20):
            result = client.get(
                "/api/v1/cameras/12/detections/latest",
                headers=worker_headers(),
            )
            if result.status_code == 200 and result.json()["detections"]:
                break
            time.sleep(0.05)

        assert result.status_code == 200
        assert result.json()["camera_id"] == 12
        assert result.json()["detections"][0]["class_name"] == "car"
        assert result.json()["detections"][0]["vehicle_class"] == "4W"
        assert result.json()["detections"][0]["track_id"] > 0
        preview_bbox = result.json()["detections"][0]["preview_bbox"]
        assert preview_bbox["x1"] == 1 / 640
        assert preview_bbox["y1"] == 2 / 480
        assert preview_bbox["x2"] == 8 / 640
        assert preview_bbox["y2"] == 9 / 480
        assert result.json()["inference_fps"] == 10.0

        assert client.get(
            "/api/v1/cameras/404/detections/latest",
            headers=worker_headers(),
        ).status_code == 404

    assert detector.loaded is False
