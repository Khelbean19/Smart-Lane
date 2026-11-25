import threading
import time

import numpy as np

from app.vision.inference import CameraInferenceRunner, VehicleTracker


class FakeCamera:
    def __init__(self):
        self.sequence = 0
        self.lock = threading.Lock()

    def next_detection_frame(self, after_sequence, timeout=1.0):
        with self.lock:
            if self.sequence <= after_sequence:
                self.sequence += 1
                return None
            sequence = self.sequence
            self.sequence += 1
        return sequence, np.zeros((32, 32, 3), dtype=np.uint8)


class FakeDetector:
    def predict(self, frame):
        time.sleep(0.01)
        return [
            {
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.9,
                "bbox": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
            }
        ]

    def status(self):
        return {"loaded": True}


def test_runner_keeps_latest_result_and_stops_camera():
    camera = FakeCamera()
    detector = FakeDetector()
    runner = CameraInferenceRunner(detector, inference_fps=100.0)

    runner.attach_camera(7, camera)

    for _ in range(50):
        result = runner.latest(7)
        if result is not None and result["detections"]:
            break
        time.sleep(0.01)

    assert result is not None
    assert result["camera_id"] == 7
    assert result["detections"][0]["class_name"] == "car"
    assert result["detections"][0]["vehicle_class"] == "4W"
    assert result["detections"][0]["track_id"] == 1
    assert runner.status()["active_camera_ids"] == [7]
    assert runner.status()["inference_fps"] == 100.0

    runner.stop(7)
    assert runner.latest(7) is None
    assert runner.status()["active_camera_ids"] == []


def test_runner_fairly_schedules_two_cameras_and_reports_achieved_rate():
    class AlwaysReadyCamera:
        def __init__(self, camera_id):
            self.camera_id = camera_id

        def next_detection_frame(self, after_sequence, timeout=1.0):
            return after_sequence + 1, np.full(
                (1, 1, 3),
                self.camera_id,
                dtype=np.uint8,
            )

    class RecordingDetector(FakeDetector):
        def __init__(self):
            self.camera_ids = []
            self.thread_ids = set()
            self.lock = threading.Lock()

        def predict(self, frame):
            time.sleep(0.005)
            with self.lock:
                self.camera_ids.append(int(frame[0, 0, 0]))
                self.thread_ids.add(threading.get_ident())
            return []

    detector = RecordingDetector()
    runner = CameraInferenceRunner(detector, inference_fps=1000.0)
    runner.sync(
        {1, 2},
        {1: AlwaysReadyCamera(1), 2: AlwaysReadyCamera(2)},
    )

    deadline = time.monotonic() + 2.0
    while len(detector.camera_ids) < 20 and time.monotonic() < deadline:
        time.sleep(0.005)

    status = runner.status()
    counts = {
        camera_id: detector.camera_ids.count(camera_id)
        for camera_id in (1, 2)
    }

    runner.stop_all()

    assert counts[1] >= 5
    assert counts[2] >= 5
    assert abs(counts[1] - counts[2]) <= 1
    assert len(detector.thread_ids) == 1
    assert status["inference_fps"] == 1000.0
    assert status["actual_inference_fps"] > 0
    assert status["camera_results"]["1"]["actual_inference_fps"] > 0
    assert status["camera_results"]["2"]["inference_duration_ms"] > 0


def test_sync_restarts_inference_after_camera_sequence_resets():
    class ResettableCamera:
        def __init__(self):
            self.sequence = 0
            self.condition = threading.Condition()

        def next_detection_frame(self, after_sequence, timeout=1.0):
            with self.condition:
                available = self.condition.wait_for(
                    lambda: self.sequence > after_sequence,
                    timeout=timeout,
                )
                if not available:
                    return None
                return self.sequence, np.zeros((32, 32, 3), dtype=np.uint8)

        def advance_frame(self):
            with self.condition:
                self.sequence += 1
                self.condition.notify_all()

    runner = CameraInferenceRunner(FakeDetector(), inference_fps=100.0)
    camera = ResettableCamera()
    runner.sync({7}, {7: camera})

    for sequence in range(1, 6):
        camera.advance_frame()
        for _ in range(50):
            result = runner.latest(7)
            if result is not None and result["frame_sequence"] == sequence:
                break
            time.sleep(0.01)
        assert result is not None and result["frame_sequence"] == sequence

    with camera.condition:
        camera.sequence = 0
    runner.sync({7}, {7: camera}, restart_camera_ids={7})

    camera.advance_frame()
    for _ in range(50):
        result = runner.latest(7)
        if result is not None and result["frame_sequence"] == 1:
            break
        time.sleep(0.01)
    assert result is not None and result["frame_sequence"] == 1
    runner.stop_all()


def test_vehicle_tracker_preserves_ids_during_motion_and_brief_misses():
    tracker = VehicleTracker()
    detection = {
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.9,
        "bbox": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
    }

    first = tracker.update([detection], now=0.0)
    moved = tracker.update(
        [{
            **detection,
            "bbox": {"x1": 1.0, "y1": 0.0, "x2": 11.0, "y2": 10.0},
        }],
        now=0.5,
    )
    tracker.update([], now=1.0)
    reacquired = tracker.update([detection], now=1.9)
    expired = tracker.update([detection], now=3.5)

    assert first[0]["track_id"] == moved[0]["track_id"] == reacquired[0]["track_id"]
    assert expired[0]["track_id"] != reacquired[0]["track_id"]


def test_vehicle_tracker_filters_nonvehicles_and_keeps_classes_distinct():
    tracker = VehicleTracker()
    detections = [
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.9,
            "bbox": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
        },
        {
            "class_id": 3,
            "class_name": "motorcycle",
            "confidence": 0.9,
            "bbox": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
        },
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.9,
            "bbox": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
        },
    ]

    tracked = tracker.update(detections, now=0.0)
    reordered = tracker.update([detections[1], detections[0]], now=0.5)

    assert len(tracked) == 2
    assert tracked[0]["vehicle_class"] == "4W"
    assert tracked[0]["class_name"] == "car"
    assert tracked[1]["vehicle_class"] == "2W"
    assert reordered[0]["track_id"] == tracked[1]["track_id"]
    assert reordered[1]["track_id"] == tracked[0]["track_id"]
