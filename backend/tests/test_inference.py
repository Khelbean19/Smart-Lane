import threading
import time

import numpy as np

from app.vision.inference import CameraInferenceRunner


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
    assert runner.status()["active_camera_ids"] == [7]

    runner.stop(7)
    assert runner.latest(7) is None
    assert runner.status()["active_camera_ids"] == []
