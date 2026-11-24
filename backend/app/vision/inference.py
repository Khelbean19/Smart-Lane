from __future__ import annotations

import logging
import threading
import time
from typing import Any, Protocol

import numpy as np


logger = logging.getLogger(__name__)


class FrameCamera(Protocol):
    def next_detection_frame(
        self,
        after_sequence: int,
        timeout: float = 1.0,
    ) -> tuple[int, np.ndarray] | None:
        ...


class FrameDetector(Protocol):
    def predict(self, frame: np.ndarray) -> list[dict[str, Any]]:
        ...

    def status(self) -> dict[str, Any]:
        ...


class CameraInferenceRunner:
    def __init__(
        self,
        detector: FrameDetector,
        inference_fps: float = 10.0,
    ) -> None:
        self.detector = detector
        self.inference_fps = inference_fps
        self._latest: dict[int, dict[str, Any]] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._events: dict[int, threading.Event] = {}
        self._lock = threading.RLock()

    def attach_camera(self, camera_id: int, camera: FrameCamera) -> None:
        with self._lock:
            existing = self._threads.get(camera_id)
            if existing is not None and existing.is_alive():
                return

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run,
                args=(camera_id, camera, stop_event),
                name=f"inference-camera-{camera_id}",
                daemon=True,
            )
            self._events[camera_id] = stop_event
            self._threads[camera_id] = thread
            self._latest[camera_id] = {
                "camera_id": camera_id,
                "frame_sequence": None,
                "timestamp": None,
                "status": "running",
                "detections": [],
                "error": None,
            }
            thread.start()

    def sync(self, camera_ids: set[int], cameras: dict[int, FrameCamera]) -> None:
        with self._lock:
            removed = set(self._threads) - camera_ids
        for camera_id in removed:
            self.stop(camera_id)

        for camera_id in camera_ids:
            camera = cameras.get(camera_id)
            if camera is not None:
                self.attach_camera(camera_id, camera)

    def stop(self, camera_id: int) -> None:
        with self._lock:
            event = self._events.pop(camera_id, None)
            thread = self._threads.pop(camera_id, None)
            self._latest.pop(camera_id, None)
        if event is not None:
            event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def stop_all(self) -> None:
        with self._lock:
            camera_ids = list(self._threads)
        for camera_id in camera_ids:
            self.stop(camera_id)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "detector": self.detector.status(),
                "active_camera_ids": sorted(self._threads),
                "camera_results": {
                    str(camera_id): dict(result)
                    for camera_id, result in self._latest.items()
                },
            }

    def latest(self, camera_id: int) -> dict[str, Any] | None:
        with self._lock:
            result = self._latest.get(camera_id)
            return dict(result) if result is not None else None

    def _run(
        self,
        camera_id: int,
        camera: FrameCamera,
        stop_event: threading.Event,
    ) -> None:
        sequence = 0
        interval = 1.0 / max(self.inference_fps, 0.1)
        next_allowed = 0.0

        while not stop_event.is_set():
            try:
                frame_result = camera.next_detection_frame(sequence, timeout=0.5)
                if frame_result is None:
                    continue

                frame_sequence, frame = frame_result
                sequence = frame_sequence
                now = time.monotonic()
                if now < next_allowed:
                    continue
                next_allowed = now + interval

                detections = self.detector.predict(frame)
                if stop_event.is_set():
                    return
                with self._lock:
                    self._latest[camera_id] = {
                        "camera_id": camera_id,
                        "frame_sequence": frame_sequence,
                        "timestamp": time.time(),
                        "status": "ok",
                        "detections": detections,
                        "error": None,
                    }
            except Exception as error:
                if stop_event.is_set():
                    return
                logger.exception("Inference failed for camera %s", camera_id)
                with self._lock:
                    self._latest[camera_id] = {
                        "camera_id": camera_id,
                        "frame_sequence": sequence,
                        "timestamp": time.time(),
                        "status": "error",
                        "detections": [],
                        "error": str(error),
                    }
