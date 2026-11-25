from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


logger = logging.getLogger(__name__)

VEHICLE_CLASS_GROUPS = {
    "2w": "2W",
    "two wheeler": "2W",
    "bicycle": "2W",
    "motorcycle": "2W",
    "motorbike": "2W",
    "moped": "2W",
    "scooter": "2W",
    "tricycle": "2W",
    "motorized tricycle": "2W",
    "4w": "4W",
    "four wheeler": "4W",
    "car": "4W",
    "van": "4W",
    "bus": "4W",
    "truck": "4W",
    "pickup": "4W",
    "pickup truck": "4W",
    "jeepney": "4W",
    "emg": "EMG",
    "emergency": "EMG",
    "emergency vehicle": "EMG",
    "ambulance": "EMG",
    "fire truck": "EMG",
    "fire engine": "EMG",
    "police car": "EMG",
}


def vehicle_class_group(class_name: str) -> str | None:
    normalized_name = " ".join(
        class_name.casefold().replace("_", " ").replace("-", " ").split()
    )
    return VEHICLE_CLASS_GROUPS.get(normalized_name)


@dataclass
class _Track:
    track_id: int
    class_id: int
    bbox: dict[str, float]
    last_seen: float


def _bbox_iou(left: dict[str, float], right: dict[str, float]) -> float:
    intersection_width = max(
        0.0,
        min(left["x2"], right["x2"]) - max(left["x1"], right["x1"]),
    )
    intersection_height = max(
        0.0,
        min(left["y2"], right["y2"]) - max(left["y1"], right["y1"]),
    )
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left["x2"] - left["x1"]) * max(
        0.0,
        left["y2"] - left["y1"],
    )
    right_area = max(0.0, right["x2"] - right["x1"]) * max(
        0.0,
        right["y2"] - right["y1"],
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


class VehicleTracker:
    def __init__(
        self,
        *,
        max_age_seconds: float = 1.5,
        iou_threshold: float = 0.3,
    ) -> None:
        if max_age_seconds <= 0:
            raise ValueError("Track retention must be positive.")
        if not 0 <= iou_threshold <= 1:
            raise ValueError("Tracker IoU threshold must be between 0 and 1.")
        self.max_age_seconds = max_age_seconds
        self.iou_threshold = iou_threshold
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1

    def update(
        self,
        detections: list[dict[str, Any]],
        *,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        timestamp = time.monotonic() if now is None else now
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if timestamp - track.last_seen <= self.max_age_seconds
        }

        vehicle_detections = [
            (detection, vehicle_class_group(detection["class_name"]))
            for detection in detections
        ]
        vehicle_detections = [
            (detection, vehicle_class)
            for detection, vehicle_class in vehicle_detections
            if vehicle_class is not None
        ]

        candidates = [
            (
                _bbox_iou(track.bbox, detection["bbox"]),
                track_id,
                detection_index,
            )
            for track_id, track in self._tracks.items()
            for detection_index, (detection, _) in enumerate(vehicle_detections)
            if track.class_id == detection["class_id"]
        ]
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_detections: dict[int, int] = {}
        for overlap, track_id, detection_index in candidates:
            if overlap < self.iou_threshold:
                break
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections[detection_index] = track_id

        tracked_detections: list[dict[str, Any]] = []
        for detection_index, (detection, vehicle_class) in enumerate(vehicle_detections):
            track_id = matched_detections.get(detection_index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
            self._tracks[track_id] = _Track(
                track_id=track_id,
                class_id=detection["class_id"],
                bbox=dict(detection["bbox"]),
                last_seen=timestamp,
            )
            tracked_detections.append(
                {
                    **detection,
                    "vehicle_class": vehicle_class,
                    "track_id": track_id,
                }
            )

        return tracked_detections

    def reset(self) -> None:
        self._tracks.clear()


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


@dataclass
class _CameraInferenceState:
    camera: FrameCamera
    tracker: VehicleTracker
    sequence: int
    next_allowed: float
    completed_at: deque[float]
    result: dict[str, Any]
    inference_duration_ms: float = 0.0
    inferences_completed: int = 0


class CameraInferenceRunner:
    _POLL_INTERVAL = 0.01
    _RATE_WINDOW = 30

    def __init__(
        self,
        detector: FrameDetector,
        inference_fps: float = 10.0,
    ) -> None:
        self.detector = detector
        self.inference_fps = inference_fps
        self._cameras: dict[int, _CameraInferenceState] = {}
        self._completed_at: deque[float] = deque(maxlen=self._RATE_WINDOW)
        self._inferences_completed = 0
        self._last_served_camera_id: int | None = None
        self._condition = threading.Condition(threading.RLock())
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def attach_camera(self, camera_id: int, camera: FrameCamera) -> None:
        with self._condition:
            existing = self._cameras.get(camera_id)
            if existing is not None and existing.camera is camera:
                return

            self._cameras[camera_id] = self._new_camera_state(camera_id, camera)
            self._ensure_scheduler_locked()
            self._condition.notify_all()

    def sync(
        self,
        camera_ids: set[int],
        cameras: dict[int, FrameCamera],
        restart_camera_ids: set[int] | None = None,
    ) -> None:
        restart_ids = (restart_camera_ids or set()) & camera_ids
        with self._condition:
            removed = (set(self._cameras) - camera_ids) | restart_ids
            for camera_id in removed:
                self._cameras.pop(camera_id, None)
            self._condition.notify_all()

        for camera_id in sorted(camera_ids):
            camera = cameras.get(camera_id)
            if camera is not None:
                self.attach_camera(camera_id, camera)

    def restart_camera(self, camera_id: int, camera: FrameCamera) -> None:
        self.sync({camera_id}, {camera_id: camera}, restart_camera_ids={camera_id})

    def stop(self, camera_id: int) -> None:
        with self._condition:
            self._cameras.pop(camera_id, None)
            self._condition.notify_all()

    def stop_all(self) -> None:
        with self._condition:
            thread = self._thread
            stop_event = self._stop_event
            self._cameras.clear()
            if stop_event is not None:
                stop_event.set()
            self._condition.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._condition:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
                self._stop_event = None

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "detector": self.detector.status(),
                "active_camera_ids": sorted(self._cameras),
                "inference_fps": self.inference_fps,
                "actual_inference_fps": self._rate(self._completed_at),
                "inferences_completed": self._inferences_completed,
                "camera_results": {
                    str(camera_id): {
                        **state.result,
                        "actual_inference_fps": self._rate(state.completed_at),
                        "inference_duration_ms": state.inference_duration_ms,
                        "inferences_completed": state.inferences_completed,
                        "detections": [
                            dict(detection)
                            for detection in state.result["detections"]
                        ],
                    }
                    for camera_id, state in self._cameras.items()
                },
            }

    def latest(self, camera_id: int) -> dict[str, Any] | None:
        with self._condition:
            state = self._cameras.get(camera_id)
            if state is None:
                return None
            return {
                **state.result,
                "actual_inference_fps": self._rate(state.completed_at),
                "inference_duration_ms": state.inference_duration_ms,
                "inferences_completed": state.inferences_completed,
                "detections": [
                    dict(detection) for detection in state.result["detections"]
                ],
            }

    def _new_camera_state(
        self,
        camera_id: int,
        camera: FrameCamera,
    ) -> _CameraInferenceState:
        return _CameraInferenceState(
            camera=camera,
            tracker=VehicleTracker(),
            sequence=0,
            next_allowed=0.0,
            completed_at=deque(maxlen=self._RATE_WINDOW),
            result={
                "camera_id": camera_id,
                "frame_sequence": None,
                "timestamp": None,
                "status": "running",
                "detections": [],
                "error": None,
                "inference_fps": self.inference_fps,
                "actual_inference_fps": 0.0,
                "inference_duration_ms": 0.0,
                "inferences_completed": 0,
            },
        )

    def _ensure_scheduler_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run,
            args=(stop_event,),
            name="camera-inference-scheduler",
            daemon=True,
        )
        self._stop_event = stop_event
        self._thread = thread
        thread.start()

    def _run(self, stop_event: threading.Event) -> None:
        interval = 1.0 / max(self.inference_fps, 0.1)
        while not stop_event.is_set():
            task: tuple[int, _CameraInferenceState, int, np.ndarray] | None = None
            with self._condition:
                camera_ids = sorted(self._cameras)
                if not camera_ids:
                    if self._thread is threading.current_thread():
                        self._thread = None
                        self._stop_event = None
                    return
                now = time.monotonic()
                if self._last_served_camera_id in camera_ids:
                    pivot = camera_ids.index(self._last_served_camera_id) + 1
                    camera_ids = camera_ids[pivot:] + camera_ids[:pivot]

                due_cameras = [
                    (camera_id, self._cameras[camera_id])
                    for camera_id in camera_ids
                    if self._cameras[camera_id].next_allowed <= now
                ]
                next_due = min(
                    (
                        max(0.0, state.next_allowed - now)
                        for state in self._cameras.values()
                    ),
                    default=self._POLL_INTERVAL,
                )

            for camera_id, state in due_cameras:
                try:
                    frame_result = state.camera.next_detection_frame(
                        state.sequence,
                        timeout=0.0,
                    )
                except Exception as error:
                    with self._condition:
                        if self._cameras.get(camera_id) is state:
                            state.next_allowed = time.monotonic() + interval
                    self._record_error(camera_id, state, error, state.sequence)
                    continue
                if frame_result is None:
                    continue

                frame_sequence, frame = frame_result
                with self._condition:
                    if self._cameras.get(camera_id) is not state:
                        continue
                    state.sequence = frame_sequence
                    state.next_allowed = time.monotonic() + interval
                    self._last_served_camera_id = camera_id
                task = (camera_id, state, frame_sequence, frame)
                break

            if task is None:
                with self._condition:
                    if stop_event.is_set():
                        return
                    wait_time = (
                        min(self._POLL_INTERVAL, next_due)
                        if next_due > 0
                        else self._POLL_INTERVAL
                    )
                    self._condition.wait(timeout=wait_time)
                continue

            camera_id, state, frame_sequence, frame = task
            started = time.monotonic()
            try:
                detections = self.detector.predict(frame)
            except Exception as error:
                if stop_event.is_set():
                    return
                self._record_error(camera_id, state, error, frame_sequence)
                continue

            completed = time.monotonic()
            if stop_event.is_set():
                return
            with self._condition:
                if self._cameras.get(camera_id) is not state:
                    continue
                state.completed_at.append(completed)
                self._completed_at.append(completed)
                state.inferences_completed += 1
                self._inferences_completed += 1
                state.inference_duration_ms = (completed - started) * 1000
                state.result = {
                    "camera_id": camera_id,
                    "frame_sequence": frame_sequence,
                    "timestamp": time.time(),
                    "status": "ok",
                    "detections": state.tracker.update(detections),
                    "error": None,
                    "inference_fps": self.inference_fps,
                    "actual_inference_fps": self._rate(state.completed_at),
                    "inference_duration_ms": state.inference_duration_ms,
                    "inferences_completed": state.inferences_completed,
                }

    def _record_error(
        self,
        camera_id: int,
        state: _CameraInferenceState,
        error: Exception,
        frame_sequence: int,
    ) -> None:
        logger.exception("Inference failed for camera %s", camera_id)
        with self._condition:
            if self._cameras.get(camera_id) is not state:
                return
            state.result = {
                "camera_id": camera_id,
                "frame_sequence": frame_sequence,
                "timestamp": time.time(),
                "status": "error",
                "detections": [],
                "error": str(error),
                "inference_fps": self.inference_fps,
                "actual_inference_fps": self._rate(state.completed_at),
                "inference_duration_ms": state.inference_duration_ms,
                "inferences_completed": state.inferences_completed,
            }

    @staticmethod
    def _rate(completed_at: deque[float]) -> float:
        if len(completed_at) < 2:
            return 0.0
        elapsed = completed_at[-1] - completed_at[0]
        return (len(completed_at) - 1) / elapsed if elapsed > 0 else 0.0
