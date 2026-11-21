"""Managed camera capture primitives for the standalone vision worker."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from threading import Condition, Event, RLock, Thread
from typing import Any, Callable

import cv2

from .pipeline import (
    DEFAULT_PREPROCESSING_SETTINGS,
    cap_frame,
    get_global_frame_cap,
    preprocess_detection_frame,
    validate_preprocessing_settings,
)


logger = logging.getLogger(__name__)


class CameraUnavailable(RuntimeError):
    """Raised when a camera cannot be opened, configured, or read."""


@dataclass(frozen=True)
class CameraConfig:
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    jpeg_quality: int = 80
    detection_width: int = 640
    detection_height: int = 480
    preprocessing_mode: str = "fit"
    roi_x: float = 0.0
    roi_y: float = 0.0
    roi_width: float = 1.0
    roi_height: float = 1.0

    def preprocessing_settings(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in DEFAULT_PREPROCESSING_SETTINGS
        }


def normalize_source(source: str) -> int | str:
    value = source.strip()
    if not value:
        raise ValueError("Camera source cannot be empty.")
    if value.isdecimal():
        return int(value)
    return value


class CameraCapture:
    """Own one camera device and publish its latest JPEG frame."""

    def __init__(
        self,
        camera_id: int,
        source: int | str,
        capture_factory: Callable[[int | str], Any] | None = None,
        max_frame_width: int | None = None,
        max_frame_height: int | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self._capture_factory = capture_factory or cv2.VideoCapture
        if max_frame_width is None and max_frame_height is None:
            max_frame_width, max_frame_height = get_global_frame_cap()
        elif max_frame_width is None or max_frame_height is None:
            raise ValueError("Both maximum frame dimensions must be provided together.")
        self._max_frame_width = max_frame_width
        self._max_frame_height = max_frame_height
        self._condition = Condition(RLock())
        self._operation_lock = RLock()
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._ready_event = Event()
        self._config = CameraConfig()
        self._state = "stopped"
        self._error: str | None = None
        self._jpeg: bytes | None = None
        self._detection_frame: Any | None = None
        self._source_resolution: dict[str, int] | None = None
        self._preview_resolution: dict[str, int] | None = None
        self._sequence = 0

    def start(self, timeout: float = 10.0) -> dict[str, Any]:
        with self._operation_lock:
            with self._condition:
                if self._state == "running":
                    return self.status()
                if self._thread and self._thread.is_alive():
                    raise CameraUnavailable("Camera capture is already starting.")
                self._stop_event = Event()
                self._ready_event = Event()
                self._jpeg = None
                self._detection_frame = None
                self._source_resolution = None
                self._preview_resolution = None
                self._sequence = 0
                self._error = None
                self._state = "starting"
                self._thread = Thread(
                    target=self._capture_loop,
                    name=f"camera-{self.camera_id}",
                    daemon=True,
                )
                thread = self._thread
                thread.start()

            if not self._ready_event.wait(timeout):
                self._stop_event.set()
                thread.join(timeout=1.0)
                message = "Timed out waiting for the camera to produce a frame."
                with self._condition:
                    self._state = "error"
                    self._error = message
                    self._condition.notify_all()
                raise CameraUnavailable(message)

            with self._condition:
                if self._state != "running":
                    raise CameraUnavailable(self._error or "Camera did not start.")
                return self.status()

    def stop(self, timeout: float = 5.0) -> dict[str, Any]:
        with self._operation_lock:
            with self._condition:
                thread = self._thread
                self._stop_event.set()
                self._condition.notify_all()

            if thread and thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    message = "Camera capture did not stop before the timeout."
                    with self._condition:
                        self._state = "error"
                        self._error = message
                    raise CameraUnavailable(message)

            with self._condition:
                self._thread = None
                self._state = "stopped"
                self._error = None
                self._jpeg = None
                self._condition.notify_all()
                return self.status()

    def configure(self, updates: dict[str, Any]) -> dict[str, Any]:
        with self._operation_lock:
            config = replace(self._config, **updates)
            validate_preprocessing_settings(
                config.preprocessing_settings(),
                self._max_frame_width,
                self._max_frame_height,
            )
            was_running = self.status()["state"] == "running"
            if was_running:
                self.stop()

            with self._condition:
                self._config = config
                self._error = None

            if was_running:
                return self.start()
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "camera_id": self.camera_id,
                "source": str(self.source),
                "state": self._state,
                "config": asdict(self._config),
                "max_frame_resolution": {
                    "width": self._max_frame_width,
                    "height": self._max_frame_height,
                },
                "source_resolution": self._source_resolution,
                "preview_resolution": self._preview_resolution,
                "frame_sequence": self._sequence,
                "error": self._error,
            }

    def next_frame(self, after_sequence: int, timeout: float = 1.0) -> tuple[int, bytes] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._state != "running",
                timeout=timeout,
            )
            if self._sequence <= after_sequence or self._jpeg is None:
                return None
            return self._sequence, self._jpeg

    def next_detection_frame(
        self,
        after_sequence: int,
        timeout: float = 1.0,
    ) -> tuple[int, Any] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._state != "running",
                timeout=timeout,
            )
            if self._sequence <= after_sequence or self._detection_frame is None:
                return None
            return self._sequence, self._detection_frame.copy()

    def _capture_loop(self) -> None:
        capture = None
        try:
            capture = self._capture_factory(self.source)
            if not capture.isOpened():
                raise CameraUnavailable(f"Camera source {self.source} could not be opened.")

            self._apply_config(capture)
            while not self._stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    if self._stop_event.is_set():
                        break
                    raise CameraUnavailable(f"Camera source {self.source} failed to read a frame.")

                source_height, source_width = frame.shape[:2]
                capped_frame, _ = cap_frame(
                    frame,
                    self._max_frame_width,
                    self._max_frame_height,
                )
                detection_frame = preprocess_detection_frame(
                    capped_frame,
                    self._config.preprocessing_settings(),
                )
                encoded, buffer = cv2.imencode(
                    ".jpg",
                    capped_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality],
                )
                if not encoded:
                    raise CameraUnavailable("OpenCV could not encode a camera frame.")

                with self._condition:
                    if self._stop_event.is_set():
                        break
                    self._jpeg = buffer.tobytes()
                    self._detection_frame = detection_frame
                    self._source_resolution = {
                        "width": source_width,
                        "height": source_height,
                    }
                    preview_height, preview_width = capped_frame.shape[:2]
                    self._preview_resolution = {
                        "width": preview_width,
                        "height": preview_height,
                    }
                    self._sequence += 1
                    self._state = "running"
                    self._error = None
                    self._condition.notify_all()
                self._ready_event.set()
        except Exception as error:
            logger.exception("Camera %s capture failed", self.camera_id)
            with self._condition:
                if not self._stop_event.is_set():
                    self._state = "error"
                    self._error = str(error)
                    self._condition.notify_all()
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    logger.exception("Could not release camera %s", self.camera_id)
            self._ready_event.set()
            with self._condition:
                if self._stop_event.is_set() and self._state != "error":
                    self._state = "stopped"
                self._condition.notify_all()

    def _apply_config(self, capture: Any) -> None:
        requested_width = min(
            self._config.width or self._max_frame_width,
            self._max_frame_width,
        )
        requested_height = min(
            self._config.height or self._max_frame_height,
            self._max_frame_height,
        )
        properties = (
            (requested_width, cv2.CAP_PROP_FRAME_WIDTH, "width"),
            (requested_height, cv2.CAP_PROP_FRAME_HEIGHT, "height"),
            (self._config.fps, cv2.CAP_PROP_FPS, "frame rate"),
        )
        for value, property_id, name in properties:
            if value is not None and not capture.set(property_id, value):
                logger.warning(
                    "Camera %s did not accept the requested %s; software frame "
                    "capping will still be applied.",
                    self.camera_id,
                    name,
                )


class CameraManager:
    """Reconcile a database-provided camera registry with active camera IDs."""

    def __init__(
        self,
        capture_factory: Callable[[int | str], Any] | None = None,
        max_frame_width: int | None = None,
        max_frame_height: int | None = None,
    ) -> None:
        self._capture_factory = capture_factory
        if max_frame_width is None and max_frame_height is None:
            max_frame_width, max_frame_height = get_global_frame_cap()
        elif max_frame_width is None or max_frame_height is None:
            raise ValueError("Both maximum frame dimensions must be provided together.")
        self._max_frame_width = max_frame_width
        self._max_frame_height = max_frame_height
        self._cameras: dict[int, CameraCapture] = {}
        self._active_camera_ids: set[int] = set()
        self._registry_lock = RLock()

    def reconcile(
        self,
        registry: list[dict[str, Any]],
        active_camera_ids: list[int],
        retry_failed: bool = False,
    ) -> list[dict[str, Any]]:
        requested: dict[int, tuple[int | str, dict[str, Any]]] = {}
        for record in registry:
            camera_id = record["camera_id"]
            if camera_id in requested:
                raise ValueError(f"Duplicate camera ID {camera_id} in registry.")
            settings = {
                name: record.get(name, default)
                for name, default in DEFAULT_PREPROCESSING_SETTINGS.items()
            }
            validate_preprocessing_settings(
                settings,
                self._max_frame_width,
                self._max_frame_height,
            )
            requested[camera_id] = (normalize_source(record["source"]), settings)

        active = set(active_camera_ids)
        if len(active) != len(active_camera_ids):
            raise ValueError("A camera cannot be active more than once.")
        if not active.issubset(requested):
            raise ValueError("Active camera IDs must exist in the worker registry.")
        active_sources = [requested[camera_id][0] for camera_id in active]
        if len(set(active_sources)) != len(active_sources):
            raise ValueError("A camera source cannot be active more than once.")

        with self._registry_lock:
            previous_active = self._active_camera_ids
            changed_sources: set[int] = set()
            changed_settings: set[int] = set()
            blocked_ids: set[int] = set()

            for camera_id, camera in list(self._cameras.items()):
                if (
                    camera_id not in requested
                    or camera.source != requested[camera_id][0]
                ):
                    if camera.status()["state"] != "stopped":
                        try:
                            camera.stop()
                        except CameraUnavailable as error:
                            logger.error(
                                "Camera %s did not stop before registry update: %s",
                                camera_id,
                                error,
                            )
                            blocked_ids.add(camera_id)
                            continue
                    self._cameras.pop(camera_id)
                    changed_sources.add(camera_id)

            for camera_id, source in requested.items():
                source = source[0]
                if camera_id in blocked_ids:
                    continue
                if camera_id not in self._cameras:
                    self._cameras[camera_id] = CameraCapture(
                        camera_id,
                        source,
                        self._capture_factory,
                        self._max_frame_width,
                        self._max_frame_height,
                    )
                    changed_sources.add(camera_id)

            for camera_id, camera in self._cameras.items():
                if camera_id in blocked_ids:
                    continue
                desired_settings = requested[camera_id][1]
                current_settings = camera.status()["config"]
                if any(
                    current_settings.get(name) != value
                    for name, value in desired_settings.items()
                ):
                    changed_settings.add(camera_id)
                    try:
                        camera.configure(desired_settings)
                    except CameraUnavailable as error:
                        logger.error(
                            "Camera %s settings could not be applied: %s",
                            camera_id,
                            error,
                        )
                state = camera.status()["state"]
                if camera_id in active:
                    should_retry = (
                        retry_failed
                        or camera_id not in previous_active
                        or camera_id in changed_sources
                        or camera_id in changed_settings
                    )
                    if state == "running" or (state == "error" and not should_retry):
                        continue
                    try:
                        camera.start()
                    except CameraUnavailable as error:
                        logger.error("Camera %s did not start: %s", camera_id, error)
                elif state != "stopped":
                    try:
                        camera.stop()
                    except CameraUnavailable as error:
                        logger.error("Camera %s did not stop: %s", camera_id, error)

            self._active_camera_ids = active
            return self.statuses()

    def get(self, camera_id: int) -> CameraCapture:
        with self._registry_lock:
            try:
                return self._cameras[camera_id]
            except KeyError as error:
                raise KeyError(f"Unknown camera ID {camera_id}.") from error

    def statuses(self) -> list[dict[str, Any]]:
        with self._registry_lock:
            return [camera.status() for camera in self._cameras.values()]

    def stop_all(self) -> None:
        with self._registry_lock:
            errors: list[str] = []
            for camera in self._cameras.values():
                try:
                    camera.stop()
                except CameraUnavailable as error:
                    errors.append(str(error))
            self._active_camera_ids.clear()
            if errors:
                raise CameraUnavailable("; ".join(errors))
