"""Managed camera capture primitives for the standalone vision worker.

This module owns OpenCV capture lifecycles and separates preview frames from
detection frames. A registry is reconciled through :class:`CameraManager`;
individual captures are accessed through :class:`CameraCapture`.

Typical usage:

    from app.vision.camera import CameraManager

    manager = CameraManager()
    manager.reconcile(
        [{"camera_id": 1, "source": "0"}],
        active_camera_ids=[1],
    )
    camera = manager.get(1)

    frame = camera.next_frame(after_sequence=0, timeout=2)
    detection = camera.next_detection_frame(after_sequence=0, timeout=2)

    manager.stop_all()

Numeric source strings are treated as OpenCV device indexes (``"0"`` becomes
``0``); RTSP and file paths remain strings. Camera lifecycle failures use
:class:`CameraUnavailable`, while structural registry/configuration errors use
``ValueError``.
"""

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
    """Signal a camera lifecycle or capture failure.

    ``CameraUnavailable`` is raised by capture and manager operations that
    cannot open, start, configure, read, or stop a camera within their
    timeout or validation limits. Callers can catch this exception to report
    the failure as a service-unavailable response or as a camera status.

    Example:
        try:
            camera.start()
        except CameraUnavailable as error:
            print(f"Camera unavailable: {error}")
    """


@dataclass(frozen=True)
class CameraConfig:
    """Immutable settings used for preview and detection preprocessing.

    ``width`` and ``height`` request a capture resolution. ``fps`` requests
    the capture frame rate. ``jpeg_quality`` controls encoded preview JPEG
    quality. Detection dimensions and the ROI settings control the frame
    passed to the detection pipeline.

    Attributes:
        width: Optional requested capture width.
        height: Optional requested capture height.
        fps: Optional requested capture frame rate.
        jpeg_quality: JPEG quality from 1 through 100.
        detection_width: Detection-frame width after preprocessing.
        detection_height: Detection-frame height after preprocessing.
        preprocessing_mode: Preprocessing mode, such as ``"fit"``.
        roi_x: Horizontal ROI origin as a fraction from 0.0 to 1.0.
        roi_y: Vertical ROI origin as a fraction from 0.0 to 1.0.
        roi_width: ROI width as a fraction from 0.0 to 1.0.
        roi_height: ROI height as a fraction from 0.0 to 1.0.

    Example:
        config = CameraConfig(
            width=1280,
            height=720,
            fps=30,
            detection_width=640,
            detection_height=480,
        )
    """

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
        """Return the preprocessing fields in a pipeline-ready mapping.

        The returned dictionary contains only the keys consumed by
        :mod:`app.vision.pipeline`; it does not include capture-only fields
        such as width, height, FPS, or JPEG quality.

        Returns:
            A dictionary mapping preprocessing setting names to values.

        Example:
            settings = CameraConfig(detection_width=320).preprocessing_settings()
            validate_preprocessing_settings(settings, 1280, 720)
        """
        return {
            name: getattr(self, name)
            for name in DEFAULT_PREPROCESSING_SETTINGS
        }


def normalize_source(source: str) -> int | str:
    """Normalize a camera-source string for OpenCV.

    A string containing only decimal digits is converted to an integer so
    OpenCV treats it as a device index. All other values, including RTSP,
    file, and HTTP URLs, are returned unchanged after surrounding whitespace
    is removed.

    Args:
        source: Camera device index or capture URL/file path.

    Returns:
        An integer device index or stripped string source.

    Raises:
        ValueError: If ``source`` is empty or contains only whitespace.

    Example:
        normalize_source("0")       # 0
        normalize_source(" rtsp://camera.local/live ")
        # "rtsp://camera.local/live"
    """
    value = source.strip()
    if not value:
        raise ValueError("Camera source cannot be empty.")
    if value.isdecimal():
        return int(value)
    return value


class CameraCapture:
    """Own and run one camera capture in a background thread.

    Each instance publishes a frame-capped JPEG for preview consumers and the
    latest capped source frame for on-demand detection preprocessing.
    Construction does not open the camera;
    call :meth:`start` to begin capture.

    Args:
        camera_id: Stable identifier used in status output and thread names.
        source: Integer OpenCV device index or a string capture source.
        capture_factory: Optional callable that creates an OpenCV-compatible
            capture object. It is primarily useful for tests and custom
            captures; it defaults to ``cv2.VideoCapture``.
        max_frame_width: Maximum preview width. If omitted with
            ``max_frame_height``, the global pipeline frame cap is used.
        max_frame_height: Maximum preview height. Must be supplied together
            with ``max_frame_width``.

    Raises:
        ValueError: If only one maximum frame dimension is supplied.

    Example:
        camera = CameraCapture(camera_id=1, source="0")
        status = camera.start(timeout=10)
        frame = camera.next_frame(after_sequence=0, timeout=2)
        camera.stop()
    """

    def __init__(
        self,
        camera_id: int,
        source: int | str,
        capture_factory: Callable[[int | str], Any] | None = None,
        max_frame_width: int | None = None,
        max_frame_height: int | None = None,
    ) -> None:
        """Initialize capture state without opening the camera source.

        Args:
            camera_id: Stable identifier for status and logging.
            source: Integer device index or string capture source.
            capture_factory: Optional OpenCV-compatible capture constructor.
            max_frame_width: Optional preview-width limit.
            max_frame_height: Optional preview-height limit.

        Raises:
            ValueError: If only one maximum frame dimension is supplied.
        """
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
        """Start the capture thread and wait for its first frame.

        Calling ``start`` on an already-running camera returns current status
        and does not create another thread. The operation is serialized with
        stop/configuration operations.

        Args:
            timeout: Maximum seconds to wait for the first frame.

        Returns:
            The running camera status mapping.

        Raises:
            CameraUnavailable: If another start is in progress, the first
                frame times out, capture fails, or the camera is not running
                after startup.

        Example:
            camera = CameraCapture(1, "0")
            status = camera.start(timeout=15)
            if status["state"] == "running":
                print(status["preview_resolution"])
        """
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
        """Stop the capture thread and clear the latest preview frame.

        Args:
            timeout: Maximum seconds to wait for the capture thread to exit.

        Returns:
            The stopped camera status mapping.

        Raises:
            CameraUnavailable: If the capture thread remains alive after
                ``timeout``.

        Example:
            status = camera.stop(timeout=3)
            assert status["state"] == "stopped"
        """
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
        """Validate and apply one or more camera settings.

        Unknown keys raise ``TypeError`` through ``dataclasses.replace``.
        Preprocessing values are validated before the running capture is
        restarted. If the camera was running, it is stopped and then started
        again with the new configuration.

        Args:
            updates: Mapping of ``CameraConfig`` field names to new values.

        Returns:
            Updated camera status. A running camera will be running again
            before this method returns.

        Raises:
            TypeError: If ``updates`` contains an unknown configuration field.
            ValueError: If preprocessing dimensions or ROI values are invalid.
            CameraUnavailable: If an active capture cannot be stopped or
                restarted.

        Example:
            camera.configure({
                "width": 1280,
                "height": 720,
                "jpeg_quality": 90,
            })
        """
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
        """Return a thread-safe snapshot of the camera state.

        The snapshot includes the camera and source identifiers, lifecycle
        state, effective configuration, source/preview resolutions, latest
        frame sequence, and the latest error message.

        Returns:
            A dictionary suitable for API responses and logging.

        Example:
            snapshot = camera.status()
            print(snapshot["state"], snapshot["frame_sequence"])
        """
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
        """Wait for and return the latest preview JPEG.

        This long-poll-style method suppresses duplicate frames: pass the
        sequence returned by the previous call. If no newer frame arrives
        before ``timeout``, or if the camera has stopped, it returns ``None``.

        Args:
            after_sequence: Last sequence successfully consumed by the caller.
            timeout: Maximum seconds to wait for a newer frame.

        Returns:
            ``(sequence, jpeg_bytes)`` for a new frame, otherwise ``None``.

        Example:
            sequence = 0
            frame = camera.next_frame(sequence, timeout=2)
            if frame is not None:
                sequence, jpeg = frame
        """
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
        """Wait for the latest frame and preprocess it for detection.

        Capture retains the capped source frame and defers detection resizing
        until a consumer requests a newer frame. This avoids preprocessing
        captured frames that the inference scheduler will skip. The returned
        array is independent of the frame held by ``CameraCapture``.

        Args:
            after_sequence: Last sequence consumed by the caller.
            timeout: Maximum seconds to wait for a newer frame.

        Returns:
            ``(sequence, detection_frame)`` for a new frame, otherwise
            ``None``.

        Example:
            result = camera.next_detection_frame(0, timeout=2)
            if result is not None:
                sequence, detection_frame = result
                detections = detector.detect(detection_frame)
        """
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._state != "running",
                timeout=timeout,
            )
            if self._sequence <= after_sequence or self._detection_frame is None:
                return None
            sequence = self._sequence
            frame = self._detection_frame.copy()
            settings = self._config.preprocessing_settings()
        return sequence, preprocess_detection_frame(frame, settings)

    def _capture_loop(self) -> None:
        """Run the blocking OpenCV capture loop until stopped or failed.

        This private method is the target of the background thread. It opens
        the source, applies capture properties, caps frames, publishes JPEG and
        source-frame representations, and updates lifecycle state.
        Call :meth:`start` rather than invoking this method directly.
        """
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
                    self._detection_frame = capped_frame
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
        """Request configured capture width, height, and frame rate.

        Width and height are capped by the instance's maximum preview
        dimensions. OpenCV properties that a device rejects are logged as
        warnings because the software preview cap remains available.

        Args:
            capture: OpenCV-compatible object exposing ``set``.

        Note:
            This is an internal setup helper. Call :meth:`configure` to change
            settings safely while accounting for a running thread.
        """
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
    """Reconcile a camera registry with active and inactive devices.

    ``CameraManager`` creates ``CameraCapture`` instances only for registry
    entries and opens devices only when an entry is active. Use
    :meth:`reconcile` after receiving registry updates from the API or
    database, then :meth:`statuses` or :meth:`get` to inspect and consume
    cameras.

    Args:
        capture_factory: Optional capture constructor passed to new
            ``CameraCapture`` instances. It defaults to ``cv2.VideoCapture``.
        max_frame_width: Maximum preview width; omitted values use the global
            preprocessing frame cap.
        max_frame_height: Maximum preview height; omitted values use the
            global preprocessing frame cap.

    Raises:
        ValueError: If only one maximum frame dimension is supplied.

    Example:
        manager = CameraManager()
        statuses = manager.reconcile(
            [{"camera_id": 1, "source": "0"}],
            active_camera_ids=[1],
        )
        camera = manager.get(1)
    """

    def __init__(
        self,
        capture_factory: Callable[[int | str], Any] | None = None,
        max_frame_width: int | None = None,
        max_frame_height: int | None = None,
    ) -> None:
        """Initialize an empty registry manager without opening cameras.

        Args:
            capture_factory: Optional capture constructor passed to individual
                ``CameraCapture`` instances.
            max_frame_width: Optional preview-width limit.
            max_frame_height: Optional preview-height limit.

        Raises:
            ValueError: If only one maximum frame dimension is supplied.
        """
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
        """Apply a complete camera registry and desired active set.

        The operation validates every registry entry before changing the
        managed cameras. Removed sources are stopped and discarded, inactive
        cameras are stopped, active cameras are started, and changed settings
        are applied. Camera failures are represented in status output rather
        than raised for an individual camera; structural registry errors raise
        ``ValueError``.

        Args:
            registry: Complete list of records containing ``camera_id`` and
                ``source`` plus optional preprocessing settings.
            active_camera_ids: IDs that should be running. Duplicate IDs,
                unknown IDs, and duplicate active sources are rejected.
            retry_failed: If true, start an errored active camera even when
                its source and settings did not change.

        Returns:
            Status dictionaries for every managed camera, including inactive
            and errored entries.

        Raises:
            ValueError: For duplicate registry/active IDs, unknown active IDs,
                duplicate active sources, empty sources, or invalid
                preprocessing settings.

        Example:
            manager.reconcile(
                [
                    {
                        "camera_id": 10,
                        "source": "0",
                        "detection_width": 640,
                        "detection_height": 480,
                    },
                    {"camera_id": 22, "source": "rtsp://camera.local/live"},
                ],
                active_camera_ids=[10],
                retry_failed=True,
            )
        """
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
        """Get a managed camera by stable camera ID.

        Args:
            camera_id: ID present in a reconciled registry.

        Returns:
            The managed ``CameraCapture`` instance.

        Raises:
            KeyError: If the ID is not in the current registry.

        Example:
            camera = manager.get(10)
            status = camera.status()
        """
        with self._registry_lock:
            try:
                return self._cameras[camera_id]
            except KeyError as error:
                raise KeyError(f"Unknown camera ID {camera_id}.") from error

    def statuses(self) -> list[dict[str, Any]]:
        """Get status snapshots for every camera in the current registry.

        Returns:
            Status dictionaries ordered by the registry insertion order held
            by the manager.

        Example:
            for status in manager.statuses():
                print(status["camera_id"], status["state"])
        """
        with self._registry_lock:
            return [camera.status() for camera in self._cameras.values()]

    def stop_all(self) -> None:
        """Stop every managed camera and clear the active-camera set.

        Each camera is stopped even if another camera fails. If one or more
        stops fail, they are combined into one ``CameraUnavailable`` exception
        after all stop attempts have completed.

        Raises:
            CameraUnavailable: If at least one capture thread cannot stop.

        Example:
            try:
                manager.stop_all()
            except CameraUnavailable as error:
                logger.error("Camera shutdown failed: %s", error)
        """
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
