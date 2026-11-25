import time

import numpy as np
import pytest

from app.vision import camera as camera_module
from app.vision.camera import CameraCapture
from app.vision.pipeline import (
    cap_frame,
    get_global_frame_cap,
    map_detection_bbox_to_preview,
    preprocess_detection_frame,
    validate_preprocessing_settings,
)


def test_global_frame_cap_uses_documented_defaults_and_validates_environment(monkeypatch):
    monkeypatch.delenv("VISION_MAX_FRAME_WIDTH", raising=False)
    monkeypatch.delenv("VISION_MAX_FRAME_HEIGHT", raising=False)
    assert get_global_frame_cap() == (1280, 720)

    monkeypatch.setenv("VISION_MAX_FRAME_WIDTH", "1920")
    monkeypatch.setenv("VISION_MAX_FRAME_HEIGHT", "1080")
    assert get_global_frame_cap() == (1920, 1080)

    monkeypatch.setenv("VISION_MAX_FRAME_WIDTH", "wide")
    with pytest.raises(RuntimeError, match="whole-number"):
        get_global_frame_cap()


def test_cap_frame_limits_dimensions_without_changing_aspect_ratio():
    frame = np.full((2160, 3840, 3), 25, dtype=np.uint8)

    capped, source_resolution = cap_frame(frame, 1280, 720)

    assert source_resolution == (3840, 2160)
    assert capped.shape == (720, 1280, 3)


def test_fit_preprocessing_letterboxes_full_frame_without_distortion():
    frame = np.full((720, 1280, 3), 20, dtype=np.uint8)
    settings = {
        "detection_width": 640,
        "detection_height": 480,
        "preprocessing_mode": "fit",
        "roi_x": 0.0,
        "roi_y": 0.0,
        "roi_width": 1.0,
        "roi_height": 1.0,
    }

    result = preprocess_detection_frame(frame, settings)

    assert result.shape == (480, 640, 3)
    assert np.all(result[:60] == 114)
    assert np.all(result[60:420] == 20)
    assert np.all(result[420:] == 114)


def test_crop_preprocessing_applies_normalized_roi_then_letterboxes():
    frame = np.full((720, 1280, 3), 20, dtype=np.uint8)
    frame[:, :320] = 40
    settings = {
        "detection_width": 640,
        "detection_height": 480,
        "preprocessing_mode": "crop",
        "roi_x": 0.25,
        "roi_y": 0.1,
        "roi_width": 0.5,
        "roi_height": 0.8,
    }

    result = preprocess_detection_frame(frame, settings)

    assert result.shape == (480, 640, 3)
    assert np.all(result[:, :53] == 114)
    assert np.all(result[:, 53:586] == 20)
    assert np.all(result[:, 587:] == 114)


def test_fit_detection_box_maps_back_through_letterbox_padding():
    settings = {
        "detection_width": 640,
        "detection_height": 480,
        "preprocessing_mode": "fit",
        "roi_x": 0.0,
        "roi_y": 0.0,
        "roi_width": 1.0,
        "roi_height": 1.0,
    }

    mapped = map_detection_bbox_to_preview(
        {"x1": 100, "y1": 110, "x2": 300, "y2": 210},
        preview_width=1280,
        preview_height=720,
        settings=settings,
    )

    assert mapped == pytest.approx(
        {"x1": 0.15625, "y1": 100 / 720, "x2": 0.46875, "y2": 300 / 720}
    )
    assert map_detection_bbox_to_preview(
        {"x1": 10, "y1": 0, "x2": 50, "y2": 20},
        preview_width=1280,
        preview_height=720,
        settings=settings,
    ) is None


def test_crop_detection_box_maps_to_and_stays_within_roi():
    settings = {
        "detection_width": 640,
        "detection_height": 480,
        "preprocessing_mode": "crop",
        "roi_x": 0.2,
        "roi_y": 0.25,
        "roi_width": 0.5,
        "roi_height": 0.5,
    }

    mapped = map_detection_bbox_to_preview(
        {"x1": -10, "y1": 0, "x2": 650, "y2": 480},
        preview_width=1000,
        preview_height=800,
        settings=settings,
    )

    assert mapped == pytest.approx(
        {"x1": 0.2, "y1": 0.25, "x2": 0.7, "y2": 0.75}
    )


def test_preprocessing_validation_rejects_resolution_or_roi_overflow():
    settings = {
        "detection_width": 641,
        "detection_height": 480,
        "preprocessing_mode": "fit",
        "roi_x": 0.0,
        "roi_y": 0.0,
        "roi_width": 1.0,
        "roi_height": 1.0,
    }
    with pytest.raises(ValueError, match="global frame cap"):
        validate_preprocessing_settings(settings, 640, 480)

    settings["detection_width"] = 640
    settings["roi_x"] = 0.6
    with pytest.raises(ValueError, match="normalized crop region"):
        validate_preprocessing_settings(settings, 640, 480)


def test_camera_defers_detection_preprocessing_until_a_frame_is_requested(monkeypatch):
    class SlowCapture:
        def __init__(self, _source):
            self.sequence = 0

        def isOpened(self):
            return True

        def set(self, _property_id, _value):
            return True

        def read(self):
            time.sleep(0.005)
            self.sequence += 1
            return True, np.full((8, 8, 3), self.sequence, dtype=np.uint8)

        def release(self):
            pass

    processed_frames = []

    def record_preprocessing(frame, _settings):
        processed_frames.append(frame)
        return frame

    monkeypatch.setattr(
        camera_module,
        "preprocess_detection_frame",
        record_preprocessing,
    )
    camera = CameraCapture(
        1,
        "0",
        capture_factory=SlowCapture,
        max_frame_width=8,
        max_frame_height=8,
    )

    try:
        camera.start()
        deadline = time.monotonic() + 1
        while camera.status()["frame_sequence"] < 3:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert processed_frames == []

        result = camera.next_detection_frame(0, timeout=1)

        assert result is not None
        assert result[0] >= 3
        assert len(processed_frames) == 1
        assert processed_frames[0].shape == (8, 8, 3)
    finally:
        camera.stop()
