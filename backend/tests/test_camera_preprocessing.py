import numpy as np
import pytest

from app.vision.pipeline import (
    cap_frame,
    get_global_frame_cap,
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
