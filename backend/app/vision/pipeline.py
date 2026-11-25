"""Frame sizing and per-camera inference preprocessing."""

from __future__ import annotations

import math
import os
from typing import Any

import cv2
import numpy as np


DEFAULT_MAX_FRAME_WIDTH = 1280
DEFAULT_MAX_FRAME_HEIGHT = 720
MAX_FRAME_WIDTH = 7680
MAX_FRAME_HEIGHT = 4320
DEFAULT_PREPROCESSING_SETTINGS: dict[str, Any] = {
    "detection_width": 640,
    "detection_height": 480,
    "preprocessing_mode": "fit",
    "roi_x": 0.0,
    "roi_y": 0.0,
    "roi_width": 1.0,
    "roi_height": 1.0,
}


def _read_positive_env(name: str, default: int, maximum: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a whole-number resolution.") from error
    if parsed <= 0 or parsed > maximum:
        raise RuntimeError(f"{name} must be between 1 and {maximum}.")
    return parsed


def get_global_frame_cap() -> tuple[int, int]:
    """Return configured maximum frame dimensions, with conservative defaults."""
    return (
        _read_positive_env(
            "VISION_MAX_FRAME_WIDTH",
            DEFAULT_MAX_FRAME_WIDTH,
            MAX_FRAME_WIDTH,
        ),
        _read_positive_env(
            "VISION_MAX_FRAME_HEIGHT",
            DEFAULT_MAX_FRAME_HEIGHT,
            MAX_FRAME_HEIGHT,
        ),
    )


def validate_preprocessing_settings(
    settings: dict[str, Any],
    max_width: int,
    max_height: int,
) -> None:
    detection_width = settings["detection_width"]
    detection_height = settings["detection_height"]
    if not 1 <= detection_width <= max_width:
        raise ValueError("Detection width must be within the global frame cap.")
    if not 1 <= detection_height <= max_height:
        raise ValueError("Detection height must be within the global frame cap.")
    if settings["preprocessing_mode"] not in {"fit", "crop"}:
        raise ValueError("Preprocessing mode must be 'fit' or 'crop'.")

    x = settings["roi_x"]
    y = settings["roi_y"]
    width = settings["roi_width"]
    height = settings["roi_height"]
    if (
        not all(0 <= value <= 1 for value in (x, y, width, height))
        or width <= 0
        or height <= 0
        or x + width > 1
        or y + height > 1
    ):
        raise ValueError("The normalized crop region must lie within the frame.")


def cap_frame(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Downscale a frame to fit the global cap without changing its aspect ratio."""
    source_height, source_width = frame.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Camera returned an empty frame.")

    scale = min(max_width / source_width, max_height / source_height, 1.0)
    if scale == 1:
        return frame, (source_width, source_height)

    output_width = max(1, min(max_width, round(source_width * scale)))
    output_height = max(1, min(max_height, round(source_height * scale)))
    return (
        cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA),
        (source_width, source_height),
    )


def _letterbox_dimensions(
    source_width: int,
    source_height: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, round(source_width * scale)))
    resized_height = max(1, min(height, round(source_height * scale)))
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    return resized_width, resized_height, left, top


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    resized_width, resized_height, left, top = _letterbox_dimensions(
        source_width,
        source_height,
        width,
        height,
    )
    scale = min(width / source_width, height / source_height)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)

    if frame.ndim == 2:
        canvas = np.full((height, width), 114, dtype=frame.dtype)
    else:
        canvas = np.full((height, width, frame.shape[2]), 114, dtype=frame.dtype)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def map_detection_bbox_to_preview(
    bbox: dict[str, float],
    preview_width: int,
    preview_height: int,
    settings: dict[str, Any],
) -> dict[str, float] | None:
    """Map a detection-frame box to normalized coordinates in the preview frame."""
    detection_width = settings["detection_width"]
    detection_height = settings["detection_height"]
    if min(preview_width, preview_height, detection_width, detection_height) <= 0:
        raise ValueError("Preview and detection dimensions must be positive.")

    crop_left = 0
    crop_top = 0
    crop_right = preview_width
    crop_bottom = preview_height
    if settings["preprocessing_mode"] == "crop":
        crop_left = min(preview_width - 1, math.floor(settings["roi_x"] * preview_width))
        crop_top = min(preview_height - 1, math.floor(settings["roi_y"] * preview_height))
        crop_right = min(
            preview_width,
            max(
                crop_left + 1,
                math.ceil(
                    (settings["roi_x"] + settings["roi_width"]) * preview_width
                ),
            ),
        )
        crop_bottom = min(
            preview_height,
            max(
                crop_top + 1,
                math.ceil(
                    (settings["roi_y"] + settings["roi_height"]) * preview_height
                ),
            ),
        )

    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    resized_width, resized_height, pad_left, pad_top = _letterbox_dimensions(
        crop_width,
        crop_height,
        detection_width,
        detection_height,
    )

    box_left = max(pad_left, min(bbox["x1"], bbox["x2"]))
    box_top = max(pad_top, min(bbox["y1"], bbox["y2"]))
    box_right = min(pad_left + resized_width, max(bbox["x1"], bbox["x2"]))
    box_bottom = min(pad_top + resized_height, max(bbox["y1"], bbox["y2"]))
    if box_right <= box_left or box_bottom <= box_top:
        return None

    preview_left = crop_left + (box_left - pad_left) * crop_width / resized_width
    preview_top = crop_top + (box_top - pad_top) * crop_height / resized_height
    preview_right = crop_left + (box_right - pad_left) * crop_width / resized_width
    preview_bottom = crop_top + (box_bottom - pad_top) * crop_height / resized_height
    return {
        "x1": max(0.0, min(1.0, preview_left / preview_width)),
        "y1": max(0.0, min(1.0, preview_top / preview_height)),
        "x2": max(0.0, min(1.0, preview_right / preview_width)),
        "y2": max(0.0, min(1.0, preview_bottom / preview_height)),
    }


def preprocess_detection_frame(
    frame: np.ndarray,
    settings: dict[str, Any],
) -> np.ndarray:
    """Create a fixed-size, aspect-preserving model input from a capped frame."""
    if settings["preprocessing_mode"] == "crop":
        frame_height, frame_width = frame.shape[:2]
        x = settings["roi_x"]
        y = settings["roi_y"]
        width = settings["roi_width"]
        height = settings["roi_height"]
        left = min(frame_width - 1, math.floor(x * frame_width))
        top = min(frame_height - 1, math.floor(y * frame_height))
        right = min(frame_width, max(left + 1, math.ceil((x + width) * frame_width)))
        bottom = min(frame_height, max(top + 1, math.ceil((y + height) * frame_height)))
        frame = frame[top:bottom, left:right]

    return _letterbox(
        frame,
        settings["detection_width"],
        settings["detection_height"],
    )