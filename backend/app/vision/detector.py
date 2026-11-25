from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any, TypedDict

import numpy as np
from loguru import logger
from ultralytics import YOLO


MODELS_DIR = Path(__file__).resolve().parents[3] / "backend" / "models_weights"


class DetectorConfig(TypedDict):
    model_path: str
    confidence: float
    iou: float
    image_size: int
    device: str
    classes: list[int] | None
    inference_fps: float


def _positive_float(name: str, default: float, maximum: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number.") from error
    if not 0 < parsed <= maximum:
        raise RuntimeError(f"{name} must be greater than 0 and at most {maximum}.")
    return parsed


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a whole number.") from error
    if parsed <= 0:
        raise RuntimeError(f"{name} must be greater than 0.")
    return parsed


def _unit_interval_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number.") from error
    if not 0 <= parsed <= 1:
        raise RuntimeError(f"{name} must be between 0 and 1.")
    return parsed


def get_detector_config() -> DetectorConfig:
    model_path_value = os.getenv("VISION_MODEL_PATH")
    model_path = (
        model_path_value.strip()
        if model_path_value and model_path_value.strip()
        else str(MODELS_DIR / "yolo26n.pt")
    )
    device_value = os.getenv("VISION_MODEL_DEVICE")
    device = device_value.strip() if device_value and device_value.strip() else "cpu"
    classes_value = os.getenv("VISION_MODEL_CLASSES")
    classes: list[int] | None = None
    if classes_value and classes_value.strip():
        class_values = [value.strip() for value in classes_value.split(",")]
        if any(not value for value in class_values):
            raise RuntimeError(
                "VISION_MODEL_CLASSES must be comma-separated class IDs."
            )
        try:
            classes = [int(value) for value in class_values]
        except ValueError as error:
            raise RuntimeError(
                "VISION_MODEL_CLASSES must be comma-separated class IDs."
            ) from error
        if any(class_id < 0 for class_id in classes):
            raise RuntimeError("VISION_MODEL_CLASSES cannot contain negative IDs.")
        if len(set(classes)) != len(classes):
            raise RuntimeError("VISION_MODEL_CLASSES cannot contain duplicate IDs.")

    return {
        "model_path": model_path,
        "confidence": _unit_interval_float("VISION_CONFIDENCE_THRESHOLD", 0.4),
        "iou": _unit_interval_float("VISION_IOU_THRESHOLD", 0.5),
        "image_size": _positive_int("VISION_MODEL_IMAGE_SIZE", 640),
        "device": device,
        "classes": classes,
        "inference_fps": _positive_float("VISION_INFERENCE_FPS", 10.0, 120.0),
    }


class YOLODetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.4,
        iou: float = 0.5,
        image_size: int = 640,
        device: str = "cpu",
        classes: list[int] | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device
        self.classes = classes
        self.model: YOLO | None = None  # initialized in warmup()
        self._lock = Lock()

    def warmup(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights were not found: {self.model_path}"
            )

        model = YOLO(str(self.model_path))
        frame = np.zeros(
            (self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )
        self.model = model
        logger.info(
            "Loaded YOLO model {} on {}",
            self.model_path,
            self.device,
        )
        logger.info(f"Model configuration set: confidence = {self.confidence}")
        logger.info(f"Model configuration set: iou = {self.iou}")
        logger.info(f"Model configuration set: image_size = {self.image_size}")
        logger.info(f"Model configuration set: device = {self.device}")
        logger.info(f"Model configuration set: classes = {self.classes}")

    def predict(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Detector is not loaded. Call warmup() first.")
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a NumPy array.")
        if frame.ndim != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError("Frame must have non-empty height and width.")

        with self._lock:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                device=self.device,
                classes=self.classes,
                verbose=False,
            )

        if not results:
            return []

        detections: list[dict[str, Any]] = []
        result = results[0]

        if result.boxes is None:
            return detections

        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": result.names[int(class_id)],
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    },
                }
            )

        return detections

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.model is not None,
            "model_path": str(self.model_path),
            "device": self.device,
            "confidence": self.confidence,
            "iou": self.iou,
            "image_size": self.image_size,
            "classes": self.classes,
        }

    def unload(self) -> None:
        with self._lock:
            self.model = None
        logger.info("YOLO detector unloaded")