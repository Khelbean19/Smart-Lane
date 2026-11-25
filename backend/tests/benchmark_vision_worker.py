"""Benchmark concurrent camera capture and inference on the local host.

Run from the repository root with:
``uv run python -m backend.tests.benchmark_vision_worker --source 0 --source 1``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from backend.app.vision.camera import CameraManager
from backend.app.vision.detector import YOLODetector, get_detector_config
from backend.app.vision.inference import CameraInferenceRunner


class _LoopingVideoCapture:
    def __init__(self, source: str) -> None:
        self._capture = cv2.VideoCapture(source)
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._frame_interval = 1 / fps if fps > 0 else 0.0
        self._next_frame_at = time.monotonic()

    def isOpened(self) -> bool:
        return self._capture.isOpened()

    def set(self, property_id: int, value: float) -> bool:
        return self._capture.set(property_id, value)

    def read(self) -> tuple[bool, np.ndarray | None]:
        delay = self._next_frame_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        success, frame = self._capture.read()
        if not success:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self._capture.read()
        self._next_frame_at = time.monotonic() + self._frame_interval
        return success, frame

    def release(self) -> None:
        self._capture.release()


def _capture_factory(source: int | str) -> Any:
    if isinstance(source, str) and Path(source).is_file():
        return _LoopingVideoCapture(source)
    return cv2.VideoCapture(source)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure concurrent camera capture and inference throughput."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Camera index, stream URL, or local video file; local files loop at "
            "their recorded frame rate. Pass once per camera."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Measurement duration in seconds (default: 30).",
    )
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error("--duration must be greater than zero.")
    return args


def run_benchmark(sources: list[str], duration: float) -> dict[str, object]:
    config = get_detector_config()
    detector = YOLODetector(
        model_path=config["model_path"],
        confidence=config["confidence"],
        iou=config["iou"],
        image_size=config["image_size"],
        device=config["device"],
        classes=config["classes"],
    )
    manager = CameraManager(capture_factory=_capture_factory)
    runner = CameraInferenceRunner(detector, config["inference_fps"])
    camera_ids = list(range(1, len(sources) + 1))

    try:
        detector.warmup()
        camera_statuses = manager.reconcile(
            [
                {"camera_id": camera_id, "source": source}
                for camera_id, source in zip(camera_ids, sources)
            ],
            camera_ids,
            retry_failed=True,
        )
        failed_cameras = [
            status for status in camera_statuses if status["state"] != "running"
        ]
        if failed_cameras:
            raise RuntimeError(
                "Could not start every camera: "
                + json.dumps(failed_cameras, sort_keys=True)
            )

        cameras = {camera_id: manager.get(camera_id) for camera_id in camera_ids}
        runner.sync(set(camera_ids), cameras)
        initial_sequences = {
            camera_id: cameras[camera_id].status()["frame_sequence"]
            for camera_id in camera_ids
        }
        initial_inference = runner.status()
        initial_counts = {
            camera_id: initial_inference["camera_results"][str(camera_id)][
                "inferences_completed"
            ]
            for camera_id in camera_ids
        }

        started = time.monotonic()
        time.sleep(duration)
        elapsed = time.monotonic() - started

        final_status = runner.status()
        final_sequences = {
            camera_id: cameras[camera_id].status()["frame_sequence"]
            for camera_id in camera_ids
        }
        final_camera_statuses = {
            camera_id: cameras[camera_id].status() for camera_id in camera_ids
        }
        camera_results = final_status["camera_results"]
        rows = {}
        for camera_id in camera_ids:
            result = camera_results[str(camera_id)]
            camera_status = final_camera_statuses[camera_id]
            completed = result["inferences_completed"] - initial_counts[camera_id]
            rows[str(camera_id)] = {
                "status": result["status"],
                "camera_state": camera_status["state"],
                "camera_error": camera_status["error"],
                "capture_fps": max(
                    0,
                    final_sequences[camera_id] - initial_sequences[camera_id],
                )
                / elapsed,
                "target_inference_fps": result["inference_fps"],
                "actual_inference_fps": result["actual_inference_fps"],
                "window_inference_fps": completed / elapsed,
                "inferences_completed": completed,
                "inference_duration_ms": result["inference_duration_ms"],
            }

        total_completed = sum(
            row["inferences_completed"] for row in rows.values()
        )
        return {
            "model_path": config["model_path"],
            "device": config["device"],
            "image_size": config["image_size"],
            "target_inference_fps_per_camera": config["inference_fps"],
            "measurement_seconds": elapsed,
            "aggregate_window_inference_fps": total_completed / elapsed,
            "rolling_aggregate_inference_fps": final_status[
                "actual_inference_fps"
            ],
            "cameras": rows,
        }
    finally:
        runner.stop_all()
        try:
            manager.stop_all()
        finally:
            detector.unload()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_benchmark(args.source, args.duration), indent=2))


if __name__ == "__main__":
    main()
