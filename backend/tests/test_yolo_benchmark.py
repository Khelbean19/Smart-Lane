from pathlib import Path

import cv2
import numpy as np

import sys

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from benchmark_yolo import (
    BenchmarkRow,
    RunSummary,
    discover_models,
    discover_videos,
    format_report,
    preprocess_benchmark_frame,
)


def test_benchmark_asset_discovery_finds_footage_and_weights():
    assert discover_videos()[0].suffix == ".mp4"
    models = discover_models()
    assert "n" in models
    assert all(path.is_file() for path in models.values())


def test_preprocessing_matrix_preserves_640_by_480_model_shape():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for aspect in ("16:9", "4:3"):
        for mode in ("fit", "crop"):
            model_frame = preprocess_benchmark_frame(frame, aspect, mode)
            assert model_frame.shape == (480, 640, 3)


def test_text_report_marks_missing_labels_as_unavailable():
    row = BenchmarkRow(
        model_size="n",
        model_path="yolo26n.pt",
        footage="sample.mp4",
        aspect="16:9",
        preprocessing_mode="fit",
        frame_count=300,
        inference_fps=20.0,
        mean_ms=50.0,
        median_ms=49.0,
        p95_ms=60.0,
        min_ms=40.0,
        max_ms=70.0,
        total_detections=10,
        mean_detections_per_frame=0.033,
        vehicle_detections=8,
        mean_vehicle_confidence=0.75,
        label_backed=False,
    )
    summary = RunSummary(
        generated_at="2026-01-01T00:00:00",
        platform="test",
        python="3",
        device="cpu",
        confidence=0.4,
        iou=0.5,
        image_size=640,
        duration_seconds=10.0,
        model_weights=[],
        footages=[],
        results=[row],
    )
    report = format_report(summary)
    assert "Accuracy unavailable" in report
    assert "FPS" in report
