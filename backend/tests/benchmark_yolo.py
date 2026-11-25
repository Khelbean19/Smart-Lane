"""Isolated YOLO26 benchmark and detection-video module.

Run from the repository root:
    uv run python backend/tests/benchmark_yolo.py
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from backend.app.vision.pipeline import preprocess_detection_frame


FOOTAGE_DIR = TESTS_DIR / "footage"
BENCHMARK_DIR = TESTS_DIR / "benchmarks"
MODEL_DIRS = (
    PROJECT_ROOT / "backend" / "models_weights",
    PROJECT_ROOT / "backend" / "model_weights",
)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}


@dataclass
class BenchmarkRow:
    model_size: str
    model_path: str
    footage: str
    aspect: str
    preprocessing_mode: str
    frame_count: int
    inference_fps: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    total_detections: int
    mean_detections_per_frame: float
    vehicle_detections: int
    mean_vehicle_confidence: float
    label_backed: bool
    precision_at_0_5: float | None = None
    recall_at_0_5: float | None = None
    ap_at_0_5: float | None = None
    labeled_frames: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class RunSummary:
    generated_at: str
    platform: str
    python: str
    device: str
    confidence: float
    iou: float
    image_size: int
    duration_seconds: float
    model_weights: list[dict[str, object]]
    footages: list[dict[str, object]]
    results: list[BenchmarkRow] = field(default_factory=list)


def discover_videos(footage_dir: Path = FOOTAGE_DIR) -> list[Path]:
    if not footage_dir.is_dir():
        return []
    return sorted(
        path
        for path in footage_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def discover_models(model_dirs: Iterable[Path] = MODEL_DIRS) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        for size in ("n", "s", "m"):
            candidates = (
                model_dir / f"yolo26{size}.pt",
                model_dir / f"yolo{size}.pt",
            )
            for candidate in candidates:
                if candidate.is_file() and size not in found:
                    found[size] = candidate
    return dict(found.items())


def _presentation_frame(
    frame: np.ndarray,
    aspect: str,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if aspect == "16:9":
        return cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)

    if aspect != "4:3":
        raise ValueError("Aspect must be '16:9' or '4:3'.")
    source_ratio = width / height
    target_ratio = 4.0 / 3.0
    if source_ratio >= target_ratio:
        crop_width = round(height * target_ratio)
        left = (width - crop_width) // 2
        cropped = frame[:, left:left + crop_width]
    else:
        crop_height = round(width / target_ratio)
        top = (height - crop_height) // 2
        cropped = frame[top:top + crop_height]
    return cv2.resize(cropped, (640, 480), interpolation=cv2.INTER_AREA)


def preprocess_benchmark_frame(
    frame: np.ndarray,
    aspect: str,
    preprocessing_mode: str,
) -> np.ndarray:
    presented = _presentation_frame(frame, aspect)
    roi_x = 0.125 if preprocessing_mode == "crop" else 0.0
    roi_y = 0.125 if preprocessing_mode == "crop" else 0.0
    roi_width = 0.75 if preprocessing_mode == "crop" else 1.0
    roi_height = 0.75 if preprocessing_mode == "crop" else 1.0
    settings = {
        "detection_width": 640,
        "detection_height": 480,
        "preprocessing_mode": preprocessing_mode,
        "roi_x": roi_x,
        "roi_y": roi_y,
        "roi_width": roi_width,
        "roi_height": roi_height,
    }
    return preprocess_detection_frame(presented, settings)


def _label_directories(video: Path) -> list[Path]:
    stem = video.stem
    return [
        video.parent / "labels" / stem,
        video.parent / f"{stem}_labels",
        video.parent / f"{stem}.labels",
        video.parent / stem,
    ]


def load_frame_labels(video: Path, frame_index: int) -> list[list[float]] | None:
    frame_name = f"{frame_index:06d}"
    candidates = []
    for label_dir in _label_directories(video):
        candidates.extend([
            label_dir / f"{frame_name}.txt",
            label_dir / f"{frame_index}.txt",
        ])

    for candidate in candidates:
        if not candidate.is_file():
            continue
        labels: list[list[float]] = []
        with candidate.open("r", encoding="utf-8") as label_file:
            for line in label_file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(part) for part in parts[:5]])
        return labels
    return None


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _evaluate_labels(
    detections: list[dict],
    labels: list[list[float]],
    width: int,
    height: int,
) -> tuple[int, int, int]:
    ground_truth = []
    for class_id, x_center, y_center, box_width, box_height in labels:
        cx = x_center * width
        cy = y_center * height
        bw = box_width * width
        bh = box_height * height
        ground_truth.append([
            cx - bw / 2,
            cy - bh / 2,
            cx + bw / 2,
            cy + bh / 2,
            int(class_id),
        ])

    matched = set()
    true_positives = 0
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        box = detection["bbox"]
        predicted_box = [
            box["x1"], box["y1"], box["x2"], box["y2"],
        ]
        best_index = -1
        best_iou = 0.0
        for index, truth in enumerate(ground_truth):
            if index in matched or truth[4] != detection["class_id"]:
                continue
            iou = _box_iou(predicted_box, truth[:4])
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_iou >= 0.5:
            matched.add(best_index)
            true_positives += 1

    false_positives = len(detections) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return true_positives, false_positives, false_negatives


def _draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        class_id = detection["class_id"]
        if class_id not in VEHICLE_CLASS_IDS:
            continue
        box = detection["bbox"]
        x1 = max(0, round(box["x1"]))
        y1 = max(0, round(box["y1"]))
        x2 = min(output.shape[1] - 1, round(box["x2"]))
        y2 = min(output.shape[0] - 1, round(box["y2"]))
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def benchmark_model(
    model: YOLO,
    model_size: str,
    model_path: Path,
    video_path: Path,
    aspect: str,
    preprocessing_mode: str,
    device: str,
    confidence: float,
    iou: float,
    duration_seconds: float,
    writer: cv2.VideoWriter | None = None,
) -> BenchmarkRow:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open footage: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = max(1, round(duration_seconds * source_fps))
    latencies: list[float] = []
    confidences: list[float] = []
    vehicle_confidences: list[float] = []
    total_detections = 0
    vehicle_detections = 0
    labeled_frames = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    frame_count = 0

    while frame_count < max_frames:
        ok, source_frame = capture.read()
        if not ok:
            break

        model_frame = preprocess_benchmark_frame(
            source_frame,
            aspect,
            preprocessing_mode,
        )
        started = time.perf_counter()
        results = model.predict(
            source=model_frame,
            conf=confidence,
            iou=iou,
            imgsz=640,
            device=device,
            verbose=False,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)

        detections: list[dict] = []
        if results:
            result = results[0]
            if result.boxes is not None:
                for box, detection_confidence, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                ):
                    class_number = int(class_id)
                    detections.append({
                        "class_id": class_number,
                        "class_name": result.names[class_number],
                        "confidence": float(detection_confidence),
                        "bbox": {
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                        },
                    })

        for detection in detections:
            confidences.append(detection["confidence"])
            if detection["class_id"] in VEHICLE_CLASS_IDS:
                vehicle_detections += 1
                vehicle_confidences.append(detection["confidence"])
        total_detections += len(detections)

        labels = load_frame_labels(video_path, frame_count)
        if labels is not None:
            labeled_frames += 1
            tp, fp, fn = _evaluate_labels(detections, labels, 640, 480)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        if writer is not None:
            writer.write(_draw_detections(model_frame, detections))
        frame_count += 1

    capture.release()

    mean_ms = statistics.fmean(latencies) if latencies else 0.0
    sorted_latencies = sorted(latencies)
    p95_index = max(0, round(0.95 * (len(sorted_latencies) - 1)))
    denominator = max(1, frame_count)
    precision = (
        total_tp / (total_tp + total_fp)
        if labeled_frames and (total_tp + total_fp) else 0.0
    )
    recall = (
        total_tp / (total_tp + total_fn)
        if labeled_frames and (total_tp + total_fn) else 0.0
    )
    ap = (precision * recall) if labeled_frames else None
    if labeled_frames == 0:
        precision = None
        recall = None

    return BenchmarkRow(
        model_size=model_size,
        model_path=str(model_path),
        footage=video_path.name,
        aspect=aspect,
        preprocessing_mode=preprocessing_mode,
        frame_count=frame_count,
        inference_fps=(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        mean_ms=mean_ms,
        median_ms=statistics.median(latencies) if latencies else 0.0,
        p95_ms=sorted_latencies[p95_index] if sorted_latencies else 0.0,
        min_ms=min(latencies) if latencies else 0.0,
        max_ms=max(latencies) if latencies else 0.0,
        total_detections=total_detections,
        mean_detections_per_frame=total_detections / denominator,
        vehicle_detections=vehicle_detections,
        mean_vehicle_confidence=(
            statistics.fmean(vehicle_confidences)
            if vehicle_confidences
            else 0.0
        ),
        label_backed=labeled_frames > 0,
        precision_at_0_5=precision,
        recall_at_0_5=recall,
        ap_at_0_5=ap,
        labeled_frames=labeled_frames,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
    )


def _video_writer(path: Path, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (640, 480),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create detection video: {path}")
    return writer


def format_report(summary: RunSummary) -> str:
    lines = [
        "YOLO26 Benchmark Report",
        "=" * 72,
        f"Generated: {summary.generated_at}",
        f"Platform: {summary.platform}",
        f"Python: {summary.python}",
        f"Device: {summary.device}",
        f"Confidence: {summary.confidence}",
        f"IoU: {summary.iou}",
        f"Inference size: {summary.image_size}x{summary.image_size}",
        f"Clip duration: {summary.duration_seconds:.1f} seconds",
        "",
        "Model weights:",
    ]
    if summary.model_weights:
        for model in summary.model_weights:
            lines.append(
                f"- {model['size']}: {model['path']} "
                f"({model['size_bytes']} bytes)"
            )
    else:
        lines.append("- No nano/small/medium weights found.")

    lines.extend(["", "Footage:"])
    if summary.footages:
        for footage in summary.footages:
            lines.append(
                f"- {footage['name']}: {footage['width']}x{footage['height']} "
                f"at {footage['fps']:.2f} FPS, "
                f"{footage['frame_count']} frames"
            )
    else:
        lines.append("- No footage found.")

    lines.extend(["", "Results:", "-" * 72])
    if not summary.results:
        lines.append("No benchmark runs were completed.")
    for row in summary.results:
        accuracy = (
            f"Precision@0.5={row.precision_at_0_5:.4f}, "
            f"Recall@0.5={row.recall_at_0_5:.4f}, "
            f"AP@0.5={row.ap_at_0_5:.4f}"
            if row.label_backed
            else "Accuracy unavailable (no matching frame labels)"
        )
        lines.extend([
            (
                f"{row.model_size.upper()} | {row.footage} | {row.aspect} | "
                f"{row.preprocessing_mode}"
            ),
            f"  Model: {row.model_path}",
            f"  Frames: {row.frame_count}",
            (
                f"  Speed: {row.inference_fps:.2f} FPS | "
                f"mean {row.mean_ms:.2f} ms | median {row.median_ms:.2f} ms | "
                f"p95 {row.p95_ms:.2f} ms"
            ),
            f"  Latency range: {row.min_ms:.2f}-{row.max_ms:.2f} ms",
            (
                f"  Detections: {row.total_detections} total | "
                f"{row.mean_detections_per_frame:.3f}/frame | "
                f"{row.vehicle_detections} vehicle detections"
            ),
            f"  Mean vehicle confidence: {row.mean_vehicle_confidence:.4f}",
            f"  Accuracy: {accuracy}",
            (
                f"  Label counts: TP={row.true_positives}, "
                f"FP={row.false_positives}, FN={row.false_negatives}"
            )
            if row.label_backed
            else "  Label counts: no labeled frames found",
            "",
        ])
    return "\n".join(lines) + "\n"


def run_benchmark(args: argparse.Namespace) -> RunSummary:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    models = discover_models()
    selected_sizes = {"n", "s", "m"} if not args.model_size else set(args.model_size)
    models = {
        size: path
        for size, path in models.items()
        if size in selected_sizes
    }
    footages = discover_videos()
    if args.video:
        footages = [path for path in footages if path.stem in args.video]

    footage_stats = []
    for footage in footages:
        capture = cv2.VideoCapture(str(footage))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        footage_stats.append({
            "name": footage.name,
            "path": str(footage),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        })

    summary = RunSummary(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        platform=platform.platform(),
        python=platform.python_version(),
        device=args.device,
        confidence=args.confidence,
        iou=args.iou,
        image_size=640,
        duration_seconds=args.duration,
        model_weights=[
            {
                "size": size,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }
            for size, path in models.items()
        ],
        footages=footage_stats,
    )

    for size, model_path in models.items():
        print(f"Loading model yolo26{size}: {model_path}")
        model = YOLO(str(model_path))
        model.predict(
            source=np.zeros((480, 640, 3), dtype=np.uint8),
            conf=args.confidence,
            iou=args.iou,
            imgsz=640,
            device=args.device,
            verbose=False,
        )

        for footage in footages:
            capture = cv2.VideoCapture(str(footage))
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            capture.release()
            for aspect in args.aspect:
                for mode in args.mode:
                    print(
                        f"  Running {footage.name} | {aspect} | {mode}"
                    )
                    video_dir = BENCHMARK_DIR / "detection_videos"
                    video_name = (
                        f"{footage.stem}_yolo26{size}_{aspect.replace(':', 'x')}_{mode}.mp4"
                    )
                    writer = (
                        None
                        if args.skip_videos
                        else _video_writer(video_dir / video_name, source_fps)
                    )
                    try:
                        row = benchmark_model(
                            model=model,
                            model_size=size,
                            model_path=model_path,
                            video_path=footage,
                            aspect=aspect,
                            preprocessing_mode=mode,
                            device=args.device,
                            confidence=args.confidence,
                            iou=args.iou,
                            duration_seconds=args.duration,
                            writer=writer,
                        )
                    finally:
                        if writer is not None:
                            writer.release()
                    summary.results.append(row)

    report_path = BENCHMARK_DIR / args.report
    report_path.write_text(format_report(summary), encoding="utf-8")

    json_path = report_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                **{key: value for key, value in asdict(summary).items()
                   if key != "results"},
                "results": [asdict(row) for row in summary.results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Text report: {report_path}")
    print(f"JSON report: {json_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO26 nano, small, and medium weights."
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-size",
        nargs="+",
        choices=("n", "s", "m"),
        default=None,
    )
    parser.add_argument("--video", nargs="+", default=None)
    parser.add_argument(
        "--aspect",
        nargs="+",
        choices=("16:9", "4:3"),
        default=["16:9", "4:3"],
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=("fit", "crop"),
        default=["fit", "crop"],
    )
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--report", default="benchmark_results.txt")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than 0")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1")
    if not 0 < args.iou <= 1:
        parser.error("--iou must be greater than 0 and at most 1")

    try:
        summary = run_benchmark(args)
    except RuntimeError as error:
        parser.error(str(error))
        return 2

    if not summary.model_weights:
        print("No YOLO26 nano/small/medium weights were found.")
    if not summary.footages:
        print(f"No supported footage was found in {FOOTAGE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
