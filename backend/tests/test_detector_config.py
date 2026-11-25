import pytest

from app.vision.detector import MODELS_DIR, get_detector_config


CONFIG_ENV_VARS = (
    "VISION_MODEL_PATH",
    "VISION_MODEL_DEVICE",
    "VISION_CONFIDENCE_THRESHOLD",
    "VISION_IOU_THRESHOLD",
    "VISION_MODEL_IMAGE_SIZE",
    "VISION_MODEL_CLASSES",
    "VISION_INFERENCE_FPS",
)


def clear_detector_config(monkeypatch):
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_detector_config_defaults_to_bundled_model_and_documented_values(monkeypatch):
    clear_detector_config(monkeypatch)

    assert get_detector_config() == {
        "model_path": str(MODELS_DIR / "yolo26n.pt"),
        "confidence": 0.4,
        "iou": 0.5,
        "image_size": 640,
        "device": "cpu",
        "classes": None,
        "inference_fps": 10.0,
    }


def test_detector_config_parses_environment_overrides(monkeypatch):
    clear_detector_config(monkeypatch)
    monkeypatch.setenv("VISION_MODEL_PATH", "custom.pt")
    monkeypatch.setenv("VISION_MODEL_DEVICE", " cuda:0 ")
    monkeypatch.setenv("VISION_CONFIDENCE_THRESHOLD", "0.65")
    monkeypatch.setenv("VISION_IOU_THRESHOLD", "0.72")
    monkeypatch.setenv("VISION_MODEL_IMAGE_SIZE", "960")
    monkeypatch.setenv("VISION_MODEL_CLASSES", "0, 2,7")
    monkeypatch.setenv("VISION_INFERENCE_FPS", "12.5")

    config = get_detector_config()

    assert config == {
        "model_path": "custom.pt",
        "confidence": 0.65,
        "iou": 0.72,
        "image_size": 960,
        "device": "cuda:0",
        "classes": [0, 2, 7],
        "inference_fps": 12.5,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VISION_CONFIDENCE_THRESHOLD", "1.1"),
        ("VISION_IOU_THRESHOLD", "-0.1"),
        ("VISION_MODEL_IMAGE_SIZE", "0"),
        ("VISION_MODEL_CLASSES", "0,,1"),
        ("VISION_MODEL_CLASSES", "-1"),
        ("VISION_MODEL_CLASSES", "1,1"),
        ("VISION_INFERENCE_FPS", "0"),
        ("VISION_INFERENCE_FPS", "120.1"),
    ],
)
def test_detector_config_rejects_invalid_environment_values(
    monkeypatch,
    name,
    value,
):
    clear_detector_config(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError):
        get_detector_config()
