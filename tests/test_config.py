import pytest
import yaml
from config import load_config, get_device
import torch

@pytest.fixture
def mock_config_file(tmp_path):
    """Creates a temporary config file for testing."""
    config_data = {
        "telegram": {
            "token": "test_token",
            "chat_ids": [12345],
        },
        "alerting": {
            "cooldown_seconds": 60,
            "cooldown_threshold_multiplier": 2,
            "min_interval_seconds": 10,
            "sensitivity_threshold": 2.0,
        },
        "camera": {
            "index": 0,
            "width": 1280,
            "height": 720,
        },
        "processing": {
            "device": "cpu",
            "coco_classes_path": "coco_classes.pickle",
        },
        "models": {
            "detr_resnet_50": {"name": "facebook/detr-resnet-50", "confidence_threshold": 0.9},
            "yolos_tiny": {"name": "hustvl/yolos-tiny", "confidence_threshold": 0.9},
            "frcnn_resnet": {"name": "frcnn-resnet", "confidence_threshold": 0.8},
            "retinanet": {"name": "retinanet", "confidence_threshold": 0.6},
            "yolov5s": {"name": "yolov5s", "confidence_threshold": 0.6},
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path

def test_load_config(mock_config_file):
    """Tests that the configuration is loaded correctly."""
    config = load_config(mock_config_file)
    assert config["telegram"]["token"] == "test_token"
    assert config["camera"]["width"] == 1280

def test_get_device_cpu():
    """Tests that the device is set to CPU when specified."""
    device = get_device("cpu")
    assert device.type == "cpu"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_device_cuda():
    """Tests that the device is set to CUDA when specified and available."""
    device = get_device("cuda")
    assert device.type == "cuda"

def test_get_device_cuda_fallback():
    """Tests that the device falls back to CPU when CUDA is specified but not available."""
    if not torch.cuda.is_available():
        device = get_device("cuda")
        assert device.type == "cpu"
