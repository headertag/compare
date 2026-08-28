import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from model_loader import (
    BaseDetector,
    DetrDetector,
    YolosDetector,
    TorchvisionDetector,
    YOLOv5Detector,
    create_detector,
    ModelPipeline,
)

def test_base_detector_init():
    config = {
        "name": "test-model",
        "confidence_threshold": 0.75,
        "weight": 0.8,
        "color": [10, 20, 30],
    }
    detector = BaseDetector("test_key", config, torch.device("cpu"))
    assert detector.key == "test_key"
    assert detector.name == "test-model"
    assert detector.confidence_threshold == 0.75
    assert detector.weight == 0.8
    assert detector.color == (10, 20, 30)

def test_model_pipeline_disabled_models():
    mock_config = {
        "model_a": {
            "type": "detr",
            "name": "mock-detr",
            "enabled": False,
        },
        "model_b": {
            "type": "yolos",
            "name": "mock-yolos",
            "enabled": False,
        }
    }
    pipeline = ModelPipeline(models_config=mock_config, device=torch.device("cpu"))
    assert len(pipeline.detectors) == 0
    results, multi_box = pipeline.run_inference(np.zeros((100, 100, 3), dtype=np.uint8))
    assert results == []
    assert multi_box == []

def test_model_pipeline_custom_detector_execution():
    class MockDetector(BaseDetector):
        def run(self, img, results_list, box_list):
            results_list.append(0.95 * self.weight)
            box_list.append(([10, 20, 30, 40], self.key))

    pipeline = ModelPipeline(models_config={}, device=torch.device("cpu"))
    pipeline.detectors = [
        MockDetector("mock1", {"confidence_threshold": 0.5, "weight": 1.0, "color": [255, 0, 0]}, torch.device("cpu")),
        MockDetector("mock2", {"confidence_threshold": 0.5, "weight": 0.5, "color": [0, 255, 0]}, torch.device("cpu")),
    ]

    # Test sequential execution
    results, multi_box = pipeline.run_inference(np.zeros((100, 100, 3), dtype=np.uint8), execution_mode="sequential")
    assert len(results) == 2
    assert results[0] == pytest.approx(0.95)
    assert results[1] == pytest.approx(0.475)
    assert len(multi_box) == 2

    # Test parallel execution
    results_p, multi_box_p = pipeline.run_inference(np.zeros((100, 100, 3), dtype=np.uint8), execution_mode="parallel")
    assert len(results_p) == 2
    assert len(multi_box_p) == 2

    # Test colors mapping
    colors = pipeline.get_model_colors()
    assert colors["mock1"] == (255, 0, 0)
    assert colors["mock2"] == (0, 255, 0)
