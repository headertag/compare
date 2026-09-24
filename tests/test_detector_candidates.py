"""Every backend must expose all people and pass the tracking candidate floor."""
from unittest.mock import MagicMock
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from model_loader import (BaseDetector, DetrDetector, RfDetrDetector, YolosDetector,
                          TorchvisionDetector, YOLOv5Detector, UltralyticsDetector)


@pytest.mark.parametrize('kind', [DetrDetector, RfDetrDetector, YolosDetector,
                                  TorchvisionDetector, YOLOv5Detector, UltralyticsDetector])
@pytest.mark.parametrize('tracking', [False, True])
def test_all_people_exposed_only_when_tracking(kind, tracking):
    detector = kind.__new__(kind)
    BaseDetector.__init__(detector, 'test', dict(confidence_threshold=.1), torch.device('cpu'))
    detector.collect_all = tracking
    detector.candidate_threshold = .05
    detector.model = MagicMock()
    boxes = torch.tensor([[10., 10., 30., 60.], [40., 10., 60., 60.]])
    scores = torch.tensor([.8, .2])
    results = dict(boxes=boxes, scores=scores, labels=torch.tensor([1, 1]))
    if kind in (DetrDetector, RfDetrDetector, YolosDetector):
        detector.processor = MagicMock()
        detector.processor.post_process_object_detection.return_value = [results]
        detector.model.config.id2label = {1: 'person'}
    elif kind is TorchvisionDetector:
        detector.model.return_value = [results]
    elif kind is YOLOv5Detector:
        detector.model.return_value.pandas.return_value.xyxy = [MagicMock()]
        detector.model.return_value.pandas.return_value.xyxy[0].to_numpy.return_value = np.array(
            [[10, 10, 30, 60, .8, 0, 'person'], [40, 10, 60, 60, .2, 0, 'person']], dtype=object)
    else:
        detector.device_str = 'cpu'
        result = MagicMock()
        result.names = {0: 'person'}
        result.boxes = [SimpleNamespace(conf=score[None], cls=torch.tensor([0]), xyxy=box[None])
                        for score, box in zip(scores, boxes)]
        detector.model.return_value = [result]
    found_scores, found_boxes = [], []
    detector.run(np.zeros((100, 100, 3), np.uint8), found_scores, found_boxes)
    assert len(found_boxes) == len(found_scores) == (2 if tracking else 1)
    if kind in (DetrDetector, RfDetrDetector, YolosDetector):
        assert detector.processor.post_process_object_detection.call_args.kwargs['threshold'] == .05
    elif kind is UltralyticsDetector:
        assert detector.model.call_args.kwargs['conf'] == .05
    elif kind is YOLOv5Detector:
        assert detector.model.conf == .05
    elif kind is TorchvisionDetector:
        assert detector.model.roi_heads.score_thresh == .05
