import numpy as np
import pytest
import torch
from model_loader import BaseDetector, ModelPipeline


class ScriptedDetector(BaseDetector):
    def __init__(self, key='fake', weight=1):
        super().__init__(key, dict(confidence_threshold=.15, weight=weight), torch.device('cpu'))
        self.x = 10
        self.score = .8
        self.panes = None
        self.calls = 0

    def run(self, img, results, boxes):
        pane = int(img[0, 0, 0])
        self.calls += 1
        if self.panes is None or pane in self.panes:
            results.append(self.score * self.weight)
            boxes.append(([self.x, 10, self.x + 30, 60], self.key))


def make_pipeline(**options):
    cfg = dict(rows=3, columns=3, min_movement_frames=2, movement_pixels=1,
               movement_box_fraction=0, score_multiplier=1.5)
    cfg.update(options)
    pipeline = ModelPipeline({}, torch.device('cpu'), tracking_config=cfg)
    pipeline.detectors = [ScriptedDetector()]
    frame = np.zeros((300, 300, 3), np.uint8)
    for i in range(9):
        frame[i // 3 * 100:(i // 3 + 1) * 100, i % 3 * 100:(i % 3 + 1) * 100] = i
    return pipeline, frame


@pytest.mark.parametrize('mode', ['sequential', 'parallel'])
def test_nine_panes_boost_once_per_model_and_keep_coordinates_local(mode):
    pipeline, frame = make_pipeline()
    pipeline.detectors.append(ScriptedDetector('second', .5))
    for x in (10, 13, 16):
        for d in pipeline.detectors:
            d.x = x
        scores, boxes = pipeline.run_inference(frame, mode)
        if x < 16:
            assert scores == [] and boxes == []
    assert scores == pytest.approx([1.2, .6])
    assert len(boxes) == 18
    assert len(pipeline.trackers) == 18
    assert boxes[-1][0] == [216, 210, 246, 260]
    for tracker in pipeline.trackers.values():
        assert len(tracker.tracks[0].history) == 3
        assert tracker.tracks[0].history[0][1] == (10, 10, 40, 60)
    pipeline.draw_trajectories(frame.copy())


def test_models_in_unrelated_panes_do_not_combine_votes():
    pipeline, frame = make_pipeline()
    pipeline.detectors[0].panes = {0}
    second = ScriptedDetector('other')
    second.panes = {8}
    pipeline.detectors.append(second)
    for x in (10, 13, 16):
        for d in pipeline.detectors:
            d.x = x
        scores, _ = pipeline.run_inference(frame)
    assert sum(scores) == pytest.approx(1.2)


def test_pane_change_needs_new_history():
    pipeline, frame = make_pipeline()
    detector = pipeline.detectors[0]
    detector.panes = {0}
    for x in (10, 13, 16):
        detector.x = x
        pipeline.run_inference(frame)
    detector.panes = {1}
    assert pipeline.run_inference(frame)[0] == []
    assert pipeline.trackers[(1, 'fake')].tracks[0].movement_frames == 0


def test_zero_disables_grid_history_and_boost():
    pipeline, frame = make_pipeline(min_movement_frames=0)
    scores, boxes = pipeline.run_inference(frame)
    assert scores == [.8] and len(boxes) == 1
    assert pipeline.trackers == {}
    assert pipeline.detectors[0].calls == 1
    assert pipeline.detectors[0].candidate_threshold == .15


def test_low_confidence_candidates_can_extend_and_boost():
    pipeline, frame = make_pipeline()
    detector = pipeline.detectors[0]
    pipeline.run_inference(frame)
    detector.score = .2
    for x in (13, 16):
        detector.x = x
        scores, _ = pipeline.run_inference(frame)
    assert scores == pytest.approx([.3])
    detector.x = 19
    detector.score = .12
    assert pipeline.run_inference(frame)[0] == []  # Below model's scoring floor


def test_resize_and_idle_gap_reset_history():
    pipeline, frame = make_pipeline()
    detector = pipeline.detectors[0]
    for x in (10, 13, 16):
        detector.x = x
        pipeline.run_inference(frame)
    pipeline._last_frame_time -= 10
    assert pipeline.run_inference(frame)[0] == []
    assert pipeline.run_inference(frame[:270])[0] == []
    pipeline.reset_tracking()
    assert pipeline.trackers == {}


def test_parallel_detector_failure_propagates():
    pipeline, frame = make_pipeline()
    def fail(*args):
        raise RuntimeError('inference failed')
    pipeline.detectors[0].run = fail
    with pytest.raises(RuntimeError, match='inference failed'):
        pipeline.run_inference(frame, 'parallel')


def test_crowd_does_not_multiply_model_weight():
    pipeline, frame = make_pipeline()
    detector = pipeline.detectors[0]
    original = detector.run
    def crowd(img, scores, boxes):
        original(img, scores, boxes)
        scores.append(.7)
        boxes.append(([detector.x + 40, 10, detector.x + 65, 60], detector.key))
    detector.run = crowd
    for x in (10, 13, 16):
        detector.x = x
        scores, boxes = pipeline.run_inference(frame)
    assert scores == pytest.approx([1.2])
    assert len(boxes) == 18


def test_invalid_candidates_are_dropped_and_history_is_clipped():
    pipeline, frame = make_pipeline()
    detector = pipeline.detectors[0]
    def invalid(img, scores, boxes):
        for box in ([float('nan'), 0, 20, 20], [30, 0, 10, 20], [-10, -10, 110, 110]):
            scores.append(.8)
            boxes.append((box, detector.key))
    detector.run = invalid
    pipeline.run_inference(frame)
    for tracker in pipeline.trackers.values():
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].history[0][1] == (0, 0, 100, 100)
